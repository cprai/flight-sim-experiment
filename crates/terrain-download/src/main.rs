//! Fetches Canadian high-resolution elevation data for a longitude/latitude box
//! and writes it as a GeoTIFF the simulator can load.
//!
//! The data is Natural Resources Canada's HRDEM mosaic, published as
//! Cloud-Optimized GeoTIFFs on a Lambert grid in metres. A single mosaic block
//! runs to 142 GiB, so nothing is downloaded whole: the STAC catalogue says
//! which blocks overlap the box, the COG headers say which of their tiles hold
//! data, and only those tiles are fetched.
//!
//! One-metre data is preferred and two-metre fills its gaps, because the
//! one-metre mosaic only covers ground a LiDAR survey delivered at that
//! resolution. In practice the two-metre fill is small -- across two whole
//! 500 km blocks it added 9 km2 and 1 km2 -- because both products derive from
//! the same surveys. The percentages printed at the end say exactly how much
//! each contributed.
//!
//! Colour comes from somewhere else entirely: annual cloud-free Sentinel-2
//! composites, which are already mosaicked and need no scene picking or cloud
//! masking here. They are published on a different grid at a different
//! resolution, so the two products share the requested box and the code that
//! places pixels in it, but little else.

mod bbox;
mod coverage;
mod project;
mod resample;
mod source;
mod stac;
mod write;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use bbox::{LatLon, LatLonBox, OutputGrid};
use project::Projector;
use resample::Canvas;
use source::{RasterSpec, SourceRaster, Window};
use stac::{Product, Resolution};
use write::Provenance;

/// Elevation is sampled at the finest resolution HRDEM offers.
const ELEVATION_METRES_PER_PIXEL: f64 = 1.0;

/// Colour is sampled at the mosaics' own resolution rather than the elevation's.
///
/// Their pixels are 19.1 m of Web Mercator, which is about 12.4 m of ground at
/// fifty degrees north; ten metres keeps a little headroom without pretending
/// to detail that is not there. The two products deliberately do not match:
/// `is_co_registered_with` compares the ground each covers, not their pixel
/// counts, so a coarse colour raster pairs with a fine elevation one.
const COLOUR_METRES_PER_PIXEL: f64 = 10.0;

#[derive(Parser, Debug)]
#[command(about = "Fetch HRDEM elevation for a bounding box as a GeoTIFF", long_about = None)]
struct Arguments {
    /// One corner of the box, as `lat,lon` in degrees.
    ///
    /// Hyphens are allowed through so a southern latitude, or a pair given the
    /// wrong way round, reaches the parser and gets told what is wrong --
    /// otherwise clap reads the leading minus as an unknown flag.
    #[arg(long, value_name = "LAT,LON", allow_hyphen_values = true)]
    from: LatLon,

    /// The opposite corner, as `lat,lon` in degrees. Either diagonal works.
    #[arg(long, value_name = "LAT,LON", allow_hyphen_values = true)]
    to: LatLon,

    /// Where to write the GeoTIFF.
    #[arg(short, long, value_name = "PATH")]
    output: PathBuf,

    /// What to fetch: bare ground, the top of what the sensor saw, or colour.
    #[arg(long, value_enum, default_value = "dtm")]
    product: Product,

    /// Which year's cloud-free imagery to use, for `--product albedo`.
    #[arg(long, default_value_t = 2023, value_name = "YEAR")]
    imagery_year: u16,

    /// Root of the Earth Search API, used to locate imagery tiles.
    #[arg(
        long,
        default_value = "https://earth-search.aws.element84.com/v1",
        value_name = "URL"
    )]
    earth_search_root: String,

    /// Proceed without asking, however little of the box is covered.
    #[arg(short = 'y', long)]
    yes: bool,

    /// Fraction of the box that may be missing before confirmation is required.
    #[arg(long, default_value_t = 0.2, value_name = "FRACTION")]
    prompt_threshold: f64,

    /// Refuse boxes larger than this many output pixels.
    #[arg(long, default_value_t = 400_000_000, value_name = "COUNT")]
    max_pixels: u64,

    /// How many tiles to request at once.
    #[arg(long, default_value_t = 16, value_name = "COUNT")]
    concurrency: usize,

    /// Root of the STAC API to search.
    #[arg(
        long,
        default_value = "https://datacube.services.geo.ca/stac/api",
        value_name = "URL"
    )]
    stac_root: String,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,terrain_download=info"),
    )
    .init();

    let arguments = Arguments::parse();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("starting the async runtime")?
        .block_on(run(arguments))
}

async fn run(arguments: Arguments) -> Result<()> {
    anyhow::ensure!(
        (0.0..=1.0).contains(&arguments.prompt_threshold),
        "--prompt-threshold is a fraction between 0 and 1, not {}",
        arguments.prompt_threshold
    );

    let box_ = LatLonBox::from_corners(arguments.from, arguments.to)?;
    let metres = if arguments.product.is_elevation() {
        ELEVATION_METRES_PER_PIXEL
    } else {
        COLOUR_METRES_PER_PIXEL
    };
    let grid = OutputGrid::cover(box_, metres)?;

    anyhow::ensure!(
        grid.pixel_count() <= arguments.max_pixels,
        "the box needs {} x {} = {} pixels at {metres} m, over the --max-pixels \
         limit of {}; about {:.3} degrees on its shorter side would fit",
        grid.width,
        grid.height,
        grid.pixel_count(),
        arguments.max_pixels,
        fitting_degrees(&grid, arguments.max_pixels)
    );

    log::info!(
        "{} x {} pixels at {metres} m covering {:.6},{:.6} to {:.6},{:.6}",
        grid.width,
        grid.height,
        box_.south,
        box_.west,
        box_.north,
        box_.east
    );

    let client = reqwest::Client::builder()
        .user_agent(concat!("terrain-download/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("building the HTTP client")?;

    if arguments.product == Product::Albedo {
        return fetch_albedo(&arguments, &client, box_, &grid).await;
    }

    let projector = Projector::new(project::EPSG_LAMBERT)?;

    // Both mosaics are opened up front: the coverage estimate has to know what
    // the fallback could contribute before anything is downloaded.
    let mut rasters: Vec<Vec<SourceRaster>> = Vec::new();
    for resolution in Resolution::ALL {
        let items = stac::find_items(
            &client,
            &arguments.stac_root,
            resolution,
            arguments.product,
            box_,
        )
        .await?;
        log::info!(
            "{} {} item(s) overlap the box",
            items.len(),
            resolution.label()
        );

        let mut opened = Vec::new();
        for item in items {
            let spec = RasterSpec {
                epsg: u32::from(project::EPSG_LAMBERT),
                metres_per_pixel: resolution.metres(),
                bands: 1,
                fallback_nodata: ELEVATION_NODATA,
                empty_tile_limit: source::ELEVATION_EMPTY_TILE_LIMIT,
            };
            opened.push(SourceRaster::open(&client, item, spec).await?);
        }
        rasters.push(opened);
    }
    let coarse = rasters.pop().expect("both resolutions were searched");
    let fine = rasters.pop().expect("both resolutions were searched");

    anyhow::ensure!(
        !fine.is_empty() || !coarse.is_empty(),
        "no published {} raster covers that box; HRDEM only exists over surveyed \
         areas of Canada",
        arguments.product.label()
    );

    // Every raster declares -32767, but reading it rather than assuming keeps
    // the output honest if that ever changes.
    let nodata = fine
        .first()
        .or_else(|| coarse.first())
        .map(SourceRaster::nodata)
        .expect("at least one raster was just checked for");

    let mut estimate = coverage::estimate(&grid, &projector, &fine, &coarse)?;

    // Sized now rather than later, so the estimate can also report the download.
    let fine_extent = resample::projected_extent(&grid, &projector, Resolution::OneMetre.metres())?;
    let mut fine_window = Window::covering(
        fine_extent.min_x,
        fine_extent.min_y,
        fine_extent.max_x,
        fine_extent.max_y,
        Resolution::OneMetre.metres(),
        1,
        nodata,
    )?;
    estimate.bytes = fine.iter().map(|r| r.bytes_for(&fine_window)).sum();

    // Deliberately phrased as bounds. The estimate judges whole 512 m tiles, so
    // a tile with one valid pixel counts the same as a full one; over ground at
    // the edge of a survey it can read 16% where the truth is 0.6%.
    let (one, two, none) = estimate.percentages();
    println!(
        "Estimated coverage, judged per 512 m tile: at most {one:.1}% at 1 m and \
         {two:.1}% at 2 m, at least {none:.1}% missing"
    );
    println!(
        "Estimated download: {} of 1 m tiles",
        coverage::describe_bytes(estimate.bytes)
    );

    if !coverage::confirm(&estimate, arguments.prompt_threshold, arguments.yes)? {
        println!("Nothing downloaded.");
        return Ok(());
    }

    let mut canvas = Canvas::new(&grid, 1, nodata)?;
    let mut downloaded = 0;

    for raster in &fine {
        downloaded += raster.fill(&mut fine_window, arguments.concurrency).await?;
    }
    if !fine.is_empty() {
        let filled = canvas.fill_from(&grid, &projector, &fine_window, Provenance::OneMetre)?;
        log::info!("1 m data covered {filled} pixels");
    }
    drop(fine_window);

    // Two-metre tiles are fetched only for ground the one-metre pass could not
    // fill, which is usually none of it.
    if canvas.has_holes() && !coarse.is_empty() {
        let coarse_metres = Resolution::TwoMetre.metres();
        if let Some(holes) = canvas.hole_extent(&grid, &projector, coarse_metres)? {
            let mut coarse_window = Window::covering(
                holes.min_x,
                holes.min_y,
                holes.max_x,
                holes.max_y,
                coarse_metres,
                1,
                nodata,
            )?;
            for raster in &coarse {
                downloaded += raster
                    .fill(&mut coarse_window, arguments.concurrency)
                    .await?;
            }
            let filled =
                canvas.fill_from(&grid, &projector, &coarse_window, Provenance::TwoMetre)?;
            log::info!("2 m data covered a further {filled} pixels");
        }
    }

    write::write_geotiff(
        &arguments.output,
        &grid,
        &canvas.with_provenance_band(),
        nodata,
    )?;

    let tally = canvas.tally();
    let (one, two, none) = tally.percentages();
    println!("Wrote {}", arguments.output.display());
    println!(
        "  {} x {} pixels, {} downloaded",
        grid.width,
        grid.height,
        coverage::describe_bytes(downloaded)
    );
    println!("  {one:.2}% from 1 m data, {two:.2}% from 2 m data, {none:.2}% no data");

    Ok(())
}

/// The value HRDEM uses for ground it has no measurement of.
const ELEVATION_NODATA: f32 = -32767.0;

/// Fetches cloud-free Sentinel-2 colour for the box.
///
/// Much shorter than the elevation path because the compositing has already
/// been done upstream: there is one mosaic per grid square, no second
/// resolution to fall back to, and no cloud to reason about.
async fn fetch_albedo(
    arguments: &Arguments,
    client: &reqwest::Client,
    box_: LatLonBox,
    grid: &OutputGrid,
) -> Result<()> {
    anyhow::ensure!(
        stac::MOSAIC_YEARS.contains(&arguments.imagery_year),
        "only {:?} have published mosaics, not {}",
        stac::MOSAIC_YEARS,
        arguments.imagery_year
    );

    let projector = Projector::new(project::EPSG_WEB_MERCATOR)?;
    let tiles = stac::find_mosaic_tiles(client, &arguments.earth_search_root, box_).await?;
    log::info!("imagery grid squares: {}", tiles.join(", "));

    // The mosaics are Web Mercator, whose pixels are metres of projection
    // rather than metres of ground -- 19.1 of them is about 12.4 m of ground
    // at this latitude. The size is asserted rather than read, so a change of
    // zoom level upstream fails loudly instead of silently mis-placing pixels.
    let mut rasters = Vec::new();
    for tile in &tiles {
        let href = stac::mosaic_href(tile, arguments.imagery_year);
        let item = stac::SourceItem {
            id: format!("{tile} {}", arguments.imagery_year),
            href,
        };
        let spec = RasterSpec {
            epsg: u32::from(project::EPSG_WEB_MERCATOR),
            metres_per_pixel: MOSAIC_PIXEL_METRES,
            bands: 3,
            // No threshold: a coastal mosaic tile that is a few percent land
            // compresses smaller than some blank ones, so guessing by size
            // would discard the shoreline.
            fallback_nodata: 0.0,
            empty_tile_limit: 0,
        };
        rasters.push(SourceRaster::open(client, item, spec).await?);
    }

    let extent = resample::projected_extent(grid, &projector, MOSAIC_PIXEL_METRES)?;
    let mut window = Window::covering(
        extent.min_x,
        extent.min_y,
        extent.max_x,
        extent.max_y,
        MOSAIC_PIXEL_METRES,
        3,
        0.0,
    )?;

    let bytes: u64 = rasters.iter().map(|r| r.bytes_for(&window)).sum();
    println!("Estimated download: {}", coverage::describe_bytes(bytes));

    let mut downloaded = 0;
    for raster in &rasters {
        downloaded += raster.fill(&mut window, arguments.concurrency).await?;
    }

    let mut canvas = Canvas::new(grid, 3, 0.0)?;
    canvas.fill_from(grid, &projector, &window, Provenance::FILLED)?;
    drop(window);

    write::write_rgb_geotiff(&arguments.output, grid, canvas.values())?;

    let tally = canvas.tally();
    let (covered, _, none) = tally.percentages();
    println!("Wrote {}", arguments.output.display());
    println!(
        "  {} x {} pixels, {} downloaded",
        grid.width,
        grid.height,
        coverage::describe_bytes(downloaded)
    );
    println!("  {covered:.2}% imagery, {none:.2}% no data");
    Ok(())
}

/// The Web Mercator pixel size the Sentinel-2 mosaics are published at.
const MOSAIC_PIXEL_METRES: f64 = 19.109_257_071_294_063;

/// The shorter side a box could have and still fit within a pixel budget.
///
/// Only used to make the "too large" error actionable.
fn fitting_degrees(grid: &OutputGrid, max_pixels: u64) -> f64 {
    let scale = (max_pixels as f64 / grid.pixel_count() as f64).sqrt();
    grid.box_
        .width_degrees()
        .min(grid.box_.height_degrees())
        .max(0.0)
        * scale
}
