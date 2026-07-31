//! Fetches Canadian high-resolution elevation and colour for a
//! longitude/latitude box and writes them as a directory of mip-mapped tiles.
//!
//! The elevation is Natural Resources Canada's HRDEM mosaic, published as
//! Cloud-Optimized GeoTIFFs on a Lambert grid in metres. A single mosaic block
//! runs to 142 GiB, so nothing is downloaded whole: the STAC catalogue says
//! which blocks overlap the box, the COG headers say which of their tiles hold
//! data, and only those tiles are fetched.
//!
//! The output is written on that same Lambert grid, EPSG:3979, rather than
//! resampled to longitude and latitude. HRDEM sits on an integer-metre lattice,
//! so the tile grid's boundaries are source pixel boundaries and the finest
//! level is a copy rather than an interpolation.
//!
//! One-metre data is preferred and two-metre fills its gaps, because the
//! one-metre mosaic only covers ground a LiDAR survey delivered at that
//! resolution. In practice the two-metre fill is small -- across two whole
//! 500 km blocks it added 9 km2 and 1 km2 -- because both products derive from
//! the same surveys.
//!
//! What is not small is what neither of them covers. HRDEM exists only where
//! someone flew a survey, and over the Squamish box that left 11.15% of the
//! ground with no elevation at all -- which the renderer draws as sky, so the
//! terrain has holes you can fly through. MRDEM, a 30 m model covering Canada
//! without gaps, is the last tier and fills whatever the mosaics did not. It is
//! thirty times coarser than the output's finest level and is meant to be: the
//! choice is between coarse ground and no ground. The percentages printed at
//! the end say exactly how much each tier contributed.
//!
//! Colour comes from somewhere else entirely: annual cloud-free Sentinel-2
//! composites, which are already mosaicked and need no scene picking or cloud
//! masking here. They are a 19 m product, so they are stored from level 4 --
//! 16 m -- upward, and the renderer magnifies them when it wants a finer level.
//! Writing them at one metre would be storing sixteen times their own
//! resolution in detail that is not there.

mod bbox;
mod coverage;
mod extent;
mod mip;
mod project;
mod resample;
mod retry;
mod source;
mod stac;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use terrain_tiles::write::{self, TilePlacement};
use terrain_tiles::{COLOUR_BASE_LEVEL, Tile, TileGrid};

use bbox::{LatLon, LatLonBox};
use extent::{Block, TileExtent};
use project::Projector;
use resample::{Canvas, Provenance, Tally};
use source::{RasterSpec, SourceRaster, Window};
use stac::{Product, Resolution};

/// The value HRDEM uses for ground it has no measurement of.
const ELEVATION_NODATA: f32 = -32767.0;

/// The Web Mercator pixel size the Sentinel-2 mosaics are published at.
const MOSAIC_PIXEL_METRES: f64 = 19.109_257_071_294_063;

/// How many tiles square a block of elevation is filled in.
///
/// Eight tiles is 4096 texels, 67 MB as `f32`, and is the tool's largest
/// allocation: nothing ever holds more than one block, so a box a hundred times
/// larger costs a hundred times the time and none of the memory.
///
/// Smaller blocks were tried and are worse on both counts. Source tiles are
/// fetched per block, and one that straddles a block boundary is fetched once
/// per block it touches: halving the block to four tiles took the same download
/// from 298.3 MiB to 365.1 MiB and made it slower, while peak memory did not
/// improve because the canvas is not what bounds it.
const ELEVATION_BLOCK_TILES: u32 = 8;

/// How many tiles square a block of colour is filled in.
///
/// Fewer than elevation because a colour tile covers sixteen times the ground
/// and carries three bands, so four tiles square is already 33 km and 50 MB.
const COLOUR_BLOCK_TILES: u32 = 4;

#[derive(Parser, Debug)]
#[command(about = "Fetch HRDEM elevation and Sentinel-2 colour as a tile pyramid", long_about = None)]
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

    /// Directory to write the tile pyramid into, under a per-product subdirectory.
    ///
    /// What is downloaded, not what the simulator flies over: `terrain-process`
    /// reads this and writes the tree the renderer opens.
    #[arg(short, long, value_name = "DIR")]
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

    /// Skip blocks whose tiles are all already written, to continue a run that
    /// died part way through.
    ///
    /// Off by default, because "already on disk" and "already correct" are not
    /// the same claim. Tiles are named by their place on a global lattice, so a
    /// file at a given path always covers the same ground -- but it may have
    /// been written by an older version of this tool, from fewer sources, or
    /// before a bug was fixed. Resuming into such a directory would silently
    /// keep the stale tiles and hide the very difference the run was for.
    #[arg(long)]
    resume: bool,

    /// Fraction of the box that may be missing before confirmation is required.
    #[arg(long, default_value_t = 0.2, value_name = "FRACTION")]
    prompt_threshold: f64,

    /// Refuse boxes needing more than this many tiles at the finest level.
    ///
    /// A guard on disk and time rather than on memory: tiles are written
    /// uncompressed, so a level-0 elevation tile is a megabyte. Working block by
    /// block means the box no longer has a memory ceiling worth guarding.
    #[arg(long, default_value_t = 100_000, value_name = "COUNT")]
    max_tiles: u64,

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
    let extent = TileExtent::cover(box_)?;
    let base_level = if arguments.product.is_elevation() {
        0
    } else {
        COLOUR_BASE_LEVEL
    };

    let tiles = extent.tile_count(base_level);
    anyhow::ensure!(
        tiles <= arguments.max_tiles,
        "the box needs {tiles} tiles at level {base_level}, over the --max-tiles \
         limit of {}; about {:.3} degrees on its shorter side would fit",
        arguments.max_tiles,
        fitting_degrees(box_, tiles, arguments.max_tiles)
    );

    let (across, down) = extent.tiles(base_level);
    log::info!(
        "{} x {} texels on EPSG:3979, {across} x {down} tiles at level {base_level}, \
         levels {base_level}..={}",
        extent.width,
        extent.height,
        extent.max_level
    );

    // Both catalogues are searched by the ground the tiles will cover, which is
    // the requested box snapped out to whole tiles -- often much larger.
    let search = extent.geographic_box()?;
    let root = arguments.output.join(arguments.product.label());
    let client = reqwest::Client::builder()
        .user_agent(concat!("terrain-download/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("building the HTTP client")?;

    if arguments.product == Product::Albedo {
        return fetch_albedo(&arguments, &client, search, &extent, &root).await;
    }
    fetch_elevation(&arguments, &client, search, &extent, &root).await
}

/// Fetches HRDEM elevation, block by block, and writes the whole pyramid.
async fn fetch_elevation(
    arguments: &Arguments,
    client: &reqwest::Client,
    search: LatLonBox,
    extent: &TileExtent,
    root: &Path,
) -> Result<()> {
    // Every tier is opened up front: the coverage estimate has to know what the
    // fallbacks could contribute before anything is downloaded.
    let mut rasters: Vec<Vec<SourceRaster>> = Vec::new();
    for resolution in Resolution::ALL {
        let items = stac::find_items(
            client,
            &arguments.stac_root,
            resolution,
            arguments.product,
            search,
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
                // Matched exhaustively rather than defaulted: the threshold is
                // a property of how a particular product compresses, and a new
                // tier silently inheriting HRDEM's would throw away its data.
                empty_tile_limit: match resolution {
                    Resolution::OneMetre | Resolution::TwoMetre => {
                        source::ELEVATION_EMPTY_TILE_LIMIT
                    }
                    Resolution::ThirtyMetre => source::MRDEM_EMPTY_TILE_LIMIT,
                },
            };
            opened.push(SourceRaster::open(client, item, spec).await?);
        }
        rasters.push(opened);
    }
    let medium = rasters.pop().expect("every resolution was searched");
    let coarse = rasters.pop().expect("every resolution was searched");
    let fine = rasters.pop().expect("every resolution was searched");

    anyhow::ensure!(
        !fine.is_empty() || !coarse.is_empty() || !medium.is_empty(),
        "no published {} raster covers that box; even MRDEM stops at Canada's \
         borders",
        arguments.product.label()
    );

    // Every raster declares -32767, but reading it rather than assuming keeps
    // the output honest if that ever changes.
    let nodata = fine
        .first()
        .or_else(|| coarse.first())
        .or_else(|| medium.first())
        .map(SourceRaster::nodata)
        .expect("at least one raster was just checked for");

    let mut estimate = coverage::estimate(extent, &fine, &coarse, &medium)?;
    let whole = extent.grid(0).extent();
    estimate.bytes = fine.iter().map(|r| r.bytes_for(whole)).sum();

    // Deliberately phrased as bounds. The estimate judges whole 512 m tiles, so
    // a tile with one valid pixel counts the same as a full one; over ground at
    // the edge of a survey it can read 16% where the truth is 0.6%.
    let shares = estimate.percentages();
    println!(
        "Estimated coverage, judged per 512 m tile: at most {:.1}% at 1 m and \
         {:.1}% at 2 m, at least {:.1}% from 30 m data and {:.1}% missing",
        shares.one_metre, shares.two_metre, shares.thirty_metre, shares.missing
    );
    println!(
        "Estimated download: {} of 1 m tiles",
        coverage::describe_bytes(estimate.bytes)
    );

    if !coverage::confirm(&estimate, arguments.prompt_threshold, arguments.yes)? {
        println!("Nothing downloaded.");
        return Ok(());
    }

    // The one-metre mosaic is copied into the canvas tile by tile rather than
    // interpolated, which is only sound if its pixels land on the output's own
    // texels. Checked once, against the whole extent: every block's origin is a
    // tile origin, so they all share this lattice.
    //
    // Only this tier is checked, because only this tier is copied. Two metres
    // halves the output's lattice and thirty neither divides it nor lands on
    // it -- MRDEM's own origin sits 20 m off a round number -- so both go
    // through `fill_from` and interpolate, which is what the bilinear margin on
    // their windows is for. Asserting alignment for them would fail on correct
    // data.
    let base = extent.grid(0);
    for raster in &fine {
        let (west, north) = raster.origin();
        anyhow::ensure!(
            base.aligns_with(west, north, raster.metres_per_pixel()),
            "{} sits at ({west}, {north}) with {} m pixels, which is not the \
             output's metre lattice; EPSG:3979 output assumes HRDEM's integer-metre \
             grid, so this product would have to be resampled rather than copied",
            raster.item.id,
            raster.metres_per_pixel()
        );
    }

    let grid = extent.tile_grid();
    let blocks = extent.blocks(0, ELEVATION_BLOCK_TILES);
    let mut tally = Tally::default();
    let mut downloaded = 0;
    let mut written = 0;
    let mut skipped = 0;
    let mut samples = Vec::new();

    // One canvas for the whole run: the first block is the largest, so its
    // buffers fit every block after it. Scoped so it is dropped before the mip
    // pass, which has no use for it and wants the room.
    let first = blocks.first().context("the extent covers no tiles")?;
    let mut canvas = Canvas::new(first.grid, 1, nodata)?;

    for (index, block) in blocks.iter().enumerate() {
        if arguments.resume && block_is_written(root, &grid, block, 0) {
            skipped += 1;
            continue;
        }
        log::info!("block {}/{}", index + 1, blocks.len());
        canvas.reset(block.grid)?;

        // No window: each source tile is copied into the canvas as it decodes
        // and dropped straight after, so the block's source pixels are never
        // resident all at once. The extent carries no bilinear margin either,
        // because nothing is being interpolated -- and a margin would drag in a
        // whole extra ring of source tiles around every block.
        let ground = block.grid.extent();
        for raster in &fine {
            downloaded += raster
                .stream(ground, arguments.concurrency, |patch| {
                    canvas.absorb(patch, Provenance::OneMetre);
                })
                .await?;
        }

        // Two-metre tiles are fetched only for ground the one-metre pass could
        // not fill, which is usually none of it. Then MRDEM for whatever is
        // still empty after that, which over an unsurveyed box is most of it.
        //
        // Both are coarser than the output, so both interpolate rather than
        // copy, and interpolation needs the pixels either side of each sample
        // point -- more than one tile carries. Those passes keep their windows.
        downloaded += fill_holes_from(
            &mut canvas,
            &coarse,
            Resolution::TwoMetre,
            Provenance::TwoMetre,
            arguments.concurrency,
            nodata,
        )
        .await?;
        downloaded += fill_holes_from(
            &mut canvas,
            &medium,
            Resolution::ThirtyMetre,
            Provenance::ThirtyMetre,
            arguments.concurrency,
            nodata,
        )
        .await?;

        tally.add(canvas.tally());
        written += write_block(root, &grid, block, &canvas, 0, nodata, &mut samples)?;
    }

    drop((canvas, samples));
    written += mip::build_levels(root, extent, 0, 1, nodata)?;
    extent
        .manifest(arguments.product.label(), 0, 1, nodata)
        .write(root)?;

    let shares = tally.percentages();
    println!("Wrote {}", root.display());
    println!(
        "  {} x {} texels, {written} tiles, {} downloaded",
        extent.width,
        extent.height,
        coverage::describe_bytes(downloaded)
    );
    println!(
        "  {:.2}% from 1 m data, {:.2}% from 2 m data, {:.2}% from 30 m data, \
         {:.2}% no data",
        shares.one_metre, shares.two_metre, shares.thirty_metre, shares.missing
    );
    // Provenance is not stored on disk, so a resumed run cannot say where the
    // tiles it skipped came from. Saying which blocks the percentages cover is
    // the only honest option -- quietly reporting a fraction of the pyramid as
    // though it were all of it would be worse than not reporting at all.
    if skipped > 0 {
        println!(
            "  {skipped} of {} blocks were already written and were skipped; the \
             percentages above cover the {} this run filled",
            blocks.len(),
            blocks.len() - skipped
        );
    }
    Ok(())
}

/// Whether every tile this block covers is already written.
///
/// Deliberately conservative in one direction. A block does not write tiles it
/// has no data for, so one that legitimately covers some empty ground can never
/// satisfy this and is filled again on every resumed run. That wastes work on a
/// handful of blocks at the edge of the data; the alternative -- treating "some
/// tiles present" as done -- would skip a block that died half way through
/// writing and leave a hole nothing would ever fill.
fn block_is_written(root: &Path, grid: &TileGrid, block: &Block, level: u32) -> bool {
    (0..block.tiles_down).all(|row| {
        (0..block.tiles_across).all(|column| {
            let tile = Tile::new(block.tile.x + column as i32, block.tile.y + row as i32);
            grid.tile_path(root, level, tile).is_file()
        })
    })
}

/// Fills whatever the passes before it left empty, from one coarser tier.
///
/// Returns the bytes downloaded, which is zero when there was nothing left to
/// fill or no raster to fill it from -- both ordinary. The window is sized to
/// the holes rather than to the block, so a block the finer tiers already
/// covered costs one scan of the provenance and no request at all.
///
/// Texels that already have a value are left alone by `fill_from`, and that is
/// the whole of what makes this a fallback rather than an overwrite: the order
/// these are called in *is* the preference between the tiers.
async fn fill_holes_from(
    canvas: &mut Canvas,
    rasters: &[SourceRaster],
    resolution: Resolution,
    provenance: Provenance,
    concurrency: usize,
    nodata: f32,
) -> Result<u64> {
    if rasters.is_empty() || !canvas.has_holes() {
        return Ok(0);
    }

    let metres = resolution.metres();
    let Some(holes) = canvas.hole_extent(None, metres)? else {
        return Ok(0);
    };
    let mut window = Window::covering(
        holes.min_x,
        holes.min_y,
        holes.max_x,
        holes.max_y,
        metres,
        1,
        nodata,
    )?;

    let mut downloaded = 0;
    for raster in rasters {
        downloaded += raster.fill(&mut window, concurrency).await?;
    }
    canvas.fill_from(&window, None, provenance)?;
    Ok(downloaded)
}

/// Fetches cloud-free Sentinel-2 colour for the box.
///
/// Much shorter than the elevation path because the compositing has already
/// been done upstream: there is one mosaic per grid square, no second
/// resolution to fall back to, and no cloud to reason about. The coverage
/// pre-check and its prompt stay on the elevation path -- Sentinel-2 covers the
/// whole country, so a box that misses the imagery is not the failure mode
/// worth guarding, and `--yes` and `--prompt-threshold` do nothing here.
async fn fetch_albedo(
    arguments: &Arguments,
    client: &reqwest::Client,
    search: LatLonBox,
    extent: &TileExtent,
    root: &Path,
) -> Result<()> {
    anyhow::ensure!(
        stac::MOSAIC_YEARS.contains(&arguments.imagery_year),
        "only {:?} have published mosaics, not {}",
        stac::MOSAIC_YEARS,
        arguments.imagery_year
    );

    // The output is Lambert and the mosaics are Web Mercator, so unlike the
    // elevation path this one genuinely has to reproject.
    let projector = Projector::between(project::EPSG_LAMBERT, project::EPSG_WEB_MERCATOR)?;
    let tiles = stac::find_mosaic_tiles(client, &arguments.earth_search_root, search).await?;
    log::info!("imagery grid squares: {}", tiles.join(", "));

    // The mosaics' pixels are metres of projection rather than metres of ground
    // -- 19.1 of them is about 12.4 m of ground at this latitude. The size is
    // asserted rather than read, so a change of zoom level upstream fails
    // loudly instead of silently mis-placing pixels.
    let mut rasters = Vec::new();
    for tile in &tiles {
        let item = stac::SourceItem {
            id: format!("{tile} {}", arguments.imagery_year),
            href: stac::mosaic_href(tile, arguments.imagery_year),
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

    let whole = resample::source_extent(
        &extent.grid(COLOUR_BASE_LEVEL),
        Some(&projector),
        MOSAIC_PIXEL_METRES,
    )?;
    let bytes: u64 = rasters.iter().map(|r| r.bytes_for(whole)).sum();
    println!("Estimated download: {}", coverage::describe_bytes(bytes));

    let grid = extent.tile_grid();
    let blocks = extent.blocks(COLOUR_BASE_LEVEL, COLOUR_BLOCK_TILES);
    let mut tally = Tally::default();
    let mut downloaded = 0;
    let mut written = 0;
    let mut samples = Vec::new();

    let first = blocks.first().context("the extent covers no tiles")?;
    let mut canvas = Canvas::new(first.grid, 3, 0.0)?;

    for (index, block) in blocks.iter().enumerate() {
        log::info!("block {}/{}", index + 1, blocks.len());
        let source = resample::source_extent(&block.grid, Some(&projector), MOSAIC_PIXEL_METRES)?;
        let mut window = Window::covering(
            source.min_x,
            source.min_y,
            source.max_x,
            source.max_y,
            MOSAIC_PIXEL_METRES,
            3,
            0.0,
        )?;
        for raster in &rasters {
            downloaded += raster.fill(&mut window, arguments.concurrency).await?;
        }

        canvas.reset(block.grid)?;
        canvas.fill_from(&window, Some(&projector), Provenance::FILLED)?;
        drop(window);

        tally.add(canvas.tally());
        written += write_block(
            root,
            &grid,
            block,
            &canvas,
            COLOUR_BASE_LEVEL,
            0.0,
            &mut samples,
        )?;
    }

    drop((canvas, samples));
    written += mip::build_levels(root, extent, COLOUR_BASE_LEVEL, 3, 0.0)?;
    extent
        .manifest(arguments.product.label(), COLOUR_BASE_LEVEL, 3, 0.0)
        .write(root)?;

    let shares = tally.percentages();
    println!("Wrote {}", root.display());
    println!(
        "  {} x {} texels at level {COLOUR_BASE_LEVEL}, {written} tiles, {} downloaded",
        extent.size_texels(COLOUR_BASE_LEVEL).0,
        extent.size_texels(COLOUR_BASE_LEVEL).1,
        coverage::describe_bytes(downloaded)
    );
    // Colour has one source, so it lands wholly in the first tier -- see
    // `Provenance::FILLED` -- and the tiers below it are always zero here.
    println!(
        "  {:.2}% imagery, {:.2}% no data",
        shares.one_metre, shares.missing
    );
    Ok(())
}

/// Cuts a filled block into tiles and writes the ones that hold anything.
///
/// Returns how many were written. A tile with nothing under it is skipped
/// entirely rather than written full of nodata, which is what keeps a box over
/// patchy coverage from costing a megabyte per empty square kilometre -- and it
/// is how the renderer learns there is no data there, since it treats a missing
/// file as a hole.
///
/// One scratch buffer serves every tile. Writing them across a thread pool was
/// tried and gained nothing measurable -- a megabyte into the page cache is not
/// where the time goes -- while giving every worker its own copy of that buffer.
fn write_block(
    root: &Path,
    grid: &TileGrid,
    block: &Block,
    canvas: &Canvas,
    level: u32,
    nodata: f32,
    samples: &mut Vec<f32>,
) -> Result<u64> {
    let mut written = 0;
    for row in 0..block.tiles_down {
        for column in 0..block.tiles_across {
            if !canvas.tile_samples(column, row, samples) {
                continue;
            }
            let tile = Tile::new(block.tile.x + column as i32, block.tile.y + row as i32);
            let (west, north) = grid.tile_origin_metres(level, tile);
            let placement = TilePlacement {
                west,
                north,
                metres_per_texel: grid.metres_per_texel(level),
            };
            let path = grid.tile_path(root, level, tile);

            if canvas.bands == 1 {
                write::write_height_tile(&path, placement, samples, nodata)?;
            } else {
                let bytes: Vec<u8> = samples
                    .iter()
                    .map(|&v| v.round().clamp(0.0, 255.0) as u8)
                    .collect();
                write::write_colour_tile(&path, placement, &bytes)?;
            }
            written += 1;
        }
    }
    Ok(written)
}

/// The shorter side a box could have and still fit within a tile budget.
///
/// Only used to make the "too large" error actionable.
fn fitting_degrees(box_: LatLonBox, tiles: u64, max_tiles: u64) -> f64 {
    let scale = (max_tiles as f64 / tiles.max(1) as f64).sqrt();
    box_.width_degrees().min(box_.height_degrees()).max(0.0) * scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use terrain_tiles::TILE_SIZE;

    /// A block's tiles must be a whole number of tiles across, or `tile_samples`
    /// would read past the canvas.
    #[test]
    fn blocks_hold_whole_tiles() {
        let box_ = LatLonBox::from_corners(
            LatLon {
                latitude: 49.633,
                longitude: -123.307,
            },
            LatLon {
                latitude: 49.7,
                longitude: -123.2,
            },
        )
        .expect("failed to build a box");
        let extent = TileExtent::cover(box_).expect("failed to cover");

        for (level, block_tiles) in [
            (0, ELEVATION_BLOCK_TILES),
            (COLOUR_BASE_LEVEL, COLOUR_BLOCK_TILES),
        ] {
            for block in extent.blocks(level, block_tiles) {
                assert_eq!(block.grid.width % TILE_SIZE, 0, "level {level}");
                assert_eq!(block.grid.height % TILE_SIZE, 0, "level {level}");
                assert_eq!(block.grid.width / TILE_SIZE, block.tiles_across);
                assert_eq!(block.grid.height / TILE_SIZE, block.tiles_down);
            }
        }
    }

    /// `--resume` must never skip a block that is only partly written, and the
    /// price of that guarantee is that a block missing a tile it would never
    /// have written is filled again. Both directions are checked here, because
    /// only the first is a correctness claim and it is the one worth keeping.
    #[test]
    fn a_block_counts_as_written_only_when_every_tile_is_there() {
        let root = std::env::temp_dir().join(format!(
            "terrain-download-resume-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&root);

        let box_ = LatLonBox::from_corners(
            LatLon {
                latitude: 49.633,
                longitude: -123.307,
            },
            LatLon {
                latitude: 49.637,
                longitude: -123.303,
            },
        )
        .expect("failed to build a box");
        let extent = TileExtent::cover(box_).expect("failed to cover");
        let grid = extent.tile_grid();
        let blocks = extent.blocks(0, ELEVATION_BLOCK_TILES);
        let block = blocks.first().expect("the extent covers no tiles");

        assert!(
            !block_is_written(&root, &grid, block, 0),
            "nothing is written yet"
        );

        // Write every tile but the last, which is the state a run killed
        // part way through a block would leave behind.
        let tiles: Vec<Tile> = (0..block.tiles_down)
            .flat_map(|row| {
                (0..block.tiles_across).map(move |column| {
                    Tile::new(block.tile.x + column as i32, block.tile.y + row as i32)
                })
            })
            .collect();
        for tile in &tiles[..tiles.len() - 1] {
            let path = grid.tile_path(&root, 0, *tile);
            std::fs::create_dir_all(path.parent().expect("a tile path has a parent"))
                .expect("failed to make the level directory");
            std::fs::write(&path, b"not a real tile").expect("failed to write");
        }
        assert!(
            !block_is_written(&root, &grid, block, 0),
            "one missing tile must not count as written, or the gap it leaves \
             would never be filled"
        );

        let last = grid.tile_path(&root, 0, *tiles.last().expect("blocks have tiles"));
        std::fs::write(&last, b"not a real tile").expect("failed to write");
        assert!(
            block_is_written(&root, &grid, block, 0),
            "a complete block should be skipped"
        );

        std::fs::remove_dir_all(&root).expect("failed to clean up");
    }

    #[test]
    fn a_smaller_box_is_suggested_when_the_budget_is_exceeded() {
        let box_ = LatLonBox::from_corners(
            LatLon {
                latitude: 49.0,
                longitude: -124.0,
            },
            LatLon {
                latitude: 50.0,
                longitude: -123.0,
            },
        )
        .expect("failed to build a box");
        // Four times over budget should suggest half the side.
        let suggested = fitting_degrees(box_, 400, 100);
        assert!((suggested - 0.5).abs() < 1e-9, "{suggested}");
    }
}
