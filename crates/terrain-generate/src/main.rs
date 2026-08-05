//! Invents the tile pyramid the simulator flies over, instead of measuring it.
//!
//! `terrain-download` and `terrain-process` between them turn a survey into the
//! tree the renderer opens. This writes the same tree from a seed: a mountain
//! landscape with the structure of the Rockies, shaped by erosion rather than
//! by noise alone, with ground-cover materials read off the same fields the
//! erosion left behind. Natural cover only -- no roads, no buildings, nothing
//! anybody built.
//!
//! ```text
//! terrain-generate --output assets/terrain_generated
//! flight-sim --terrain assets/terrain_generated
//! ```
//!
//! The output is the renderer's tree, not a download: `dtm`, `dtm-max` and
//! `materials`, in the byte format `terrain-tiles` writes, so nothing in the
//! renderer knows or cares which of the two producers made the directory it was
//! pointed at. `albedo` is not written, because the renderer never opens it.
//!
//! # Two halves
//!
//! **A coarse simulation.** The whole raster is held in memory at
//! `--sim-metres` and put through, in order: fractal uplift (`shape`), thermal
//! relaxation to the angle rock stands at (`thermal`), river cutting against
//! hillslope creep (`incise` and `creep`), droplet erosion (`hydraulic`),
//! another round of each to settle what those left, and finally depression
//! filling and flow routing (`flow`). Erosion is iterative and global and
//! cannot be anything else: where a droplet goes depends on what the last one
//! did. At 16 m a 49 x 57 km raster is 11 million cells, which is affordable;
//! at one metre it would be 2.8 billion and it would not.
//!
//! The two water passes do different jobs and the landscape needs both.
//! `incise` cuts at the scale of a range -- it is what finds a way out of every
//! basin, and without it a sixth of the map ends up under standing water.
//! `hydraulic` cuts at the scale of a gully, and is what puts the fans, the
//! banks and the fine drainage texture on the valleys the other one found.
//! `creep` opposes them both, and is what gives the valleys a spacing: cutting
//! on its own runs away into whatever the grid can hold and lays a corduroy of
//! one-cell grooves over the map.
//!
//! **A per-texel function.** Every texel of every level is then a pure function
//! of its position, its level, and a smooth sample of those channels --
//! `detail` for the height and `classify` for the material. No state, no
//! neighbours, no ordering, so tiles are seamless without any overlap handling
//! and levels are band-limited by construction rather than by filtering.
//!
//! That split is not only tidiness. The intention is that this second half
//! eventually runs as a compute shader, generating ground as the camera reaches
//! it rather than reading it off disk, and `noise`, `detail` and `classify` are
//! written to port to WGSL by transcription: `f32` and `u32` only, fixed loop
//! bounds, no allocation, no tables to bind. The simulation half is the part
//! that would stay offline, and its channels are what such a shader would
//! upload.

mod build;
mod classify;
mod creep;
mod detail;
mod emit;
mod fields;
mod flow;
mod hydraulic;
mod incise;
mod noise;
mod shape;
mod thermal;
mod tiles;

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use terrain_tiles::{MATERIAL_PRODUCT, Manifest, TILE_SIZE, manifest::MATERIAL_BASE_LEVEL};

use fields::Fields;
use shape::Relief;

/// What the elevation product is called, matching the renderer's
/// `ELEVATION_PRODUCT`.
///
/// Spelled here rather than taken from the renderer because the two must agree
/// and neither should link the other; `terrain-process` names it the same way
/// for the same reason.
const ELEVATION_PRODUCT: &str = "dtm";

/// The value an elevation tile holds where nothing is known.
///
/// Nothing here is unknown -- every texel is invented ground -- but the field
/// is part of the format and the renderer treats anything below its own
/// threshold as a hole, so it has to be the sentinel the rest of the pipeline
/// writes.
const ELEVATION_NODATA: f32 = -32767.0;

/// How many rounds of river cutting shape the raw uplift, and how many settle
/// the landscape again after the droplets have been over it.
///
/// The first number is what decides how much of the map ends up under water,
/// and it is worth spending: over the default extent twenty-five rounds leave a
/// tenth of the ground flooded and eighty leave a twenty-fifth, for two minutes
/// and about two per cent off the highest peaks. Beyond that it is the peaks
/// that pay. The second number is smaller because it has less to do -- droplet
/// hollows are metres deep rather than hundreds.
const CUTTING_ROUNDS: u32 = 80;
const SETTLING_ROUNDS: u32 = 12;

#[derive(Parser, Debug)]
#[command(about = "Generate a mountain landscape as the renderer's tile pyramid", long_about = None)]
struct Arguments {
    /// Directory to build the renderer's tree in.
    #[arg(short, long, value_name = "DIR")]
    output: PathBuf,

    /// What to generate. The same seed always gives the same landscape.
    #[arg(long, default_value_t = 0)]
    seed: u32,

    /// North-west corner of the ground to cover, as `easting,northing` in
    /// EPSG:3979 metres.
    ///
    /// The default sits centred on the ground `assets/terrain` covers, so the
    /// two trees describe the same part of the world and world coordinates
    /// mean the same thing in both. Nothing about the rendering depends on it
    /// -- the world origin is the raster's own centre -- but a landscape has to
    /// be somewhere, and being somewhere comparable is free.
    #[arg(long, value_name = "E,N", value_parser = parse_origin,
          default_value = "-1990656,536576", allow_hyphen_values = true)]
    origin: [f64; 2],

    /// How much ground to cover, as `width x height` in level-0 texels.
    ///
    /// Must divide evenly by `2^max-level`, which is what lets the coarsest
    /// level describe the same ground as the finest.
    #[arg(long, value_name = "WxH", value_parser = parse_extent, default_value = "49152x57344")]
    extent: [u32; 2],

    /// Coarsest level to store.
    ///
    /// Eight is what the download of comparable ground carries. The renderer
    /// continues the chain in memory above whatever it finds.
    #[arg(long, value_name = "L", default_value_t = 8)]
    max_level: u32,

    /// Ground covered by one cell of the erosion grid, in metres.
    ///
    /// The whole raster is held at this resolution while it erodes, so halving
    /// it quadruples both the memory and the time: 16 m over the default extent
    /// is 11 million cells and about 220 MB of channels. Detail finer than this
    /// is added afterwards, per texel, and costs nothing to hold.
    #[arg(long, value_name = "M", default_value_t = 16.0)]
    sim_metres: f32,

    /// Where the highest peak ends up, in metres.
    #[arg(long, value_name = "M", default_value_t = 2600.0)]
    peak_metres: f32,

    /// Where the lowest valley floor ends up, in metres.
    #[arg(long, value_name = "M", default_value_t = 700.0)]
    valley_metres: f32,

    /// How many droplets to land on each cell of the erosion grid.
    ///
    /// The droplets are the only pass that cannot run across all cores -- see
    /// `hydraulic` -- and they are most of the time a run spends simulating.
    /// Three gives a drainage network; zero leaves the valleys the river
    /// cutting made without the gullies and fans on top of them, which is what
    /// to use when the question is about the shape of the ranges rather than
    /// about detail.
    #[arg(long, value_name = "N", default_value_t = 3)]
    droplets_per_cell: usize,

    /// Only write these products, rather than all three.
    ///
    /// For a re-run after a change to one stage: reclassifying materials should
    /// not mean rewriting fifteen gigabytes of elevation. The landscape is
    /// simulated either way, and the same seed reproduces it exactly, so the
    /// products stay in step.
    #[arg(long, value_name = "NAME")]
    product: Vec<String>,
}

/// Parses `easting,northing`.
fn parse_origin(text: &str) -> Result<[f64; 2], String> {
    let (easting, northing) = text
        .split_once(',')
        .ok_or_else(|| format!("expected `easting,northing`, got `{text}`"))?;
    let parse = |part: &str| {
        part.trim()
            .parse::<f64>()
            .map_err(|error| format!("`{part}` is not a number: {error}"))
    };
    Ok([parse(easting)?, parse(northing)?])
}

/// Parses `width x height`.
fn parse_extent(text: &str) -> Result<[u32; 2], String> {
    let (width, height) = text
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("expected `WIDTHxHEIGHT`, got `{text}`"))?;
    let parse = |part: &str| {
        part.trim()
            .parse::<u32>()
            .map_err(|error| format!("`{part}` is not a whole number of texels: {error}"))
    };
    let extent = [parse(width)?, parse(height)?];
    if extent[0] == 0 || extent[1] == 0 {
        return Err(format!("`{text}` covers no ground"));
    }
    Ok(extent)
}

impl Arguments {
    /// The manifest an elevation-shaped product of this run would carry.
    ///
    /// Every product shares the ground, so they are all this with the fields
    /// that describe *what the values mean* replaced. Building them from one
    /// place is what makes the renderer's `covers_same_ground_as` check hold by
    /// construction rather than by coincidence.
    fn elevation_manifest(&self, product: &str) -> Manifest {
        Manifest {
            version: Manifest::VERSION,
            product: product.into(),
            epsg: u32::from(terrain_tiles::write::EPSG_LAMBERT),
            tile_size: TILE_SIZE,
            base_level: 0,
            level_count: self.max_level + 1,
            base_metres_per_texel: 1.0,
            origin_metres: self.origin,
            extent_texels: self.extent,
            bands: 1,
            nodata: ELEVATION_NODATA,
        }
    }

    fn material_manifest(&self) -> Manifest {
        Manifest {
            product: MATERIAL_PRODUCT.into(),
            base_level: MATERIAL_BASE_LEVEL,
            level_count: self.max_level + 1 - MATERIAL_BASE_LEVEL,
            nodata: 0.0,
            ..self.elevation_manifest(MATERIAL_PRODUCT)
        }
    }

    fn relief(&self) -> Relief {
        Relief {
            valley_metres: self.valley_metres,
            peak_metres: self.peak_metres,
        }
    }

    /// Whether a product was asked for.
    fn wants(&self, product: &str) -> bool {
        self.product.is_empty() || self.product.iter().any(|wanted| wanted == product)
    }

    /// Refuses the combinations that would produce a tree the renderer cannot
    /// open, before anything expensive happens.
    ///
    /// `Manifest::validate` catches most of it, but only once a product is
    /// being written -- which is after the landscape has been simulated. These
    /// are cheap and they fail in the first second of a run rather than the
    /// twentieth minute.
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.max_level < 24,
            "a max level of {} is not a level anything could store",
            self.max_level
        );
        let step = 1u32 << self.max_level;
        anyhow::ensure!(
            self.extent[0].is_multiple_of(step) && self.extent[1].is_multiple_of(step),
            "an extent of {} x {} texels does not divide into level {} texels of {step} m",
            self.extent[0],
            self.extent[1],
            self.max_level
        );
        // Every level of the max pyramid reads the level below at twice its own
        // indices, which is only the same ground if the origin is a whole
        // number of the coarsest level's texels from the projection's own.
        let coarsest = f64::from(step);
        for (axis, metres) in ["easting", "northing"].iter().zip(self.origin) {
            anyhow::ensure!(
                (metres / coarsest).fract() == 0.0,
                "an origin {axis} of {metres} m is not a whole number of \
                 level-{} texels of {coarsest} m",
                self.max_level
            );
        }
        anyhow::ensure!(
            self.peak_metres > self.valley_metres,
            "a peak at {} m is not above a valley floor at {} m",
            self.peak_metres,
            self.valley_metres
        );
        anyhow::ensure!(
            self.sim_metres >= 1.0 && self.sim_metres.is_finite(),
            "an erosion grid of {} m is finer than the terrain it shapes",
            self.sim_metres
        );
        Ok(())
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,terrain_generate=info"),
    )
    .init();

    let arguments = Arguments::parse();
    arguments.validate()?;

    let elevation = arguments.elevation_manifest(ELEVATION_PRODUCT);
    let materials = arguments.material_manifest();
    let maxima = Manifest {
        product: terrain_tiles::maxima_product(ELEVATION_PRODUCT),
        ..elevation.clone()
    };
    log::info!(
        "generating {} x {} texels at {} m from seed {}, levels up to {}",
        elevation.extent_texels[0],
        elevation.extent_texels[1],
        elevation.base_metres_per_texel,
        arguments.seed,
        elevation.max_level()
    );

    let started = std::time::Instant::now();
    let fields = simulate(&arguments);
    log::info!("simulated the landscape in {:.1?}", started.elapsed());

    let mut written = 0;
    if arguments.wants(&elevation.product) {
        written += emit::heights(&arguments.output, &elevation, &fields, arguments.seed)?;
    }
    if arguments.wants(&materials.product) {
        written += emit::materials(
            &arguments.output,
            &materials,
            &fields,
            arguments.seed,
            arguments.relief(),
        )?;
    }
    if arguments.wants(&maxima.product) {
        written += build::maxima(&arguments.output, ELEVATION_PRODUCT, &elevation)?;
    }

    println!(
        "Wrote {written} tiles to {} in {:.1?}",
        arguments.output.display(),
        started.elapsed()
    );
    Ok(())
}

/// Runs the coarse simulation: uplift, then the passes that shape it.
///
/// Nothing here can fail. The landscape is invented rather than read, so there
/// is no file to be missing and no source to disagree with; the only way out is
/// a finished set of channels.
fn simulate(arguments: &Arguments) -> Fields {
    let extent_metres = [arguments.extent[0] as f32, arguments.extent[1] as f32];
    let mut fields = Fields::new(extent_metres, arguments.sim_metres);
    log::info!(
        "erosion grid is {} x {} cells at {} m, about {} MB of channels",
        fields.width(),
        fields.rows(),
        arguments.sim_metres,
        fields.width() * fields.rows() * 5 * 4 / (1024 * 1024)
    );

    // Each pass reports the range it leaves as well as its time. A pass that
    // quietly flattens the landscape, or that digs one cell far below the rest,
    // is invisible in a timing and fatal to everything after it: the rescale at
    // the end is a linear map from whatever range arrives, so a single outlier
    // presses the whole landscape into the top of the range asked for.
    let stage = |name: &str, started: std::time::Instant, fields: &Fields| {
        let (low, high) = fields.height.range();
        log::info!(
            "{name} in {:.1?}, leaving {low:.0} m to {high:.0} m",
            started.elapsed()
        );
    };

    let at = std::time::Instant::now();
    shape::raise(&mut fields, arguments.relief(), arguments.seed);
    stage("raised the hills", at, &fields);

    let at = std::time::Instant::now();
    thermal::relax(&mut fields, thermal::Settling::Bedrock);
    stage("relaxed the raw slopes", at, &fields);

    // Before the droplets, because they can deepen a channel but never cut a
    // divide: the valley network has to exist for them to work on.
    let at = std::time::Instant::now();
    incise::rivers(&mut fields, CUTTING_ROUNDS);
    stage("cut the valleys", at, &fields);

    // Again after the cutting, not only before it. A river leaves walls far
    // steeper than rock stands at, and what happens next in the real world is
    // that they fall over: the valley widens, a talus fan builds at its foot,
    // and the cliff retreats to a band of the hardest bed. Skipping this leaves
    // a landscape of knife-edged slots, and the classifier then paints better
    // than a third of it as bare rock because better than a third of it is
    // steeper than anything could grow on.
    let at = std::time::Instant::now();
    thermal::relax(&mut fields, thermal::Settling::Bedrock);
    stage("let the fresh walls fall", at, &fields);

    let at = std::time::Instant::now();
    hydraulic::erode(&mut fields, arguments.seed, arguments.droplets_per_cell);
    stage("ran the droplets", at, &fields);

    // ... and again afterwards, because a droplet run leaves the landscape full
    // of small hollows of its own, and every one of them would otherwise fill
    // and be drawn as a pond.
    let at = std::time::Instant::now();
    incise::rivers(&mut fields, SETTLING_ROUNDS);
    stage("re-cut the drainage", at, &fields);

    let at = std::time::Instant::now();
    thermal::relax(&mut fields, thermal::Settling::Sediment);
    stage("settled the fresh sediment", at, &fields);

    // Only now is the height range final, so only now can it be made the range
    // that was asked for.
    shape::rescale(&mut fields.height, arguments.relief());

    let at = std::time::Instant::now();
    flow::route(&mut fields);
    stage("routed the water", at, &fields);

    let (low, high) = fields.height.range();
    log::info!("the landscape spans {low:.0} m to {high:.0} m");
    let flooded = fields
        .filled
        .values
        .iter()
        .zip(&fields.height.values)
        .filter(|(filled, ground)| *filled - *ground > crate::detail::LAKE_METRES)
        .count();
    log::info!(
        "{:.1}% of the ground holds standing water",
        flooded as f64 * 100.0 / fields.filled.values.len() as f64
    );
    fields
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments() -> Arguments {
        Arguments::parse_from(["terrain-generate", "--output", "/tmp/does-not-exist"])
    }

    /// The defaults are the ground `assets/terrain` covers, centred, and they
    /// have to be a legal grid on their own -- this is the configuration almost
    /// every run will use.
    #[test]
    fn the_default_arguments_describe_a_legal_grid() {
        let arguments = arguments();
        arguments.validate().expect("the defaults must be legal");
        arguments
            .elevation_manifest(ELEVATION_PRODUCT)
            .write(&std::env::temp_dir().join(format!(
                "terrain-generate-{}-default-manifest",
                std::process::id()
            )))
            .expect("the default manifest must validate");
        let _ = std::fs::remove_dir_all(std::env::temp_dir().join(format!(
            "terrain-generate-{}-default-manifest",
            std::process::id()
        )));
    }

    /// The renderer refuses a tree whose products disagree about the ground, so
    /// the three manifests must agree by construction rather than by review.
    #[test]
    fn every_product_of_a_run_covers_the_same_ground() {
        let arguments = arguments();
        let elevation = arguments.elevation_manifest(ELEVATION_PRODUCT);
        let materials = arguments.material_manifest();
        let maxima = Manifest {
            product: terrain_tiles::maxima_product(ELEVATION_PRODUCT),
            ..elevation.clone()
        };
        assert!(elevation.covers_same_ground_as(&materials));
        assert!(elevation.covers_same_ground_as(&maxima));
        assert_eq!(materials.base_level, MATERIAL_BASE_LEVEL);
        assert_eq!(materials.max_level(), elevation.max_level());
        assert_eq!(maxima.product, "dtm-max");
    }

    /// An extent that does not divide into the coarsest level would be caught
    /// by the manifest, but only after the landscape had been simulated.
    #[test]
    fn an_extent_that_does_not_divide_into_the_levels_is_refused_up_front() {
        let arguments = Arguments::parse_from([
            "terrain-generate",
            "--output",
            "/tmp/does-not-exist",
            "--extent",
            "49153x57344",
        ]);
        let message = arguments.validate().expect_err("should refuse").to_string();
        assert!(message.contains("does not divide"), "got {message}");
    }

    /// The max pyramid's own check that a level sits at twice the indices of
    /// the level below fires deep inside a run; this is the same condition,
    /// stated where a user can act on it.
    #[test]
    fn an_origin_off_the_coarsest_lattice_is_refused_up_front() {
        let arguments = Arguments::parse_from([
            "terrain-generate",
            "--output",
            "/tmp/does-not-exist",
            "--origin",
            "-1990657,536576",
        ]);
        let message = arguments.validate().expect_err("should refuse").to_string();
        assert!(message.contains("whole number of level"), "got {message}");
    }

    #[test]
    fn the_extent_and_origin_parsers_accept_what_the_help_promises() {
        assert_eq!(parse_extent("49152x57344").expect("valid"), [49152, 57344]);
        assert_eq!(parse_extent("512X512").expect("valid"), [512, 512]);
        assert!(parse_extent("512").is_err());
        assert!(parse_extent("0x512").is_err());
        assert_eq!(
            parse_origin("-1990656,536576").expect("valid"),
            [-1990656.0, 536576.0]
        );
        assert!(parse_origin("-1990656").is_err());
    }

    #[test]
    fn asking_for_one_product_leaves_the_others_out() {
        let only_materials = Arguments::parse_from([
            "terrain-generate",
            "--output",
            "/tmp/does-not-exist",
            "--product",
            "materials",
        ]);
        assert!(only_materials.wants("materials"));
        assert!(!only_materials.wants("dtm"));
        assert!(!only_materials.wants("dtm-max"));
        assert!(arguments().wants("dtm"), "no filter means everything");
    }
}
