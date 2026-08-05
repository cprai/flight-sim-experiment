//! The coarse grid every erosion pass runs on, and the samples it hands out.
//!
//! A run has two halves. This is the first: a few channels covering the whole
//! raster at `--sim-metres`, which erosion reshapes over and over. Erosion is
//! iterative and global -- a droplet's path depends on where the last one put
//! its sediment -- so it cannot be a function of position and cannot be
//! evaluated per texel. Holding the whole map at a coarse resolution is what
//! makes it affordable: at 16 m a 49 x 57 km raster is 11 million cells and
//! about 45 MB a channel, against 2.8 *billion* texels at one metre.
//!
//! The second half reads this through [`Fields::sample`] and adds detail. That
//! is the half that is a pure function of position, and the half that could one
//! day run on the GPU -- at which point these channels become the textures it
//! binds. [`Sample`] is deliberately shaped like the handful of texture reads
//! such a shader would do.
//!
//! Coordinates here are **raster metres**: `x` east from the raster's western
//! edge and `y` south from its northern edge, both starting at zero. That is the
//! same sense as a texel column and row, which removes the sign flip that
//! northings would otherwise put between this crate and the tile grid.
//!
//! Nodes sit *on* the grid lines rather than in the middle of cells, and there
//! is one more node than there are cells on each axis, so that a sample
//! anywhere in the raster -- including at its far edge -- has real nodes on
//! both sides of it rather than an extrapolation of the last one.

/// What the four nodes around a position contribute to it, for a fraction of
/// the way from the second to the third.
///
/// The Catmull-Rom cubic: the curve passes through the two middle nodes, and
/// its slope at each is the central difference of that node's own neighbours,
/// which is what makes the pieces meet with a matching gradient. The four
/// weights always add to one, so a flat field stays flat and a lake stays
/// exactly at its own level.
fn catmull_rom_weights(t: f32) -> [f32; 4] {
    let (t2, t3) = (t * t, t * t * t);
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

/// One channel of the coarse grid.
#[derive(Clone, PartialEq, Debug)]
pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub values: Vec<f32>,
}

impl Grid {
    pub fn filled(width: usize, height: usize, value: f32) -> Self {
        Self {
            width,
            height,
            values: vec![value; width * height],
        }
    }

    #[allow(dead_code, reason = "read by the tests of the passes that fill a grid")]
    pub fn index(&self, column: usize, row: usize) -> usize {
        row * self.width + column
    }

    /// The node at a position, with anything outside clamped to the edge.
    ///
    /// Clamping rather than wrapping or a sentinel: every pass here treats the
    /// world as ending at the raster's edge, and a clamped read makes the edge
    /// behave like a mirror wall that neither drains nor floods.
    pub fn at(&self, column: i64, row: i64) -> f32 {
        let column = column.clamp(0, self.width as i64 - 1) as usize;
        let row = row.clamp(0, self.height as i64 - 1) as usize;
        self.values[row * self.width + column]
    }

    /// Bilinear interpolation at a position given in *nodes*, not metres.
    ///
    /// Cheap, and not what the per-texel half of the generator reads -- see
    /// [`Grid::sample_smooth`] for why. This is for callers that want the plain
    /// answer, which is the tests and nothing else.
    #[allow(
        dead_code,
        reason = "the plain interpolation the smooth one is measured against"
    )]
    pub fn sample_nodes(&self, column: f32, row: f32) -> f32 {
        let (c0, r0) = (column.floor(), row.floor());
        let (fc, fr) = (column - c0, row - r0);
        let (c0, r0) = (c0 as i64, r0 as i64);
        let top = self.at(c0, r0) + (self.at(c0 + 1, r0) - self.at(c0, r0)) * fc;
        let bottom = self.at(c0, r0 + 1) + (self.at(c0 + 1, r0 + 1) - self.at(c0, r0 + 1)) * fc;
        top + (bottom - top) * fr
    }

    /// Catmull-Rom interpolation at a position given in *nodes*, not metres.
    ///
    /// This is what the per-texel half reads, and the reason is the renderer's
    /// normals. A bilinear surface is continuous but its *gradient* is not: the
    /// slope changes abruptly at every cell line, because the interpolation
    /// switches to a different pair of nodes there. The height is fine and the
    /// shading is not -- the renderer derives its normal from the heights, so
    /// every cell line draws as a crease and a sixteen-metre grid of them
    /// covers the whole world in axis-aligned facets. Over a kilometre of
    /// generated ground the curvature landed on the lattice eleven times more
    /// strongly than off it, which is that grid, measured.
    ///
    /// It is the same mistake [`crate::noise::fade`] exists to avoid one level
    /// down, and it matters far more here: the noise it interpolates carries a
    /// few metres, and this grid carries the entire relief.
    ///
    /// Catmull-Rom rather than a quintic fade on the bilinear weights. The fade
    /// is cheaper and also smooth, but it forces the gradient to zero at every
    /// node, so a hillside comes out as a quilt of level patches with steep
    /// steps between them -- trading one visible grid for another. Catmull-Rom
    /// passes through the nodes with the slope the data implies, reproduces any
    /// linear surface exactly, and is sixteen fixed taps with no branch in
    /// them, which is what a compute-shader port wants.
    ///
    /// It can overshoot by design, which is what a cubic through four points
    /// does; callers that need a bounded channel clamp it.
    pub fn sample_smooth(&self, column: f32, row: f32) -> f32 {
        let (c0, r0) = (column.floor(), row.floor());
        let (fc, fr) = (column - c0, row - r0);
        let (c0, r0) = (c0 as i64, r0 as i64);

        let across = catmull_rom_weights(fc);
        let down = catmull_rom_weights(fr);
        let mut total = 0.0;
        for (dy, vertical) in down.iter().enumerate() {
            let mut line = 0.0;
            for (dx, horizontal) in across.iter().enumerate() {
                line += horizontal * self.at(c0 + dx as i64 - 1, r0 + dy as i64 - 1);
            }
            total += vertical * line;
        }
        total
    }

    /// The lowest and highest value held.
    pub fn range(&self) -> (f32, f32) {
        self.values
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), value| {
                (low.min(*value), high.max(*value))
            })
    }
}

/// Everything the per-texel half of the generator is allowed to know.
///
/// One struct rather than five loose arguments because this is the interface
/// between the two halves, and because a compute shader would bind it as one
/// group. Adding a channel here is the expensive kind of change -- it is
/// another 45 MB resident and another texture to bind -- so each one below
/// earns its place by being something no pure function of position could
/// recover.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Sample {
    /// Ground height, in metres.
    pub height: f32,
    /// How well the rock here resisted erosion, `0..=1`. Low is soft.
    pub hardness: f32,
    /// How much ground drains through here, as `log2` of the cell count.
    ///
    /// Logarithmic because drainage area spans six orders of magnitude between
    /// a headwater and a trunk river, and every threshold anyone wants to set
    /// against it is a threshold on the exponent.
    pub flow: f32,
    /// Net material dropped here by the water, in metres. Negative where the
    /// water cut instead.
    pub deposit: f32,
    /// The surface standing water settles to, in metres.
    ///
    /// Equal to `height` on any ground that drains, and above it inside a
    /// basin. Stored as a surface rather than as a depth or a flag so that it
    /// interpolates: a depth would have to be clamped at every shoreline, and a
    /// flag could not be interpolated at all.
    pub filled: f32,
    /// Downhill slope of the coarse surface, as a rise over run.
    pub slope: f32,
    /// Which way downhill points, as east and south components of a unit
    /// vector. Zero on ground with no slope at all.
    pub aspect: [f32; 2],
}

impl Sample {
    /// How deep the standing water is here, in metres. Zero on dry ground.
    pub fn water_depth(&self) -> f32 {
        (self.filled - self.height).max(0.0)
    }
}

/// The coarse grid, its channels, and where it sits on the ground.
#[derive(Clone, PartialEq, Debug)]
pub struct Fields {
    /// Ground covered by one grid cell, in metres.
    pub metres_per_cell: f32,
    pub height: Grid,
    pub hardness: Grid,
    pub flow: Grid,
    pub deposit: Grid,
    pub filled: Grid,
}

impl Fields {
    /// A grid covering `extent` metres, with a node on each far edge.
    pub fn new(extent_metres: [f32; 2], metres_per_cell: f32) -> Self {
        let nodes = |extent: f32| (extent / metres_per_cell).ceil() as usize + 1;
        let (width, height) = (nodes(extent_metres[0]), nodes(extent_metres[1]));
        Self {
            metres_per_cell,
            height: Grid::filled(width, height, 0.0),
            hardness: Grid::filled(width, height, 0.5),
            flow: Grid::filled(width, height, 0.0),
            deposit: Grid::filled(width, height, 0.0),
            filled: Grid::filled(width, height, 0.0),
        }
    }

    pub fn width(&self) -> usize {
        self.height.width
    }

    pub fn rows(&self) -> usize {
        self.height.height
    }

    /// Raster metres of a node.
    #[allow(dead_code, reason = "read by the tests of the passes that fill a grid")]
    pub fn metres_of_node(&self, column: usize, row: usize) -> [f32; 2] {
        [
            column as f32 * self.metres_per_cell,
            row as f32 * self.metres_per_cell,
        ]
    }

    /// Every channel at a position, given in raster metres.
    ///
    /// The slope and aspect are taken from the coarse height by central
    /// differences one cell either side, so they describe the *landform* --
    /// whether this is a cliff or a valley floor -- rather than the roughness
    /// the detail pass is about to add on top. That is the right scale for
    /// deciding what grows here: a metre of talus does not turn a meadow into
    /// a cliff.
    pub fn sample(&self, x: f32, y: f32) -> Sample {
        let column = x / self.metres_per_cell;
        let row = y / self.metres_per_cell;

        let east = self.height.sample_smooth(column + 1.0, row);
        let west = self.height.sample_smooth(column - 1.0, row);
        let south = self.height.sample_smooth(column, row + 1.0);
        let north = self.height.sample_smooth(column, row - 1.0);
        let span = 2.0 * self.metres_per_cell;
        // Downhill, so the sign of each difference is the way the ground falls.
        let (fall_east, fall_south) = ((west - east) / span, (north - south) / span);
        let slope = (fall_east * fall_east + fall_south * fall_south).sqrt();
        let aspect = if slope > 1e-6 {
            [fall_east / slope, fall_south / slope]
        } else {
            [0.0, 0.0]
        };

        Sample {
            height: self.height.sample_smooth(column, row),
            // Clamped, because the cubic can overshoot and this one's `0..=1`
            // is a contract the classifier reads straight into its thresholds.
            hardness: self.hardness.sample_smooth(column, row).clamp(0.0, 1.0),
            flow: self.flow.sample_smooth(column, row),
            deposit: self.deposit.sample_smooth(column, row),
            filled: self.filled.sample_smooth(column, row),
            slope,
            aspect,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(width: usize, height: usize) -> Grid {
        let mut grid = Grid::filled(width, height, 0.0);
        for row in 0..height {
            for column in 0..width {
                let index = grid.index(column, row);
                grid.values[index] = column as f32 * 2.0 + row as f32 * 5.0;
            }
        }
        grid
    }

    /// A node on each far edge is what lets a texel at the very corner of the
    /// raster be interpolated from four real nodes. One node fewer and the last
    /// cell of every row would be extrapolated from the edge instead, which
    /// draws as a strip of flat ground all the way round the world.
    #[test]
    fn the_grid_reaches_one_node_past_the_last_cell() {
        let fields = Fields::new([49152.0, 57344.0], 16.0);
        assert_eq!(fields.width(), 49152 / 16 + 1);
        assert_eq!(fields.rows(), 57344 / 16 + 1);
        // The far corner in metres lands exactly on the last node.
        assert_eq!(
            fields.metres_of_node(fields.width() - 1, fields.rows() - 1),
            [49152.0, 57344.0]
        );
    }

    /// An extent that is not a whole number of cells still has to be covered,
    /// or the last few metres of the raster would read off the end.
    #[test]
    fn a_ragged_extent_is_still_covered_to_its_edge() {
        let fields = Fields::new([100.0, 100.0], 32.0);
        let far = fields.metres_of_node(fields.width() - 1, fields.rows() - 1);
        assert!(far[0] >= 100.0 && far[1] >= 100.0, "grid stops at {far:?}");
    }

    #[test]
    fn sampling_a_node_returns_that_node() {
        let grid = ramp(9, 7);
        for row in 0..7 {
            for column in 0..9 {
                assert_eq!(
                    grid.sample_nodes(column as f32, row as f32),
                    grid.values[grid.index(column, row)],
                    "at ({column}, {row})"
                );
            }
        }
    }

    #[test]
    fn sampling_between_nodes_interpolates_linearly() {
        let grid = ramp(5, 5);
        assert_eq!(grid.sample_nodes(1.5, 2.0), 3.0 + 10.0);
        assert_eq!(grid.sample_nodes(1.0, 2.5), 2.0 + 12.5);
        assert_eq!(grid.sample_nodes(1.25, 1.5), 2.5 + 7.5);
    }

    /// The smooth interpolation still has to *be* an interpolation: a node's
    /// own value at the node, or the surface would no longer be the one the
    /// erosion passes shaped.
    #[test]
    fn the_smooth_sample_still_passes_through_every_node() {
        let grid = ramp(9, 7);
        for row in 0..7 {
            for column in 0..9 {
                let got = grid.sample_smooth(column as f32, row as f32);
                let want = grid.values[grid.index(column, row)];
                assert!((got - want).abs() < 1e-3, "at ({column}, {row}): {got}");
            }
        }
    }

    /// Catmull-Rom reproduces any linear surface exactly, which matters twice
    /// over: a ramp does not gain ripples, and -- because a constant is linear
    /// -- the flat surface of a lake stays flat to the last bit.
    #[test]
    fn the_smooth_sample_leaves_a_ramp_a_ramp_and_a_lake_flat() {
        let grid = ramp(9, 9);
        for step in 0..40 {
            let (column, row) = (2.0 + step as f32 * 0.1, 3.0 + step as f32 * 0.07);
            let want = column * 2.0 + row * 5.0;
            let got = grid.sample_smooth(column, row);
            assert!((got - want).abs() < 1e-2, "at ({column}, {row}): {got}");
        }

        // To within the rounding of summing four weights that add to one in
        // exact arithmetic -- which is a tenth of a millimetre at this height,
        // and a great deal flatter than anything anybody can see.
        let lake = Grid::filled(9, 9, 1234.5);
        for step in 0..40 {
            let at = 2.0 + step as f32 * 0.13;
            let got = lake.sample_smooth(at, at);
            assert!((got - 1234.5).abs() < 1e-3, "the lake read {got} at {at}");
        }
    }

    /// The whole reason for the cubic. A bilinear surface bends only *at* the
    /// cell lines and is dead straight between them, so the renderer's normals
    /// jump there and the lattice draws as a grid of creases over the world.
    ///
    /// Measured the way the artifact was found: the second difference along a
    /// row, sorted by where it falls in the cell. Bilinear puts essentially all
    /// of it on the lattice; the smooth sample has to spread it over the cell
    /// instead.
    #[test]
    fn the_smooth_sample_does_not_bend_only_on_the_lattice() {
        // A grid with something to bend around -- a ramp has no curvature
        // anywhere and would tell us nothing.
        let mut grid = Grid::filled(24, 8, 0.0);
        for row in 0..8 {
            for column in 0..24 {
                let index = grid.index(column, row);
                grid.values[index] = (column as f32 * 0.9).sin() * 40.0 + row as f32 * 3.0;
            }
        }

        let concentration = |sample: &dyn Fn(f32, f32) -> f32| {
            let step = 1.0 / 8.0;
            let curvature = |at: f32| {
                (sample(at - step, 4.0) - 2.0 * sample(at, 4.0) + sample(at + step, 4.0)).abs()
            };
            let (mut on, mut off) = (0.0f64, 0.0f64);
            for i in 0..8 * 16 {
                let at = 4.0 + i as f32 * step;
                // Within half a step of a cell line, or not.
                if (at - at.round()).abs() < step * 0.5 {
                    on += f64::from(curvature(at));
                } else {
                    off += f64::from(curvature(at)) / 7.0;
                }
            }
            on / off.max(1e-9)
        };

        let bilinear = concentration(&|column, row| grid.sample_nodes(column, row));
        let smooth = concentration(&|column, row| grid.sample_smooth(column, row));
        assert!(
            bilinear > 50.0,
            "bilinear bent {bilinear} times as hard on the lattice as off it, \
             so this grid no longer demonstrates the problem"
        );
        assert!(
            smooth < 3.0,
            "the smooth sample still bends {smooth} times as hard on the \
             lattice as off it"
        );
    }

    /// Reads outside the grid clamp rather than wrap or panic. Erosion asks for
    /// neighbours of edge nodes constantly, and a wrap would drain the eastern
    /// edge of the world into the western one.
    #[test]
    fn reads_outside_the_grid_clamp_to_its_edge() {
        let grid = ramp(4, 3);
        assert_eq!(grid.at(-5, 0), grid.at(0, 0));
        assert_eq!(grid.at(99, 0), grid.at(3, 0));
        assert_eq!(grid.at(0, -1), grid.at(0, 0));
        assert_eq!(grid.at(0, 99), grid.at(0, 2));
    }

    /// Aspect points the way water would run. Getting its sign wrong would
    /// light the terrain from inside the hills and put forests on the wrong
    /// side of every ridge, neither of which reports an error.
    #[test]
    fn aspect_points_downhill_and_slope_is_the_fall_per_metre() {
        let mut fields = Fields::new([320.0, 320.0], 32.0);
        // Ground falling ten metres per cell towards the east.
        for row in 0..fields.rows() {
            for column in 0..fields.width() {
                let index = fields.height.index(column, row);
                fields.height.values[index] = 1000.0 - column as f32 * 10.0;
            }
        }
        let sample = fields.sample(160.0, 160.0);
        assert!(
            (sample.slope - 10.0 / 32.0).abs() < 1e-5,
            "{}",
            sample.slope
        );
        assert!((sample.aspect[0] - 1.0).abs() < 1e-5, "{:?}", sample.aspect);
        assert!(sample.aspect[1].abs() < 1e-5, "{:?}", sample.aspect);

        // ... and falling towards the south instead.
        for row in 0..fields.rows() {
            for column in 0..fields.width() {
                let index = fields.height.index(column, row);
                fields.height.values[index] = 1000.0 - row as f32 * 10.0;
            }
        }
        let sample = fields.sample(160.0, 160.0);
        assert!(sample.aspect[0].abs() < 1e-5, "{:?}", sample.aspect);
        assert!((sample.aspect[1] - 1.0).abs() < 1e-5, "{:?}", sample.aspect);
    }

    #[test]
    fn flat_ground_has_no_aspect_rather_than_a_random_one() {
        let fields = Fields::new([320.0, 320.0], 32.0);
        let sample = fields.sample(160.0, 160.0);
        assert_eq!(sample.slope, 0.0);
        assert_eq!(sample.aspect, [0.0, 0.0]);
    }

    /// The filled surface is stored rather than a depth or a flag precisely so
    /// that it interpolates, and the depth falls out of it.
    #[test]
    fn water_depth_is_the_filled_surface_over_the_ground() {
        let sample = Sample {
            height: 900.0,
            filled: 912.5,
            ..Sample {
                height: 0.0,
                hardness: 0.0,
                flow: 0.0,
                deposit: 0.0,
                filled: 0.0,
                slope: 0.0,
                aspect: [0.0, 0.0],
            }
        };
        assert_eq!(sample.water_depth(), 12.5);
        let dry = Sample {
            filled: 900.0,
            ..sample
        };
        assert_eq!(dry.water_depth(), 0.0);
    }
}
