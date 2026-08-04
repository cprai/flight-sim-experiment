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
//! is one more node than there are cells on each axis, so that a bilinear
//! sample anywhere in the raster -- including at its far edge -- has four real
//! nodes around it and no clamping is needed inside the data.

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
    pub fn sample_nodes(&self, column: f32, row: f32) -> f32 {
        let (c0, r0) = (column.floor(), row.floor());
        let (fc, fr) = (column - c0, row - r0);
        let (c0, r0) = (c0 as i64, r0 as i64);
        let top = self.at(c0, r0) + (self.at(c0 + 1, r0) - self.at(c0, r0)) * fc;
        let bottom = self.at(c0, r0 + 1) + (self.at(c0 + 1, r0 + 1) - self.at(c0, r0 + 1)) * fc;
        top + (bottom - top) * fr
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

        let east = self.height.sample_nodes(column + 1.0, row);
        let west = self.height.sample_nodes(column - 1.0, row);
        let south = self.height.sample_nodes(column, row + 1.0);
        let north = self.height.sample_nodes(column, row - 1.0);
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
            height: self.height.sample_nodes(column, row),
            hardness: self.hardness.sample_nodes(column, row),
            flow: self.flow.sample_nodes(column, row),
            deposit: self.deposit.sample_nodes(column, row),
            filled: self.filled.sample_nodes(column, row),
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
