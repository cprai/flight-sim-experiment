//! Working out what a download would yield, before committing to it.
//!
//! HRDEM covers only surveyed LiDAR areas, and the files are tens to hundreds
//! of gigabytes. A plausible-looking box can easily be mostly holes, and
//! discovering that after a long download is a poor way to find out. Every tile
//! records its compressed size in the header the fetch already has to read, and
//! an all-nodata tile is unmistakable at that size, so the answer costs nothing
//! extra to compute.
//!
//! The result is an estimate, exact only to the 512-pixel tile, and it is an
//! **upper bound on coverage** rather than a best guess: a tile holding a single
//! valid pixel is indistinguishable here from a full one, so it counts as
//! covered. Near the edge of a survey that gap is wide. Measured on a real box
//! over Squamish, this reports 16.2% at one metre where the true figure is
//! 0.56% -- the four contributing tiles are almost entirely nodata within the
//! requested ground. Callers should present the numbers as "at most this much
//! data, at least this much missing", which is what makes them safe to act on.
//!
//! The percentages printed at the end of a run are counted from the resampled
//! pixels themselves and are exact.

use std::io::{IsTerminal, Write};

use anyhow::{Context, Result};

use crate::extent::TileExtent;
use crate::source::SourceRaster;

/// Roughly how many points to test. Sampling rather than projecting every pixel
/// keeps the estimate instant even for a box of a hundred million pixels; at
/// tile granularity the extra precision would be imaginary anyway.
const TARGET_SAMPLES: u64 = 250_000;

/// What a download is expected to produce.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Estimate {
    pub sampled: u64,
    pub one_metre: u64,
    pub two_metre: u64,
    pub missing: u64,
    pub bytes: u64,
}

impl Estimate {
    /// The shares of one-metre, two-metre and absent data, as percentages.
    ///
    /// The first two are upper bounds and the third a lower bound, for the
    /// reason given at the top of this module.
    pub fn percentages(&self) -> (f64, f64, f64) {
        if self.sampled == 0 {
            return (0.0, 0.0, 0.0);
        }
        let share = |n: u64| 100.0 * n as f64 / self.sampled as f64;
        (
            share(self.one_metre),
            share(self.two_metre),
            share(self.missing),
        )
    }

    /// A *lower* bound on how much of the box has no data. The prompt therefore
    /// errs towards letting a download through, never towards blocking one that
    /// would have been fine.
    pub fn missing_fraction(&self) -> f64 {
        if self.sampled == 0 {
            return 1.0;
        }
        self.missing as f64 / self.sampled as f64
    }
}

/// Estimates coverage by testing a regular sample of the output extent.
///
/// A point counts as one-metre if any one-metre raster holds data there, then
/// as two-metre on the same test, and otherwise as missing -- the same
/// preference order the fill itself applies.
///
/// No projection is involved: the output is drawn on the mosaics' own grid, so
/// a sample point's metres are already the metres the rasters are indexed by.
pub fn estimate(
    extent: &TileExtent,
    fine: &[SourceRaster],
    coarse: &[SourceRaster],
) -> Result<Estimate> {
    let grid = extent.grid(0);

    // One stride for both axes keeps the sample pattern square on the ground.
    let stride = (grid.texel_count() as f64 / TARGET_SAMPLES as f64)
        .sqrt()
        .ceil()
        .max(1.0) as u32;

    let mut estimate = Estimate {
        sampled: 0,
        one_metre: 0,
        two_metre: 0,
        missing: 0,
        bytes: 0,
    };

    let mut y = 0;
    while y < grid.height {
        let mut x = 0;
        while x < grid.width {
            let (metres_x, metres_y) = grid.centre_of(x, y);
            estimate.sampled += 1;
            if fine.iter().any(|r| r.has_data_at(metres_x, metres_y)) {
                estimate.one_metre += 1;
            } else if coarse.iter().any(|r| r.has_data_at(metres_x, metres_y)) {
                estimate.two_metre += 1;
            } else {
                estimate.missing += 1;
            }
            x += stride;
        }
        y += stride;
    }

    Ok(estimate)
}

/// Renders a byte count the way a person reads it.
pub fn describe_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// Asks the user whether to go ahead with a poorly covered download.
///
/// Returns `Ok(true)` to proceed. When there is no terminal to ask -- a script,
/// a CI job -- this fails rather than blocking on a prompt nobody will answer,
/// and says which flag would have let it through.
pub fn confirm(estimate: &Estimate, threshold: f64, assume_yes: bool) -> Result<bool> {
    if assume_yes || estimate.missing_fraction() <= threshold {
        return Ok(true);
    }

    let (_, _, missing) = estimate.percentages();
    if !std::io::stdin().is_terminal() {
        anyhow::bail!(
            "at least {missing:.1}% of the requested box has no elevation data, above \
             the {:.0}% threshold, and there is no terminal to confirm at; pass --yes \
             to download it anyway or --prompt-threshold to change the limit",
            threshold * 100.0
        );
    }

    print!("At least {missing:.1}% of this box has no data, likely more. Download anyway? [y/N] ");
    std::io::stdout().flush().context("prompting")?;

    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .context("reading the answer")?;
    Ok(matches!(answer.trim(), "y" | "Y" | "yes" | "Yes"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn estimate_of(one_metre: u64, two_metre: u64, missing: u64) -> Estimate {
        Estimate {
            sampled: one_metre + two_metre + missing,
            one_metre,
            two_metre,
            missing,
            bytes: 0,
        }
    }

    #[test]
    fn the_three_shares_add_up() {
        let (one, two, none) = estimate_of(70, 20, 10).percentages();
        assert!((one - 70.0).abs() < 1e-9);
        assert!((two - 20.0).abs() < 1e-9);
        assert!((none - 10.0).abs() < 1e-9);
    }

    #[test]
    fn a_box_with_nothing_sampled_counts_as_entirely_missing() {
        let nothing = estimate_of(0, 0, 0);
        assert_eq!(nothing.missing_fraction(), 1.0);
        assert_eq!(nothing.percentages(), (0.0, 0.0, 0.0));
    }

    #[test]
    fn a_well_covered_box_is_not_queried() {
        let good = estimate_of(95, 0, 5);
        assert!(confirm(&good, 0.2, false).expect("should not prompt"));
    }

    /// The test process has no terminal, so this exercises the non-interactive
    /// path: refuse rather than block, and name the flag that would proceed.
    #[test]
    fn a_poorly_covered_box_refuses_when_there_is_nobody_to_ask() {
        let bad = estimate_of(10, 0, 90);
        let error = confirm(&bad, 0.2, false)
            .expect_err("should refuse without a terminal")
            .to_string();
        assert!(error.contains("--yes"), "{error}");
        assert!(error.contains("90.0%"), "{error}");
    }

    #[test]
    fn assuming_yes_skips_the_question_entirely() {
        let bad = estimate_of(0, 0, 100);
        assert!(confirm(&bad, 0.2, true).expect("--yes should proceed"));
    }

    #[test]
    fn a_threshold_of_one_accepts_a_box_with_no_data_at_all() {
        let empty = estimate_of(0, 0, 100);
        assert!(confirm(&empty, 1.0, false).expect("should not prompt"));
    }

    #[test]
    fn byte_counts_are_rendered_for_people() {
        assert_eq!(describe_bytes(0), "0 B");
        assert_eq!(describe_bytes(512), "512 B");
        assert_eq!(describe_bytes(1024), "1.0 KiB");
        assert_eq!(describe_bytes(1_085_450), "1.0 MiB");
        assert_eq!(describe_bytes(152_650_649_841), "142.2 GiB");
    }
}
