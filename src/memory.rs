//! How much memory the run is using, and how much there is to use.
//!
//! Two pools, and neither of them is visible from inside the renderer without
//! asking somebody outside it.
//!
//! **GPU.** What this process allocated comes from wgpu's own allocator report:
//! [`wgpu::Device::generate_allocator_report`] returns the blocks
//! `gpu-allocator` reserved from the driver and the sub-allocations handed out
//! of them. `total_reserved_bytes` is the honest figure of the two -- it is what
//! the card lost to this process, including the slack inside a block that
//! nothing has been placed in yet. What the *card* holds is not wgpu's to know,
//! so it is read from the kernel: `amdgpu` publishes `mem_info_vram_total` and
//! `mem_info_vram_used` under the DRM device, which count every client
//! including the compositor. The alternative, [`wgpu::Device::get_internal_counters`],
//! was rejected: its `buffer_memory` and `texture_memory` are zero unless wgpu
//! is built with its `counters` feature, which puts an atomic on every resource
//! create and destroy for the whole run rather than only when a readout is
//! being drawn.
//!
//! **System.** `/proc/self/status` for this process's resident set,
//! `/proc/meminfo` for the machine's. `MemAvailable` rather than `MemFree`,
//! because the kernel's own estimate of what a new allocation could get is the
//! only one of the two that means anything on a machine with a page cache.
//!
//! Every source is a file that may not exist -- another driver, another
//! platform, a container that hid `/sys` -- so each is read independently and a
//! missing one leaves its field zero rather than failing the sample. The
//! readout draws the fields it has.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// What the run is using, and what there is, in bytes. Zero means unknown.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Memory {
    /// GPU memory this process reserved from the driver, through wgpu.
    pub vram_ours: u64,
    /// GPU memory in use across the whole device, every client included.
    pub vram_used: u64,
    /// GPU memory the device has.
    pub vram_total: u64,
    /// This process's resident set.
    pub ram_ours: u64,
    /// System memory a new allocation could not get, `MemTotal - MemAvailable`.
    pub ram_used: u64,
    /// System memory the machine has.
    pub ram_total: u64,
}

/// How often the pools are actually asked. See [`Meter::sample`].
const INTERVAL: Duration = Duration::from_millis(250);

/// Samples both pools, no more often than [`INTERVAL`].
///
/// The rate is the whole reason this is a struct rather than a function. A
/// sample is four file reads, a parse of the fifty-odd lines of `/proc/meminfo`
/// and a walk of wgpu's allocator under its lock: 49 us mean and 94 us at worst
/// over two hundred forced samples on this machine, against a frame that
/// medians 1.6 ms. Charging every frame three percent of itself to redraw
/// numbers that do not move would be paying for the measurement out of the
/// thing being measured. Nothing here moves on a frame's timescale either --
/// nothing allocates after the chain is read in -- so a quarter of a second is
/// four updates a second of figures that are steady for minutes at a time.
pub struct Meter {
    /// The card's own directory under `/sys/class/drm`, if one matched.
    card: Option<PathBuf>,
    taken: Option<Instant>,
    last: Memory,
}

impl Meter {
    /// Finds the kernel's directory for `adapter`, if this machine has one.
    ///
    /// Matched on the PCI vendor and device ids rather than on the card index:
    /// this machine enumerates an integrated GPU as `card0` and the discrete one
    /// as `card1`, `WGPU_POWER_PREF` decides which of them wgpu opened, and
    /// reading the wrong one would report a 512 MiB pool for a 12 GB card
    /// without anything looking amiss. Two identical cards are indistinguishable
    /// this way and the first wins; the numbers would be the same size and the
    /// wrong one, which is a limit worth having over no numbers at all.
    pub fn new(adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        Self {
            card: find_card(Path::new("/sys/class/drm"), info.vendor, info.device),
            taken: None,
            last: Memory::default(),
        }
    }

    /// The last sample, taken again if it has gone stale.
    pub fn sample(&mut self, device: &wgpu::Device) -> Memory {
        let now = Instant::now();
        if let Some(taken) = self.taken
            && now.duration_since(taken) < INTERVAL
        {
            return self.last;
        }
        self.taken = Some(now);
        // Reserved rather than allocated: a block wgpu took from the driver and
        // has not filled is gone from the card all the same, and it is the card
        // the two rows above this one are talking about.
        self.last.vram_ours = device
            .generate_allocator_report()
            .map(|report| report.total_reserved_bytes)
            .unwrap_or_default();
        if let Some(card) = &self.card {
            self.last.vram_total = read_number(&card.join("mem_info_vram_total"));
            self.last.vram_used = read_number(&card.join("mem_info_vram_used"));
        }
        self.last.ram_ours = read_table(Path::new("/proc/self/status"), "VmRSS");
        // Read once for both fields. `/proc/meminfo` is fifty-odd lines the
        // kernel formats on every open, and it was being opened twice.
        let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        self.last.ram_total = field(&meminfo, "MemTotal");
        // `MemAvailable` is what a new allocation could get, which is the
        // opposite of what the readout wants. Subtracting here keeps the two
        // `used` fields meaning the same thing.
        self.last.ram_used = self
            .last
            .ram_total
            .saturating_sub(field(&meminfo, "MemAvailable"));
        self.last
    }
}

/// The `cardN` directory whose PCI ids are `vendor` and `device`.
fn find_card(drm: &Path, vendor: u32, device: u32) -> Option<PathBuf> {
    let ids = |path: &Path| {
        // Written as `0x1002`, which `from_str_radix` will not take with its
        // prefix still on.
        let read = |name: &str| {
            std::fs::read_to_string(path.join(name))
                .ok()
                .and_then(|text| u32::from_str_radix(text.trim().trim_start_matches("0x"), 16).ok())
        };
        Some((read("vendor")?, read("device")?))
    };
    let mut cards: Vec<PathBuf> = std::fs::read_dir(drm)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                // `card0`, not `card0-DP-4`: the connectors are directories of
                // the same parent and carry no device of their own.
                .is_some_and(|name| {
                    name.strip_prefix("card")
                        .is_some_and(|rest| rest.chars().all(|c| c.is_ascii_digit()))
                })
        })
        .collect();
    // `read_dir` is in whatever order the filesystem hands back, and which of
    // two identical cards is picked should at least not change between runs.
    cards.sort();
    cards
        .into_iter()
        .map(|card| card.join("device"))
        .find(|path| ids(path) == Some((vendor, device)))
}

/// A file holding one integer, in bytes.
fn read_number(path: &Path) -> u64 {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| text.trim().parse().ok())
        .unwrap_or_default()
}

/// The `key:` line of a `/proc` table, for a table only one field is read from.
fn read_table(path: &Path, key: &str) -> u64 {
    let Ok(text) = std::fs::read_to_string(path) else {
        return 0;
    };
    field(&text, key)
}

/// Pulls `key` out of the `Key:   1234 kB` form both `/proc` tables use.
fn field(text: &str, key: &str) -> u64 {
    text.lines()
        .find_map(|line| {
            let rest = line.strip_prefix(key)?.strip_prefix(':')?;
            rest.split_whitespace().next()?.parse::<u64>().ok()
        })
        .map(|kib| kib * 1024)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MEMINFO: &str = "\
MemTotal:       15490644 kB
MemFree:         1234567 kB
MemAvailable:    7159136 kB
Buffers:          123456 kB
";

    #[test]
    fn a_meminfo_field_comes_back_in_bytes() {
        assert_eq!(field(MEMINFO, "MemTotal"), 15_490_644 * 1024);
        assert_eq!(field(MEMINFO, "MemAvailable"), 7_159_136 * 1024);
    }

    /// `MemFree` is a prefix of nothing, but `Mem` is a prefix of all three and
    /// a key matched without its colon would take whichever line came first.
    #[test]
    fn a_prefix_of_a_key_is_not_that_key() {
        assert_eq!(field(MEMINFO, "Mem"), 0);
        assert_eq!(field(MEMINFO, "MemTot"), 0);
    }

    #[test]
    fn a_missing_field_reads_as_unknown_rather_than_panicking() {
        assert_eq!(field(MEMINFO, "VmRSS"), 0);
        assert_eq!(field("", "MemTotal"), 0);
    }

    #[test]
    fn a_missing_file_reads_as_unknown() {
        assert_eq!(read_number(Path::new("/nonexistent/vram")), 0);
        assert_eq!(read_table(Path::new("/nonexistent/meminfo"), "MemTotal"), 0);
    }

    /// The whole point of matching on ids: the connector directories share the
    /// `card` prefix and have no device, and the card that enumerates first is
    /// not necessarily the one wgpu opened.
    #[test]
    fn the_card_is_found_by_its_ids_and_not_its_index() {
        let root = std::env::temp_dir().join("flight-sim-drm-test");
        let _ = std::fs::remove_dir_all(&root);
        for (card, vendor, device) in [("card0", "0x1002", "0x164e"), ("card1", "0x1002", "0x73df")]
        {
            let dir = root.join(card).join("device");
            std::fs::create_dir_all(&dir).expect("making a fake card");
            std::fs::write(dir.join("vendor"), format!("{vendor}\n")).expect("vendor");
            std::fs::write(dir.join("device"), format!("{device}\n")).expect("device");
        }
        // A connector, which is what a plain `card` prefix would also match.
        std::fs::create_dir_all(root.join("card0-DP-4")).expect("making a fake connector");

        assert_eq!(
            find_card(&root, 0x1002, 0x73df),
            Some(root.join("card1").join("device"))
        );
        assert_eq!(
            find_card(&root, 0x1002, 0x164e),
            Some(root.join("card0").join("device"))
        );
        // A card that is not there at all, which is every non-AMD machine.
        assert_eq!(find_card(&root, 0x10de, 0x2704), None);
        let _ = std::fs::remove_dir_all(&root);
    }
}
