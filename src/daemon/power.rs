//! Whether the machine runs on battery power, so embedding can wait for a
//! charger.

/// `Some(true)` on battery power, `Some(false)` on external power, `None`
/// when it cannot be told (such as on a desktop without a battery).
#[cfg(target_os = "macos")]
pub fn on_battery() -> Option<bool> {
    let output = std::process::Command::new("/usr/bin/pmset")
        .args(["-g", "batt"])
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    parse_pmset(&String::from_utf8_lossy(&output.stdout))
}

/// `pmset -g batt` starts with `Now drawing from 'AC Power'` or
/// `'Battery Power'`.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn parse_pmset(output: &str) -> Option<bool> {
    let first = output.lines().next()?;
    if first.contains("'Battery Power'") {
        Some(true)
    } else if first.contains("'AC Power'") || first.contains("'UPS Power'") {
        Some(false)
    } else {
        None
    }
}

#[cfg(target_os = "linux")]
pub fn on_battery() -> Option<bool> {
    on_battery_in(std::path::Path::new("/sys/class/power_supply"))
}

/// Reads the power supplies under `dir`: on battery when a battery is
/// discharging and no mains supply is online.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn on_battery_in(dir: &std::path::Path) -> Option<bool> {
    let read = |path: std::path::PathBuf| {
        std::fs::read_to_string(path)
            .map(|text| text.trim().to_owned())
            .unwrap_or_default()
    };
    let mut battery = false;
    let mut discharging = false;
    let mut mains_online = false;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let supply = entry.path();
        match read(supply.join("type")).as_str() {
            "Battery" => {
                battery = true;
                discharging |= read(supply.join("status")) == "Discharging";
            }
            "Mains" | "USB" => mains_online |= read(supply.join("online")) == "1",
            _ => {}
        }
    }
    if !battery {
        return None;
    }
    Some(discharging && !mains_online)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn on_battery() -> Option<bool> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pmset_output_names_the_power_source() {
        let battery = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t84%; discharging; 5:01 remaining present: true\n";
        assert_eq!(parse_pmset(battery), Some(true));
        let ac = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged; 0:00 remaining present: true\n";
        assert_eq!(parse_pmset(ac), Some(false));
        assert_eq!(parse_pmset(""), None);
    }

    #[test]
    fn linux_power_supplies_tell_battery_from_mains() {
        let dir = tempfile::tempdir().unwrap();
        let supply = |name: &str, files: &[(&str, &str)]| {
            let path = dir.path().join(name);
            std::fs::create_dir_all(&path).unwrap();
            for (file, value) in files {
                std::fs::write(path.join(file), format!("{value}\n")).unwrap();
            }
        };
        assert_eq!(on_battery_in(dir.path()), None);
        supply("AC", &[("type", "Mains"), ("online", "0")]);
        supply("BAT0", &[("type", "Battery"), ("status", "Discharging")]);
        assert_eq!(on_battery_in(dir.path()), Some(true));
        supply("AC", &[("type", "Mains"), ("online", "1")]);
        assert_eq!(on_battery_in(dir.path()), Some(false));
        supply("AC", &[("type", "Mains"), ("online", "0")]);
        supply("BAT0", &[("type", "Battery"), ("status", "Full")]);
        assert_eq!(on_battery_in(dir.path()), Some(false));
    }
}
