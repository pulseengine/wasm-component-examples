use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    // Get current time as seconds since Unix epoch
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");

    let total_secs = now.as_secs();

    // Calculate date components from Unix timestamp
    // Days since epoch (Jan 1, 1970)
    let days_since_epoch = (total_secs / 86400) as i64;

    // Calculate year, month, day using a simplified algorithm
    let (year, month, day) = days_to_ymd(days_since_epoch);

    // Calculate time components
    let secs_today = total_secs % 86400;
    let hours = secs_today / 3600;
    let minutes = (secs_today % 3600) / 60;
    let seconds = secs_today % 60;

    let month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ];

    let month_name = month_names[(month - 1) as usize];

    println!(
        "Hello Rust on {} {}, {} at {:02}:{:02}:{:02} UTC",
        month_name, day, year, hours, minutes, seconds
    );
}

/// Convert days since Unix epoch to (year, month, day)
fn days_to_ymd(days: i64) -> (i64, i64, i64) {
    // Algorithm based on Howard Hinnant's date algorithms
    // http://howardhinnant.github.io/date_algorithms.html

    let z = days + 719468; // shift epoch from 1970-01-01 to 0000-03-01
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // year of era [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year [0, 365]
    let mp = (5 * doy + 2) / 153; // month of year (Mar=0) [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // day [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // month [1, 12]
    let y = if m <= 2 { y + 1 } else { y };

    (y, m as i64, d as i64)
}
