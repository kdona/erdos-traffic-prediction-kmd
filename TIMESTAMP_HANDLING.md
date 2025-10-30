# Timestamp and Timezone Handling

## Current Approach: Unix Epoch Integers

Right now, timestamps in `database/az511.db` are stored as Unix epoch integers (seconds since January 1, 1970).

**Example:**
- `1704067200` = 2024-01-01 00:00:00
- `1735689600` = 2025-01-01 00:00:00

**Pros:**
- Compact (just one integer)
- Fast for comparisons and arithmetic
- No timezone ambiguity (always UTC)
- This is how the AZ511 API returns timestamps

**Cons:**
- Not human-readable - you can't look at the database and tell what time it is
- Requires conversion every time you load data
- Easy to mess up the units (seconds vs milliseconds)

## Alternative: Store as ISO 8601 Strings

We could convert timestamps when storing them:

**Example:**
- `2024-01-01T07:00:00-07:00` (Arizona time)
- `2024-01-01T14:00:00Z` (UTC)

**Pros:**
- Human-readable - you can inspect the database directly
- SQLite has DATE/TIME functions that work on these
- Unambiguous with timezone info

**Cons:**
- Takes more space (string vs integer)
- Slightly slower for comparisons
- Need to convert API response before storing

## Recommendation

**For raw API storage (az511.db, workzones.db):** Keep Unix epoch

Why? The API gives us Unix epoch, and we're just storing the raw response. No need to convert twice (API → datetime → epoch when storing, epoch → datetime when loading).

**For processed/analysis data:** Convert to datetime immediately

Once we load data for analysis, convert to timezone-aware datetime objects. This is what we already do in `prepare_i10_training_data.py`.

## Arizona Timezone Handling

**Important:** Arizona doesn't observe daylight saving time!

- Most of the year: MST (UTC-7)
- Arizona NEVER switches to MDT
- Other states switch to MDT (UTC-6) in summer
- So Arizona is sometimes MST, sometimes equivalent to PDT (same offset, different name)

**How we handle this:**

All code uses `ZoneInfo("America/Phoenix")`, which correctly handles this:

```python
from zoneinfo import ZoneInfo

AZ_TZ = ZoneInfo("America/Phoenix")  # Always UTC-7, no DST
dt = pd.to_datetime(timestamp, unit='s', utc=True).dt.tz_convert(AZ_TZ)
```

This is already implemented in:
- `create_accident_visualization.py`
- `prepare_i10_training_data.py`
- All notebooks
- Dashboard scripts

## Verification

To check timestamps are correct:

```python
import pandas as pd
from zoneinfo import ZoneInfo

# Load a timestamp from database
epoch = 1704067200  # Example

# Convert to Arizona time
dt_az = pd.to_datetime(epoch, unit='s', utc=True).tz_convert(ZoneInfo("America/Phoenix"))
print(dt_az)  # Should show 2024-01-01 00:00:00-07:00

# Verify it's winter time (UTC-7)
print(dt_az.utcoffset())  # Should show -1 days +17:00:00 (i.e., -7 hours)
```

## Common Pitfalls We Avoid

1. **Using `tz_localize()` on naive timestamps**
   - Wrong: `pd.to_datetime(epoch, unit='s').tz_localize('America/Phoenix')`
   - Right: `pd.to_datetime(epoch, unit='s', utc=True).tz_convert('America/Phoenix')`
   - The first treats the epoch as if it's already in Arizona time (wrong!)
   - The second converts from UTC to Arizona time (correct!)

2. **Using 'US/Mountain' instead of 'America/Phoenix'**
   - `US/Mountain` DOES observe DST (switches to MDT)
   - `America/Phoenix` does NOT observe DST (stays MST)
   - For Arizona data, always use `America/Phoenix`

3. **Mixing timezone-aware and naive datetimes**
   - All our timestamps are timezone-aware after conversion
   - Pandas won't let you compare aware and naive (good!)

## Auto-Detection of Units

The `_to_datetime_utc()` function in `prepare_i10_training_data.py` handles both seconds and milliseconds:

```python
def _to_datetime_utc(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        median_val = float(series.dropna().median())
        unit = "ms" if median_val > 1e12 else "s"  # Auto-detect
        return pd.to_datetime(series, unit=unit, utc=True)
```

This catches cases where APIs return milliseconds instead of seconds.

## Summary

**Current timestamp storage:**
- Raw databases: Unix epoch integers (seconds)
- Analysis code: Timezone-aware datetime (America/Phoenix)
- No daylight saving issues (Arizona doesn't do DST)

**If you want human-readable timestamps in the database:**

Edit `database/az511.py` to convert before storing:

```python
# Current: stores epoch as-is
"Reported": event.get("Reported"),

# Change to:
"Reported": datetime.fromtimestamp(event.get("Reported"), tz=timezone.utc).isoformat(),
```

Then change schema from INTEGER to TEXT for timestamp columns.

But honestly, the current approach works fine and is what most APIs do.
