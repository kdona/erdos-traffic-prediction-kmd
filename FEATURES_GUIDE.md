# Feature Engineering Guide

This guide explains what each feature does and why we included it.

## Overview: Four Feature Categories

The model uses **four main categories** of features to predict travel time:

1. **Lag Features** (43.8% importance) - What happened in previous hours
2. **Time Features** (23.6% importance) - Time-of-day and day-of-week patterns
3. **Road Features** (16.8% importance) - Physical characteristics of the road segment
4. **Event Features** (15.9% importance) - Planned and unplanned incidents

---

## 1. Lag Features (Previous Hours)

Travel time from 1-3 hours ago on the same road segment. These turn out to be our best predictors (43.8% of model importance).

### How This Works

We use past observations to predict the current state. At 5pm, the model sees what actually happened at 4pm and 3pm, then predicts what's happening now.

```
Timeline:
  3pm          4pm          5pm (NOW - what we're predicting)
   ↓            ↓            ↓
[lag2]       [lag1]     [PREDICT ME!]
```

The model looks at:
- **lag1**: Travel time at 4pm
- **lag2**: Travel time at 3pm
- Uses this history to predict 5pm

This works because traffic patterns persist. Rush hour builds up over 30-60 minutes rather than starting instantly. Crashes and roadwork last for hours. If the road was congested an hour ago, it's likely still congested.

**Real example from I-10:**
| Time | Actual Speed | What Model Learns |
|------|-------------|-------------------|
| 3pm | 60 mph | Normal |
| 4pm | 45 mph | Rush hour starting (lag1 shows slowdown) |
| 5pm | 35 mph | **Model predicts: will be worse** (lag shows downward trend) |
| 6pm | 50 mph | **Model predicts: improving** (lag shows recovery) |

Patterns the model picks up:
- Lag values decreasing → traffic getting worse
- Lag values staying high → sustained congestion
- Lag values increasing → traffic improving

### Individual Lag Features

#### `lag1_tt_per_mile` / `log_lag1_tt_per_mile` (33.7% importance)
- **What it is:** Travel time per mile from **1 hour ago**
- **Example:** At 5pm, this is the travel time from 4pm
- **Why it's important:** Traffic has momentum. Rush hour doesn't start/stop instantly
- **Log transform:** We use `log(travel_time)` because travel time has a skewed distribution (most hours are fast, few hours are very slow)

#### `lag2_tt_per_mile` / `log_lag2_tt_per_mile` (10.0% importance)
- **What it is:** Travel time per mile from **2 hours ago**
- **Example:** At 5pm, this is the travel time from 3pm
- **Why it's important:** Shows the trend - is traffic getting worse or better?

#### `lag3_tt_per_mile` / `log_lag3_tt_per_mile` (included but not in top 10)
- **What it is:** Travel time per mile from **3 hours ago**
- **Why it's important:** Helps capture sustained congestion patterns (long events, major incidents)

**Key insight:** The model learns patterns like "if traffic was slow 1 hour ago and 2 hours ago, it'll probably still be slow now."

---

## 2. Time Features (Temporal Patterns)

**What they are:** Cyclical patterns in traffic based on time of day and day of week.

**Why they matter:** Traffic follows predictable daily and weekly rhythms. Rush hours happen at the same time every weekday. Weekends are different from weekdays.

### Sine/Cosine Encoding for Time

Hours are circular - 11pm and midnight are neighbors, not 23 units apart. But if we feed the model raw numbers (0-23), it treats them as a line.

The solution is to map time onto a circle using sine and cosine:

```python
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

Think of it like a clock. Each hour gets an (x,y) position:
- Midnight: (0.00, 1.00) - top of circle
- 6am: (1.00, 0.00) - right side
- Noon: (0.00, -1.00) - bottom
- 6pm: (-1.00, 0.00) - left side
- 11pm: (-0.26, 0.97) - almost back to top

Now 11pm and midnight are close in the model's view, which matches reality.

We need both sin and cos because sin alone gives the same value for 6am and 6pm. Together, they uniquely identify each hour.

#### `hour_sin` and `hour_cos` (4.0% importance combined)
- **What they are:** Encodes the hour of day (0-23) as a circle
  - `hour_sin = sin(2π × hour/24)`
  - `hour_cos = cos(2π × hour/24)`
- **Example:**
  - 6am: `(sin(π/2), cos(π/2))` = (1.0, 0.0)
  - 12pm: `(sin(π), cos(π))` = (0.0, -1.0)
  - 6pm: `(sin(3π/2), cos(3π/2))` = (-1.0, 0.0)
- **Why it's important:** Captures rush hour patterns (7-9am, 4-6pm dips in speed)

#### `dow_sin` and `dow_cos` (16.8% + 2.8% = 19.6% importance combined!)
- **What they are:** Encodes day of week (0=Monday, 6=Sunday) as a circle
  - `dow_sin = sin(2π × day/7)`
  - `dow_cos = cos(2π × day/7)`
- **Why it's important:** This is the **#2 most important feature!** Captures weekday vs weekend differences
- **Pattern learned:** Weekdays have rush hours and slower speeds; weekends are consistently faster

#### `hour_of_week_sin` and `hour_of_week_cos` (not in top 10)
- **What they are:** Combines hour and day into a single weekly cycle (0-167 hours)
- **Why included:** Captures patterns like "Friday evening rush is worse than Tuesday evening rush"

#### `is_weekend` (not in top 10)
- **What it is:** Binary flag (1 if Saturday/Sunday, 0 otherwise)
- **Why included:** Provides a direct weekday/weekend split in addition to the cyclical encoding
- **Redundant?** Partially overlaps with `dow_sin/cos`, but helps the model easily separate weekend behavior

**Key insight:** The model learns that Monday 8am behaves differently from Saturday 8am, and that 5pm is always slower than 11am on weekdays.

---

## 3. Road Features (Segment Characteristics)

**What they are:** Physical and geometric properties of the road segment that don't change over time.

**Why they matter:** Not all road segments are equal. Curves slow traffic. Longer segments take more time. On-ramps create merge congestion.

### Static Road Properties

#### `miles` (4.0% importance)
- **What it is:** Length of the TMC segment in miles
- **Example:** A segment might be 0.5 miles or 2.3 miles
- **Why it's important:** Longer segments naturally have higher travel time, even at the same speed
- **How it's used:** Model learns `travel_time ≈ miles × speed_factor`

#### `reference_speed` (6.4% importance)
- **What it is:** The posted speed limit or "free-flow" speed for this segment (in mph)
- **Example:** 65 mph on main freeway, 55 mph near interchanges
- **Why it's important:**
  - Sets expectations for how fast traffic *should* move
  - Helps model understand segment type (highway vs surface street)
  - Model learns that a segment with 65 mph reference going 50 mph = congestion

### Geometric Flags (Binary Features)

#### `curve` (3.4% importance)
- **What it is:** Does this segment have a significant curve? (1 = yes, 0 = no)
- **Example:** The "Broadway Curve" is flagged as `curve=1`
- **Why it's important:** Curves naturally slow traffic and create bottlenecks
- **Broadway Curve effect:** This famous bottleneck on I-10 is a major congestion point

#### `onramp` (not in top 10, ~2%)
- **What it is:** Does this segment have an on-ramp where vehicles merge onto the highway?
- **Why it's important:** Merging traffic creates turbulence and slowdowns
- **Pattern:** Segments with on-ramps tend to have more variable speeds

#### `offramp` (2.9% importance)
- **What it is:** Does this segment have an off-ramp where vehicles exit the highway?
- **Why it's important:** Exit lanes can create congestion as drivers change lanes
- **Surprising result:** Off-ramps are slightly MORE important than on-ramps in this corridor

**Key insight:** The model learns that a 2-mile curved segment with a reference speed of 65 mph will behave differently than a straight 0.5-mile segment with the same reference speed.

---

## 4. Event Features (Incidents and Roadwork)

**What they are:** Indicators of whether planned (roadwork) or unplanned (accidents) events are happening near this segment during this hour.

**Why they matter:** Events disrupt normal traffic patterns. Roadwork closes lanes. Accidents create rubbernecking delays.

### Event Categories

#### `evt_cat_planned` (15.9% importance - **#3 most important feature!**)
- **What it is:** Count of planned events (roadwork, construction, closures) affecting this segment this hour
- **Examples:**
  - Lane closure for repaving
  - Shoulder work
  - Bridge maintenance
  - Scheduled closures
- **Why it's so important:** Planned events are:
  - Usually more severe (full lane closures)
  - Last longer (hours to days)
  - Predictable (model can learn their patterns)
- **Surprising finding:** Despite events being rare (< 1% of hours), when they occur, they're the 3rd most important predictor!

#### `evt_cat_unplanned` (not in top 10, but included)
- **What it is:** Count of unplanned events (accidents, debris, incidents) affecting this segment
- **Examples:**
  - Vehicle crashes
  - Disabled vehicles
  - Debris on roadway
  - Police activity
- **Why less important than planned?**
  - Shorter duration (cleared faster)
  - Underreported in AZ511 (known data quality issue)
  - More random/unpredictable

#### `evt_total` (not explicitly shown, but derived)
- **What it is:** Total number of events (planned + unplanned)
- **Why included:** Provides overall "is something happening?" signal

### Event Assignment Logic

Events are assigned to road segments using **spatial matching**:
1. Each event has GPS coordinates (latitude, longitude)
2. Each TMC segment has start/end coordinates
3. Events are matched to the nearest TMC within ~0.1 miles
4. Direction-aware matching (eastbound events → eastbound TMCs)

**Important limitation:** Events affect multiple segments (upstream, downstream), but we only assign them to the nearest segment. This means the model **underestimates** event impact.

**Key insight:** The 15.9% importance of planned events is actually remarkable given that:
- Events only occur in 0.87% of all (segment, hour) observations
- Events are spatially limited (only matched to nearest segment)
- This suggests events have LARGE effects when they do occur

---

## How Features Work Together

The model doesn't just use features individually - it learns **interactions**:

### Example 1: Rush Hour + Event
- **Pattern:** `dow_cos ≈ 0.5` (weekday) + `hour_cos ≈ -1` (evening) + `evt_cat_planned = 1`
- **Model learns:** "Roadwork during evening rush on a weekday = major delay"
- **Why:** The event compounds the already-slow rush hour traffic

### Example 2: Lag + Time Pattern
- **Pattern:** `lag1_tt_per_mile = high` + `hour_sin increasing` (morning)
- **Model learns:** "If it was slow an hour ago and we're entering morning rush, it'll get worse"
- **Why:** Momentum + predictable pattern = strong signal

### Example 3: Curve + Reference Speed
- **Pattern:** `curve = 1` + `reference_speed = 65` + `lag1_tt_per_mile > 1.5× normal`
- **Model learns:** "Broadway Curve is congested (travel time much higher than free-flow)"
- **Why:** Geometric constraint (curve) + current state (lag) = bottleneck

---

## Why Not Include More Features?

You might wonder why we don't include:

### Weather
- **Would help!** Rain/snow definitely slows traffic
- **Problem:** No weather data in AZ511 database
- **Future work:** Could join with NOAA weather data by timestamp

### Demand (Traffic Volume)
- **Would help!** More cars = slower speeds
- **Problem:** INRIX data doesn't include volume counts
- **Future work:** Could use loop detector data from ADOT

### Event Description Text
- **Might help:** "Full closure" vs "shoulder work" have different impacts
- **Problem:** Text is inconsistent and messy in AZ511
- **Current approach:** We manually categorized events into planned/unplanned

### Distance to Event
- **Would help!** Events affect segments up to 1-2 miles away
- **Problem:** Requires more complex spatial modeling
- **Current approach:** Binary "event present" on nearest segment only

---

## Feature Engineering Best Practices

Based on this project, here are key lessons:

### 1. Lag Features are Gold
- **Always include** recent history for time series prediction
- Test multiple lag values (1hr, 2hr, 3hr, 6hr, 24hr)
- Consider log-transforming skewed targets

### 2. Use Cyclical Encoding for Time
- Don't use raw hour (0-23) - the model can't learn that 23 and 0 are adjacent
- Use `sin/cos` pairs to preserve circular relationships
- Works for: hour of day, day of week, month of year, etc.

### 3. One-Hot vs Count for Events
- We used **counts** (how many events?) rather than one-hot encoding
- This allows the model to distinguish "1 event" from "3 simultaneous events"
- Better for rare categorical features

### 4. Feature Scaling Matters
- Tree-based models (XGBoost, Random Forest) don't need scaling
- But we still log-transform travel time because of extreme outliers
- Linear models would need StandardScaler on all features

### 5. Test Individual Feature Importance
- Don't assume you know which features matter
- Run feature importance analysis (XGBoost, permutation importance, SHAP)
- Remove useless features (faster training, less overfitting)

---

## Understanding RMSE

RMSE (Root Mean Squared Error) measures prediction accuracy in the same units as the target variable.

For this project:
- Target: travel time per mile (seconds/mile)
- RMSE = 6 means predictions are off by 6 seconds/mile on average
- On a 1-mile segment: ±6 seconds error
- On a 5-mile stretch: ±30 seconds error

### Calculation

```
RMSE = √(Σ(actual - predicted)² / n)
```

Steps:
1. Calculate error for each prediction: `actual - predicted`
2. Square each error: `error²`
3. Average all squared errors
4. Take square root to get back to original units

Why square the errors?
- Makes all errors positive (-5 and +5 both become 25)
- Penalizes large errors more (one 10-second error is worse than two 5-second errors)
- Mathematically convenient for optimization

### Interpreting RMSE Values

For travel time prediction (seconds/mile):

| RMSE | Quality | What This Means |
|------|---------|----------------|
| 15 | Poor | Baseline (just using averages) |
| 10 | Okay | Simple models (no lag features) |
| 6-8 | Good | XGBoost with all features ✓ |
| 5-7 | Very Good | LSTM (our best model) ✓ |
| <5 | Excellent | Hard to achieve with this data |
| 0 | Perfect | Impossible in real world |

**Context for this project:**
- Average travel time: ~60 seconds/mile
- Standard deviation: ~20 seconds/mile
- Our RMSE: 6 seconds/mile
- **Relative error:** 6/60 = 10% error rate (pretty good!)

### RMSE vs Other Metrics

We also report these metrics:

#### MAE (Mean Absolute Error)
- **Formula:** Average of |actual - predicted|
- **Interpretation:** Simpler than RMSE, doesn't penalize big errors as much
- **Example:** MAE = 4 means average error is 4 seconds/mile

#### R² (R-squared)
- **Formula:** 1 - (error variance / total variance)
- **Range:** 0 to 1 (higher is better)
- **Interpretation:**
  - R² = 0.80 means model explains 80% of variance
  - R² = 0.90 means model explains 90% of variance (very good!)
  - R² = 0.50 means model explains 50% of variance (mediocre)

**For this project:**
- Our XGBoost: R² ≈ 0.88 (explains 88% of travel time variance)
- Our LSTM: R² ≈ 0.91 (explains 91% of variance - best!)

### Why RMSE Improved So Much

**Baseline model (no features, just average):**
- RMSE = 15 seconds/mile
- R² = 0.0
- Just predicts "average travel time" for every segment

**+ Lag features:**
- RMSE = 10 seconds/mile (33% improvement!)
- R² = 0.60
- "Use past to predict present"

**+ Time features (hour, day):**
- RMSE = 7 seconds/mile (53% improvement!)
- R² = 0.80
- "Account for rush hour patterns"

**+ Road features + Event features:**
- RMSE = 6 seconds/mile (60% improvement!)
- R² = 0.88
- "Full context: past + time + road + events"

**+ LSTM (sequential model):**
- RMSE = 5.5 seconds/mile (63% improvement!)
- R² = 0.91
- "Learn complex temporal patterns"

Each feature category and model improvement reduces error!

---

## Summary Table

| Feature | Type | Example Value | Importance | Why It Matters |
|---------|------|---------------|------------|----------------|
| `log_lag1_tt_per_mile` | Continuous | 0.35 | 33.7% | Traffic has momentum |
| `dow_cos` | Cyclical | 0.62 | 16.8% | Weekday vs weekend |
| `evt_cat_planned` | Count | 0 or 1 | 15.9% | Roadwork causes delays |
| `log_lag2_tt_per_mile` | Continuous | 0.33 | 10.0% | Shows trends |
| `reference_speed` | Continuous | 65.0 | 6.4% | Sets free-flow baseline |
| `miles` | Continuous | 0.85 | 4.0% | Longer = more time |
| `hour_cos` | Cyclical | -0.87 | 4.0% | Rush hour pattern |
| `curve` | Binary | 1 | 3.4% | Geometric bottleneck |
| `offramp` | Binary | 1 | 2.9% | Exit congestion |
| `dow_sin` | Cyclical | 0.78 | 2.8% | Weekly pattern |

**Total explained:** 99.5% of feature importance comes from these 10 features!

---

## Going Deeper: Mathematical Details

For those interested in the technical details:

### Cyclical Encoding Math
Given a value `x` in range `[0, period)`:
```
sin_feature = sin(2π × x / period)
cos_feature = cos(2π × x / period)
```

Example for hour (period = 24):
- Hour 0:  `(sin(0), cos(0))` = `(0, 1)`
- Hour 6:  `(sin(π/2), cos(π/2))` = `(1, 0)`
- Hour 12: `(sin(π), cos(π))` = `(0, -1)`
- Hour 18: `(sin(3π/2), cos(3π/2))` = `(-1, 0)`
- Hour 23: `(sin(23π/12), cos(23π/12))` ≈ `(-0.26, 0.97)` - close to hour 0!

### Log Transform Rationale
Travel time has a long right tail (most hours: 60-90 seconds, rare hours: 300+ seconds).

Benefits of `log(travel_time)`:
- Reduces skewness
- Makes residuals more normally distributed
- Puts model focus on % changes rather than absolute changes
- Example: Difference between 60s and 90s is as important as 120s and 180s (both are 50% increases)

### Feature Interaction Example
XGBoost learns splits like:
```
IF lag1_tt > 1.2:
    IF evt_cat_planned > 0:
        predict HIGH delay (2.5 sec/mile)
    ELSE IF hour_cos < -0.5:  # evening
        predict MEDIUM delay (1.8 sec/mile)
    ELSE:
        predict LOW delay (1.2 sec/mile)
ELSE:
    ...
```

This is a learned interaction between lag, events, and time of day!

---

## Practical Tips for Your Own Projects

1. **Start simple:** Baseline model with just lag features often gets 80% of the performance
2. **Add domain knowledge:** Time features for temporal data, spatial features for location data
3. **Feature engineering > model selection:** Good features + simple model usually beats bad features + complex model
4. **Visualize your features:** Plot distributions, correlations, and relationships with target
5. **Iterate:** Add features → check importance → remove useless ones → repeat

For this traffic project:
- Lag features gave us baseline RMSE ≈ 10
- Adding time features → RMSE ≈ 7
- Adding road features → RMSE ≈ 6.5
- Adding event features → RMSE ≈ 6.2

Each category mattered!
