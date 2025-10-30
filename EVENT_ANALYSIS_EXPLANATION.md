# Event Analysis: What's Actually Being Used?

## What is Counterfactual Analysis?

**Counterfactual = "What if things were different?"**

It's a way to answer: "How much did X cause Y?" when you can't run a controlled experiment.

### Simple Example

**Question:** Did the roadwork on I-10 cause traffic delays?

**The Problem:** You can't just compare:
- Days with roadwork vs days without roadwork

Why not? Because those days are different in other ways:
- Maybe roadwork happens on weekdays (which are already slower)
- Maybe roadwork happens in summer (which has different traffic patterns)
- Maybe the weather was different

**The Solution - Counterfactual:**
1. Train a model that knows about roadwork → predicts travel time WITH roadwork
2. Train a model that doesn't know about roadwork → predicts what travel time WOULD HAVE BEEN without roadwork
3. Compare the two predictions for the SAME day/time/road

Now you're comparing the same day twice: once as it actually was (with roadwork), and once as it "would have been" (without roadwork).

**The difference = delay caused by roadwork**

### Concrete Example from This Project

**Scenario:** I-10 westbound, Tuesday 5pm, lane closure for paving

**Model 1 (knows about roadwork):**
- Sees: Tuesday, 5pm, rush hour, roadwork present
- Predicts: 85 seconds/mile

**Model 2 (doesn't know about roadwork):**
- Sees: Tuesday, 5pm, rush hour (no roadwork info)
- Predicts: 70 seconds/mile

**Counterfactual estimate:** 85 - 70 = **15 seconds/mile delay caused by roadwork**

### Why This Works

Both models see the SAME conditions:
- Same day (Tuesday)
- Same time (5pm)
- Same road (I-10 westbound)
- Same weather, same season, same traffic patterns

The ONLY difference is whether the model knows roadwork is happening.

So the difference in predictions isolates the effect of roadwork.

### Visual Diagram

```
Real World (what actually happened):
├─ Tuesday 5pm with roadwork
├─ Travel time: 85 sec/mile
└─ Model 1 trained WITH roadwork feature
   └─ Prediction: 85 sec/mile ✓

Counterfactual (what would have happened):
├─ Tuesday 5pm WITHOUT roadwork (hypothetical)
├─ Travel time: ??? (unknown - never happened!)
└─ Model 2 trained WITHOUT roadwork feature
   └─ Prediction: 70 sec/mile (estimate)

Delay caused by roadwork = 85 - 70 = 15 sec/mile
```

### Why Not Just Look at Days Without Roadwork?

**Bad approach:**
```
Compare:
- Days with roadwork: average 85 sec/mile
- Days without roadwork: average 60 sec/mile
- Difference: 25 sec/mile

Problem: These are DIFFERENT days!
- Maybe days with roadwork are weekdays (naturally slower)
- Maybe days without are weekends (naturally faster)
- You're comparing apples to oranges
```

**Good approach (counterfactual):**
```
Compare:
- Same day WITH roadwork (actual): 85 sec/mile
- Same day WITHOUT roadwork (predicted): 70 sec/mile
- Difference: 15 sec/mile

Better: You're comparing the SAME day, just with/without roadwork
```

---

# Event Analysis: What's Actually Being Used?

You're right to be confused - there are two different analyses that use events differently:

## 1. Main Prediction Models (Both Event Types)

**Files:** `train_model_tabular.py`, `train_model_lstm.py`, `train_model_stgnn.py`

**What's included:** BOTH planned AND unplanned events

**Features used:**
```python
evt_features = ['evt_cat_unplanned', 'evt_cat_planned']
```

**Purpose:** Train the best possible travel time prediction model using all available information

**Why include both?**
- Unplanned events (crashes, debris) DO affect traffic
- We want the model to predict accurately when they occur
- Combined importance: 15.9% (planned events dominate this number)

**Results:**
- `evt_cat_planned`: 15.9% importance (3rd most important feature!)
- `evt_cat_unplanned`: Not in top 10 (very low importance)

---

## 2. Counterfactual Delay Analysis (Only Planned Events)

**File:** `evt_impact_analysis.py`

**What's included:** ONLY planned events

**Why exclude unplanned?**

From the script comments (lines 10-16):
```
To answer questions that agencies may care:
1. What's the best time to do road construction work such that
   construction-related delay can be minimized?
2. How would improving unplanned event reporting improve
   travel-time reliability?
```

**The analysis:**
1. Train "no_evt" model → excludes ALL event data (no planned, no unplanned)
2. Train "plnd_evt" model → includes ONLY planned event data
3. Compare predictions → difference = delay caused by planned events

**Why only planned events in counterfactual?**

Three reasons:

### 1. Data Quality Issues with Unplanned Events
- Underreported: Many crashes missing from AZ511 (mentioned in README)
- Batch reporting: Updates come in 3-hour waves, not real-time
- Inconsistent: Severity missing 62%, cryptic codes like "C34Rshoulder"

If unplanned event data is unreliable, you can't trust a counterfactual analysis based on it.

### 2. Policy Relevance
Road agencies (like ADOT) CAN control planned events:
- When to schedule roadwork
- Which lanes to close
- How long construction lasts

They CANNOT control unplanned events (when crashes happen).

So the counterfactual answers: **"If we reschedule this roadwork to 2am instead of 5pm, how much delay do we save?"**

This is actionable information for agencies.

### 3. Causality Concerns
For counterfactual analysis, you need clean data:
- **Planned events**: Have definite start/end times, clear locations
- **Unplanned events**: Reported late, location imprecise, duration uncertain

The counterfactual asks: "What if this event didn't happen?"
- For planned events: Clear comparison (with roadwork vs without)
- For unplanned events: Messy (was it even reported correctly?)

---

## Summary Table

| Analysis | Planned Events | Unplanned Events | Purpose |
|----------|----------------|------------------|---------|
| **Main Models** (train_model_*.py) | ✓ Used | ✓ Used | Best prediction accuracy |
| **Feature Importance** | 15.9% | <2% | Understand what matters |
| **Counterfactual** (evt_impact_analysis.py) | ✓ Analyzed | ✗ Excluded | Policy recommendations |

---

## What the README Says

From your README (Modeling Approach section):

> "The best-performing model and feature combination were selected to conduct a
> counterfactual analysis, enabling estimation of **event-induced delay** in the
> absence of a direct control group (i.e., no-event conditions)."

This is specifically about **planned events** because:
1. The script is called `evt_impact_analysis.py` and focuses on planned events
2. The results section talks about "planned roadwork adds 10-15 sec/mile delay"
3. The recommendations discuss "when to schedule construction"

---

## Where You Might Update Documentation

To make this clearer, you could:

### 1. In README - Event Reporting Section
Current:
> "To ensure analytical consistency, we manually reclassified events into two
> categories—planned (e.g., work zones, closures) and unplanned (e.g., crashes,
> incidents)—based on their subtype descriptions."

Add:
> "For predictive modeling, we include both planned and unplanned events as features.
> However, our counterfactual delay analysis focuses exclusively on planned events
> due to data quality concerns and policy relevance."

### 2. In README - Modeling Approach Section
Current:
> "conduct a counterfactual analysis, enabling estimation of event-induced delay"

Change to:
> "conduct a counterfactual analysis, enabling estimation of **planned event-induced delay**"

### 3. In evt_impact_analysis.py docstring
Already says "planned events" - this is clear!

---

## Bottom Line

**Prediction models:** Use everything (planned + unplanned) to predict as accurately as possible

**Delay analysis:** Only look at planned events because:
- Unplanned event data is unreliable (underreported, delayed, inconsistent)
- Agencies can only control planned events anyway
- You need clean data for counterfactual comparisons

This is a smart choice - you're not ignoring unplanned events, you're just being honest about the limitations of that data for causal analysis.
