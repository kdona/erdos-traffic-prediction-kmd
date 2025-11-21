# LSTM Validation Split Bug - The REAL Overfitting Issue

## Critical Finding: Validation Split Was Broken!

You reported seeing "crazy overtraining" even after the `shuffle=False` fix. Analysis of your metrics revealed the true problem:

```
Val RMSE: 1.47
Test RMSE: 6.92
Gap: 5.45 ← SEVERE OVERFITTING!
```

This is **NOT caused by shuffle** - both shuffle=True and shuffle=False show the same issue!

---

## The Real Bug (Lines 479-482 in `train_model_lstm.py`)

### **Original Code (WRONG):**
```python
# Line 479-482 (OLD - BROKEN)
n_val = max(1, int(0.2 * len(X_train)))
X_val, y_val = X_train[-n_val:], y_train[-n_val:]
X_train_sub, y_train_sub = X_train[:-n_val], y_train[:-n_val]
```

### **Why This is Catastrophically Wrong:**

1. `chronological_split_per_tmc()` splits each TMC chronologically
2. Then concatenates all TMCs' train sequences into one big array
3. **Order depends on TMC iteration** (arbitrary!)
4. `X_train[-n_val:]` takes last 20% of **ARRAY**, not last 20% of **TIME**

### **Visual Example:**

```
After per-TMC chronological split:
  TMC_A: [time 0-80]  (test: 81-100)
  TMC_B: [time 0-80]  (test: 81-100)
  TMC_C: [time 0-80]  (test: 81-100)

X_train concatenation:
  [TMC_A:0-80, TMC_B:0-80, TMC_C:0-80]
   ↑             ↑             ↑
  Early        Early        Early/Mid times

X_train[-20%] selects:
  Mostly sequences from TMC_C (times 0-80)
  ← These are EARLY times, not LATE times!

Test set contains:
  All TMCs (times 81-100)
  ← These are LATE times

RESULT: Val set from times 0-80, Test set from times 81-100
        → Distribution shift! → Val RMSE 1.47, Test RMSE 6.92
```

---

## The Fix (Applied)

### **New Code (CORRECT):**
```python
# Lines 479-493 (NEW - FIXED)
# Validation split (chronologically last 20% of training TIME, not array indices!)
val_frac = 0.2
sorted_times = np.sort(meta_train['time_bin'].unique())
n_time_bins = len(sorted_times)
val_time_threshold = sorted_times[int(n_time_bins * (1 - val_frac))]

# Split based on time_bin
val_mask = meta_train['time_bin'] >= val_time_threshold
train_mask = ~val_mask

X_train_sub, y_train_sub = X_train[train_mask], y_train[train_mask]
X_val, y_val = X_train[val_mask], y_train[val_mask]

log(f"Validation split: {val_mask.sum()} val sequences from time >= {val_time_threshold}")
```

### **How the Fix Works:**

1. Gets all unique time bins in training data
2. Finds the chronological threshold (80th percentile of time)
3. **Splits ALL TMCs based on time**, not array position
4. Validation set = all sequences from last 20% of TIME across all TMCs
5. Now val and test are from similar time periods!

---

## Expected Results After Fix

### **Before Fix (Your Current Results):**
| Metric | Value | Problem |
|--------|-------|---------|
| Val RMSE | 1.47 | From early times |
| Test RMSE | 6.92 | From late times |
| Gap | 5.45 | Distribution shift! |

### **After Fix (Expected):**
| Metric | Expected Value | Why |
|--------|---------------|-----|
| Val RMSE | **~4.0-4.5** | From late training times |
| Test RMSE | **~4.0-4.5** | From test times (adjacent) |
| Gap | **< 1.0** | Healthy! No distribution shift |

The validation RMSE should **INCREASE** from 1.47 to ~4.0 because it's now being evaluated on harder (later) time periods that are more similar to the test set.

---

## How to Retrain

The fix has been applied to your code. Now retrain:

```bash
# In environment with TensorFlow/Keras installed
python train_model_lstm.py \
  --save-models \
  --output-dir models/lstm_truly_fixed \
  --batch-size 32 \
  --dropout 0.3 \
  --patience 5
```

### **Check the Output:**

Look for this new log line:
```
Validation split: XXXX val sequences from time >= 2025-XX-XX
```

This confirms the chronological split is working!

---

## Verification Steps

### 1. **Check Val/Test Gap:**
```bash
cat models/lstm_truly_fixed/metrics.json | grep -E "(best_val|rmse)"
```

Expected:
```json
{
  "best_val": 16.0-20.0,  ← sqrt = 4.0-4.5 RMSE
  "rmse": 4.0-4.5,        ← Test RMSE
  ...
}
```

**The gap should be < 1.0 RMSE now!**

### 2. **Check Learning Curves:**

Plot val_loss vs train_loss:
```python
import json
import matplotlib.pyplot as plt
import numpy as np

with open('models/lstm_truly_fixed/metrics.json') as f:
    m = json.load(f)

train_rmse = [np.sqrt(x) for x in m['train_loss']]
val_rmse = [np.sqrt(x) for x in m['val_loss']]

plt.plot(train_rmse, label='Train RMSE')
plt.plot(val_rmse, label='Val RMSE')
plt.axhline(np.sqrt(m['best_val']), color='g', linestyle='--', label='Best Val')
plt.axhline(m['rmse'], color='r', linestyle='--', label='Test')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Learning Curves - Should Converge Around 4.0')
plt.show()
```

Expected: Both curves converge around 4.0-4.5 RMSE

### 3. **Compare with Teammates:**

Your results should now match your teammates:
- All models: ~4.0 RMSE
- Similar train/val/test performance
- No crazy gaps

---

## Why This Bug Existed

This is a **subtle time series bug** that's easy to miss:

1. **Per-TMC split is correct** ✓
   - Each TMC split chronologically

2. **Test set is correct** ✓
   - Each TMC's last 20% chronologically

3. **But validation split was WRONG** ✗
   - Took last 20% of concatenated array
   - Array order is arbitrary (depends on TMC iteration)
   - Not chronologically meaningful!

---

## Summary

| Issue | Status | Fix Location |
|-------|--------|--------------|
| Shuffle bug | Fixed earlier | Line 498: `shuffle=False` |
| **Validation split bug** | **FIXED NOW** | Lines 479-493: Chronological time split |
| Expected outcome | Pending retrain | Val RMSE ~4.0, Test RMSE ~4.0 |

**The validation split fix is the critical one!**

Without this fix, early stopping and model selection were based on misleading validation performance that didn't reflect actual test performance.

---

## Next Steps

1. **Retrain LSTM** with the fixed code (requires TensorFlow environment)
2. **Verify** Val RMSE ≈ Test RMSE ≈ 4.0
3. **Compare** with teammate's results (should match now!)
4. **Update** model comparison plots

**Your ST-GNN (4.37 RMSE) was correct all along!**
**Your tabular models (2.4-2.9 RMSE) were correct all along!**
**Only LSTM had broken validation!**
