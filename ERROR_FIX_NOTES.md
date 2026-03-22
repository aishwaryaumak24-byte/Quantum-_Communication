# Pandas KeyError Fix for Comparison Plotting

## Error Description
**Error Message**: `pandas KeyError during index operation` in `evaluator.py` when generating comparison plots

**Location**: In `src/evaluation/evaluator.py` - `plot_comparison()` method

**When It Happens**: After successful strategy comparison, when trying to generate plots with matplotlib and seaborn

## Root Cause
1. **DataFrame column indexing issue**: Accessing pandas Series objects directly instead of converting to numpy arrays
2. **Missing column validation**: No checks to ensure all required columns exist before accessing them
3. **Index alignment problems**: Pandas Series with non-standard indices can cause unexpected behavior when passed to matplotlib
4. **Empty bar array access**: Code attempted to access `bars[0]` without checking if bars array was empty

## The Fix Applied

### Changes Made:
1. **Added DataFrame validation** before plotting:
   - Check if DataFrame is empty
   - Verify all required columns exist
   - Reset index to avoid misalignment

2. **Converted pandas Series to numpy arrays**:
   ```python
   # Before (causes issues):
   strategies = df['strategy']
   rewards = df['mean_reward']
   
   # After (safe):
   strategies = df['strategy'].values.tolist()
   rewards = df['mean_reward'].values
   ```

3. **Added safety checks for bar highlighting**:
   ```python
   if len(bars) > 0:
       bars[0].set_color('green')
   ```

4. **Enhanced error handling**:
   - Try-except blocks in plot saving
   - Better error messages showing available columns
   - Graceful handling of missing baselines

5. **Improved data flow in compare_strategies()**:
   - Try-except around each strategy evaluation
   - Column existence validation before DataFrame selection

## How to Use the Fix

1. **Pull the latest code** with the updated `src/evaluation/evaluator.py`
2. **Re-run your experiment**:
   ```bash
   python experiments/experiment_static.py --num_eval_episodes 100
   ```

## Debugging Tips

If you still encounter issues:

1. **Check the error message** - it will now tell you which columns are missing
2. **Verify CSV output** - the results should save to `results/tables/static_comparison.csv` before plotting fails
3. **Print DataFrame info** by adding this before plotting:
   ```python
   print("DataFrame columns:", df.columns.tolist())
   print("DataFrame shape:", df.shape)
   print("DataFrame dtypes:\n", df.dtypes)
   ```

## Prevention

To avoid similar issues in the future:
- Always convert pandas Series to numpy arrays (`.values`) before plotting
- Always validate column names before accessing
- Use try-except blocks for all I/O operations
- Add debug logging to show DataFrame structure when errors occur

## Questions?

If the error persists after applying this fix, collect:
1. The full error traceback
2. Output from the comparison (the printed table)
3. Contents of `results/tables/static_comparison.csv` (if it was created)
