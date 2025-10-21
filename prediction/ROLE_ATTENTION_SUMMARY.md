# Role-Specific Attention Enhancement

## Base Model
- **Notebook:** GRU-Conv1D-Attention  
- **Proven LB Score:** 0.609  
- **Architecture:** 3-layer GRU (192 hidden) + Conv1D (128) + Multi-head Attention  
- **Features:** 82 engineered features (physics, sequences, rolling stats, ball geometry)

## Enhancement: Role-Specific Attention Queries

### Motivation
Different player roles exhibit fundamentally different trajectory patterns:
- **Targeted Receivers:** Run precise, planned routes with sharp cuts
- **Defensive Coverage:** Track and pursue with reactive, adaptive movement
- **Other players:** More varied movement patterns (blockers, rushers, etc.)

### Implementation

#### 1. Model Architecture Change
**Before:**
```python
self.pool_query = nn.Parameter(torch.randn(1, 1, 192))  # Single query
q = self.pool_query.expand(B, -1, -1)
ctx, _ = self.pool_attn(q, h_norm, h_norm)
```

**After:**
```python
self.role_queries = nn.Parameter(torch.randn(3, 1, 192))  # 3 role queries
queries = torch.stack([self.role_queries[role_labels[i], 0, :] for i in range(B)]).unsqueeze(1)
ctx, _ = self.pool_attn(queries, h_norm, h_norm)
```

#### 2. Role Label Extraction
Added to `prepare_combined_features()`:
```python
is_receiver = input_window.iloc[-1]['is_receiver']
is_coverage = input_window.iloc[-1]['is_coverage']
if is_receiver > 0.5:
    role_label = 1  # Targeted Receiver
elif is_coverage > 0.5:
    role_label = 2  # Defensive Coverage
else:
    role_label = 0  # Other
```

#### 3. Pipeline Updates
- `train_model_combined()`: Now accepts `role_train` and `role_val` parameters
- Training/validation loops: Pass `role_labels` to model forward pass
- Test inference: Extract and pass test role labels
- Logging: Added role distribution statistics

### Expected Impact

**Conservative Estimate:** +0.5-1.5% improvement
- Base LB: 0.609
- Target LB: 0.606-0.603

**Rationale:**
1. Role-specific attention allows specialization without adding parameters to prediction head
2. TR and DC have most distinctive patterns → highest potential gain
3. Minimal architectural change → low risk of degradation
4. Attention mechanism naturally learns to weight features differently per role

### Technical Details

**Parameters Added:**
- 3 role queries × 1 × 192 = 576 parameters (negligible overhead)

**Computational Cost:**
- Additional: Role label tensor creation + indexing (O(B))
- Unchanged: Attention computation complexity

**Training:**
- Same hyperparameters as base model
- No additional regularization needed
- Early stopping based on RMSE (unchanged)

## Verification Checklist

✅ Model accepts `num_roles=3` parameter  
✅ Model has `self.role_queries` instead of `self.pool_query`  
✅ Forward method signature: `def forward(self, x, role_labels)`  
✅ Role-specific query selection implemented  
✅ Feature engineering extracts and returns `role_labels`  
✅ Training function accepts `role_train` and `role_val`  
✅ Training/validation loops pass role labels to model  
✅ Test pipeline extracts and uses role labels  
✅ Role distribution logging added  

## Next Steps

1. **Upload to Kaggle** and enable GPU
2. **Run with full training** (120 epochs, 5 folds)
3. **Monitor metrics:**
   - OOF RMSE per fold
   - Overall OOF RMSE
   - Role-specific performance (if possible)
4. **Submit and compare** to base 0.609

## Potential Follow-ups

If successful:
- Ensemble with base model (0.609) for additional gain
- Try 4-5 role categories (split "Other" into Passer, Rusher, Blocker)
- Add role-specific prediction heads (more aggressive)
- Role-aware loss weighting

If neutral/negative:
- Revert to base model
- Try simpler role encoding (add role as features instead)
- Investigate if roles are already captured by existing features
