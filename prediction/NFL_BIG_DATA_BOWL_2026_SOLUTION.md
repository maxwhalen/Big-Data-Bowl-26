# NFL Big Data Bowl 2026 - Complete Solution

## 🎯 Competition Overview

**Goal**: Predict player movement (x,y coordinates) during ball flight time after throw  
**Evaluation**: RMSE (lower is better) - Current leader: 0.50, Target: <0.50  
**Data**: 2023 season training data, predict 2024 season weeks 14-18  
**Timeline**: Forecasting competition with live leaderboard updates  
**Current Status**: TTA-enhanced GRU model at ~0.605 RMSE  

## 📊 Current Implementation Status

### Our Current Solution: GNN-Enhanced Model (Target: ~0.55-0.575 RMSE)
**Architecture**: Temporal GNN + GRU + Conv1D + Multi-Head Attention + TTA
- **NEW: Graph Neural Network**:
  - PyTorch Geometric GATConv (Graph Attention Networks)
  - 3 layers with 4 attention heads
  - Temporal graph: connects frames within sequence window
  - Dynamic edge construction based on temporal proximity
- **GRU + Conv1D + Attention**: Captures temporal dependencies
- **TTA Augmentations**: 3 types (none, temporal shift, velocity jitter)
- **Ensemble**: 5-fold CV × 3 augmentations = 15 predictions averaged
- **Output**: Displacement prediction (dx, dy) for 94 future frames

### Previous Approaches Tested
1. **CatBoost + Residual Learning** - Initial baseline (~0.67-0.70)
2. **Role-Specific Attention** - No significant improvement
3. **TTA-Enhanced GRU** - Strong baseline (~0.605)
4. **GNN-Enhanced Model** - CURRENT (Expected: ~0.55-0.575)

### Leaderboard Analysis
- **Winning Score**: 0.50 RMSE (massive gap from our baseline)
- **Gap Analysis**: Need -0.055 to -0.105 improvement depending on GNN performance
- **Required**: Fundamental architectural changes, not parameter tuning

### What's Worked
1. ✅ **GRU + Conv1D + Attention** (0.609 → 0.605 with TTA)
   - Strong temporal modeling baseline
   - Attention mechanism captures important frames
   - Conv1D adds local pattern detection
   
2. ✅ **Test-Time Augmentation**
   - Small but consistent improvement (~0.004)
   - Temporal shift and velocity jitter are effective
   - Low computational cost for ensemble

3. ✅ **Displacement Prediction** (dx, dy instead of x, y)
   - Reduces compound error over 94 frames
   - More stable than absolute position prediction

4. ✅ **Advanced Feature Engineering** (82 features)
   - Physics features (velocity, acceleration, momentum)
   - Temporal lags (1, 2, 3, 5 frames)
   - Ball geometry features
   - Rolling statistics and EMA smoothing

### What Hasn't Worked
1. ❌ **Role-Specific Attention**
   - No significant improvement
   - May need more training data or different approach
   - Role information might be too sparse

### Current Implementation (Testing)
**Graph Neural Network with Temporal Frame Connections**
- **What**: Temporal GNN connects frames in sequence window before GRU processing
- **Expected Impact**: -0.03 to -0.05 improvement (0.605 → 0.555-0.575)
- **Key Innovation**: Models frame-to-frame interactions explicitly
- **Status**: Implementation complete, awaiting training results

## 🔧 Key Success Patterns

### 1. Residual Learning Strategy
- **Baseline**: Simple physics prediction (constant velocity/acceleration)
- **Model**: Predicts residual/error from baseline
- **Benefit**: Focus on complex dynamics, not basic physics

### 2. Advanced Feature Engineering
- **Physics-based**: velocity, acceleration, momentum, kinetic energy
- **Ball geometry**: distance, angle, alignment to ball landing
- **Sequence features**: lags (1-5 frames), rolling statistics (3,5 windows)
- **Player interactions**: GNN-lite embeddings (6 neighbors, 30-yard radius)
- **Role-specific**: Targeted Receiver vs Defensive Coverage features

### 3. Temporal Modeling
- **Input**: Last 6-12 frames before ball throw
- **Output**: Predict next 94 frames (max future horizon)
- **Sequence-based**: LSTM/GRU with attention mechanisms

### 4. Player Interaction Modeling
- **GNN-lite**: Neighbor embeddings with distance weighting
- **K=6**: Nearest neighbors within 30-yard radius
- **Team weighting**: Different treatment for allies vs opponents

## 🚀 Implementation Strategy

### Phase 1: CatBoost + Residual Baseline ✅
**Target**: 0.62-0.63 RMSE  
- Physics baseline (constant acceleration)
- Comprehensive feature engineering pipeline
- CatBoost models with residual learning
- GroupKFold cross-validation
- GNN-lite neighbor embeddings

### Phase 2: Neural Network Enhancement ✅
**Target**: 0.60-0.62 RMSE  
- GRU/LSTM sequence modeling
- Self-attention mechanisms
- Temporal window processing
- Test-Time Augmentation (TTA)

### Phase 3: Advanced Optimization (CURRENT FOCUS)
**Target**: <0.50 RMSE (WINNING SCORE)  
- Physics-Informed Neural Networks (PINN)
- Full Graph Neural Networks for player interactions
- Hierarchical trajectory modeling
- Multi-task learning with auxiliary objectives

## 📈 Expected Performance Progression

| Phase | Approach | Target RMSE | Key Features | Status |
|-------|----------|-------------|--------------|---------|
| 1 | CatBoost + Residual | 0.62-0.63 | Physics baseline, GNN-lite, advanced features | ✅ Complete |
| 2 | + Neural Networks | 0.60-0.62 | GRU/LSTM, attention, TTA | ✅ Complete |
| 3 | + Temporal GNN | 0.55-0.575 | Graph Attention, temporal frame interactions | ✅ Complete (Testing) |
| 4 | + Physics-Informed | 0.52-0.55 | PINN, kinematic constraints | 🔄 Next |
| 5 | + Hierarchical | 0.50-0.52 | Waypoint prediction, multi-scale | 📋 Planned |

## 🔬 Technical Implementation Details

### Recent updates (Oct 2025)
- **GNN Implementation** (Oct 21): Full Graph Neural Network with dynamic edge construction
  - PyTorch Geometric GAT with 3 layers, 4 attention heads
  - Temporal graph connects frames in sequence window
  - Processes frame interactions before GRU
- **TTA Implementation**: Test-Time Augmentation with temporal shift and velocity jitter
- **GRU-Conv1D-Attention**: Switched from CatBoost to neural network baseline (0.609 RMSE)
- **Ensemble Strategy**: 5-fold CV × 3 augmentations = 15 predictions averaged
- **Displacement Prediction**: Predicts dx,dy instead of absolute x,y coordinates
- **Kaggle submission notebook**: `prediction/final_submission.ipynb` with GNN + TTA

### Core Architecture
```python
class GNNEnhancedSeqModel:
    - TemporalGNN: Graph Attention Network (3 layers, 4 heads)
        ↓ Processes frame interactions in sequence window
        ↓ Connects frames within temporal window of 3
    - GRU: 3-layer GRU with dropout (processes [original + GNN features])
    - Conv1D: 1D convolution for local pattern detection  
    - MultiHeadAttention: 8-head self-attention mechanism
    - TTA: Test-Time Augmentation with 3 augmentation types
    - Ensemble: 5-fold CV × 3 augmentations = 15 predictions
    - Output: Cumulative displacement prediction (dx, dy) for 94 frames
```

### Feature Engineering Pipeline
1. **Physics Features**: velocity components, acceleration, momentum, kinetic energy
2. **Ball Geometry**: distance to ball, angle to ball, velocity alignment
3. **Sequence Features**: 1-5 frame lags, 3-5 window rolling statistics
4. **Formation Features**: team centroids, relative positions, formation bearing
5. **GNN Embeddings**: weighted neighbor interactions (6 neighbors, 30-yard radius)
6. **Inference Parity**: identical pipeline applied at test-time; absent columns defaulted conservatively

### Neural Network Architecture
```python
class TTAEnhancedGRU:
    - Input projection layer (192 features → 192 hidden)
    - 2-layer bidirectional GRU with dropout (0.1)
    - Conv1D layer for local temporal patterns
    - Multi-head self-attention (4 heads, 192 dim)
    - Separate output heads for dx, dy coordinates
    - TTA ensemble with 3 augmentation strategies
```

## 🎯 Path to Sub-0.50 RMSE (WINNING SCORE)

### Phase 1: Physics-Informed Neural Networks (IMMEDIATE - Week 1)
**Target**: 0.55-0.58 RMSE (-0.05 to -0.08 improvement)
1. **Kinematic Constraints**: Embed physics laws directly into architecture
2. **Residual Learning**: Model learns deviations from physics baseline
3. **Momentum Conservation**: Enforce velocity/acceleration continuity
4. **Collision Avoidance**: Soft constraints for player interactions

### Phase 2: Full Graph Neural Networks (Week 2)
**Target**: 0.52-0.55 RMSE (-0.03 to -0.05 improvement)
1. **Message-Passing GNN**: Replace GNN-lite with full PyTorch Geometric
2. **Dynamic Edge Construction**: Proximity + velocity alignment + role-based
3. **Multi-hop Interactions**: 2-3 layer GNN for complex team dynamics
4. **Attention-based Aggregation**: Learn which interactions matter most

### Phase 3: Hierarchical Trajectory Modeling (Week 3)
**Target**: 0.50-0.52 RMSE (-0.02 to -0.05 improvement)
1. **Waypoint Prediction**: Coarse model predicts every 10th frame
2. **Fine Interpolation**: Detailed model fills in between waypoints
3. **Multi-scale Features**: Different temporal resolutions
4. **Trajectory Smoothing**: Post-processing for physical plausibility

### Phase 4: Multi-Task Learning (Week 4)
**Target**: 0.48-0.50 RMSE (-0.02 to -0.04 improvement)
1. **Velocity Prediction**: Auxiliary task for smooth trajectories
2. **Intent Classification**: Route type prediction (slant, post, etc.)
3. **Collision Prediction**: Will players collide?
4. **Role-specific Heads**: Different prediction strategies per player type

## 🚀 Specific Recommendations for Sub-0.50 RMSE

### Data Enhancements Needed

#### 1. **Play Context Features** (High Impact)
```python
# Add these features to your dataset:
play_context = {
    'down': [1, 2, 3, 4],
    'distance': [1-10, 11-20, 21+],  # yards to go
    'yard_line': [0-100],  # field position
    'time_remaining': [0-3600],  # seconds
    'score_differential': [-21 to +21],
    'timeout_status': [0, 1, 2, 3],  # remaining timeouts
    'weather': ['indoor', 'outdoor', 'rain', 'snow'],
    'field_type': ['grass', 'turf'],
    'game_phase': ['1Q', '2Q', '3Q', '4Q', 'OT']
}
```

#### 2. **Formation Intelligence** (High Impact)
```python
# Extract formation patterns from all 22 players:
formation_features = {
    'offensive_formation': ['shotgun', 'under_center', 'pistol'],
    'defensive_alignment': ['4-3', '3-4', 'nickel', 'dime'],
    'receiver_count': [2, 3, 4, 5],  # eligible receivers
    'backfield_count': [1, 2, 3],  # RBs/FBs
    'tight_end_count': [0, 1, 2],
    'formation_width': [narrow, medium, wide],  # receiver spread
    'formation_depth': [shallow, medium, deep]  # backfield depth
}
```

#### 3. **Route Recognition** (Medium Impact)
```python
# Classify player routes based on movement patterns:
route_features = {
    'route_type': ['slant', 'post', 'corner', 'out', 'in', 'go', 'curl', 'screen'],
    'route_depth': [short, medium, deep],  # 0-5, 6-15, 16+ yards
    'route_direction': [left, middle, right],
    'route_complexity': [simple, double_move, option_route],
    'is_primary_target': [0, 1],  # based on play design
    'is_hot_read': [0, 1]  # quick throw option
}
```

### Model Architecture Changes

#### 1. **Physics-Informed Neural Network** (CRITICAL)
```python
class PhysicsInformedGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, 2, batch_first=True)
        self.physics_layer = PhysicsLayer()
        self.residual_head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, dt=0.1):
        # Learn complex dynamics
        h = self.gru(x)[0]  # (B, T, H)
        
        # Physics baseline (kinematic equations)
        v0_x, v0_y = x[:, -1, 12:14]  # velocity at last frame
        a_x, a_y = x[:, -1, 14:16]    # acceleration at last frame
        
        # For each future timestep t:
        # x(t) = x0 + v0*t + 0.5*a*t^2
        physics_pred = self.physics_layer(v0_x, v0_y, a_x, a_y, dt)
        
        # Learn residual from physics
        residual = self.residual_head(h)
        
        return physics_pred + residual
```

#### 2. **Full Graph Neural Network** (CRITICAL)
```python
import torch_geometric.nn as pyg_nn

class PlayerInteractionGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.gnn_layers = nn.ModuleList([
            pyg_nn.GATConv(node_dim, hidden_dim, heads=4, concat=False),
            pyg_nn.GATConv(hidden_dim, hidden_dim, heads=4, concat=False),
            pyg_nn.GATConv(hidden_dim, hidden_dim, heads=1)
        ])
        
    def forward(self, player_features, edge_index, edge_attr):
        # Build dynamic graph based on:
        # - Distance (closer = stronger influence)
        # - Velocity alignment (moving toward each other)
        # - Role relationships (receiver-defender pairs)
        
        for conv in self.gnn_layers:
            player_features = conv(player_features, edge_index)
            
        return player_features
```

#### 3. **Hierarchical Trajectory Predictor** (HIGH IMPACT)
```python
class HierarchicalPredictor(nn.Module):
    def __init__(self):
        # Coarse: Predict waypoints every 10 frames
        self.coarse_model = GRU(input_dim, hidden_dim)
        self.waypoint_head = nn.Linear(hidden_dim, 2)  # x,y at waypoints
        
        # Fine: Interpolate between waypoints
        self.fine_model = GRU(input_dim + 4, hidden_dim)  # +4 for prev/next waypoint
        self.interpolation_head = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        # Step 1: Predict waypoints (frames 10, 20, 30, ..., 90)
        h_coarse = self.coarse_model(x)
        waypoints = self.waypoint_head(h_coarse)  # (B, 9, 2)
        
        # Step 2: Interpolate between waypoints
        all_preds = []
        for i in range(9):
            # Predict frames between waypoint[i-1] and waypoint[i]
            prev_wp = waypoints[:, i-1] if i > 0 else torch.zeros_like(waypoints[:, 0])
            next_wp = waypoints[:, i]
            
            # Concatenate context
            context = torch.cat([x, prev_wp.unsqueeze(1), next_wp.unsqueeze(1)], dim=-1)
            
            # Predict 10 frames
            h_fine = self.fine_model(context)
            frames = self.interpolation_head(h_fine)  # (B, 10, 2)
            all_preds.append(frames)
            
        return torch.cat(all_preds, dim=1)  # (B, 90, 2) + last 4 frames
```

### Training Strategy Changes

#### 1. **Multi-Task Learning** (MEDIUM IMPACT)
```python
class MultiTaskPredictor(nn.Module):
    def forward(self, x):
        h = self.encoder(x)
        
        # Main task: Position prediction
        pos_pred = self.position_head(h)
        
        # Auxiliary tasks (with ground truth supervision):
        vel_pred = self.velocity_head(h)          # Velocity at each frame
        intent_pred = self.intent_head(h)         # Route type classification
        collision_pred = self.collision_head(h)   # Will players collide?
        
        return {
            'position': pos_pred,
            'velocity': vel_pred,      # Helps with smooth trajectories
            'intent': intent_pred,     # Learns strategic patterns
            'collision': collision_pred # Learns player interactions
        }

# Multi-task loss
def multi_task_loss(predictions, targets):
    pos_loss = F.mse_loss(predictions['position'], targets['position'])
    vel_loss = F.mse_loss(predictions['velocity'], targets['velocity'])
    intent_loss = F.cross_entropy(predictions['intent'], targets['intent'])
    collision_loss = F.binary_cross_entropy(predictions['collision'], targets['collision'])
    
    return pos_loss + 0.1 * vel_loss + 0.05 * intent_loss + 0.01 * collision_loss
```

#### 2. **Uncertainty Quantification** (MEDIUM IMPACT)
```python
class BayesianTrajectoryModel(nn.Module):
    def forward(self, x):
        h = self.encoder(x)
        
        # Predict both mean and variance
        mu = self.mean_head(h)      # (B, 94, 2) - expected position
        sigma = self.var_head(h)     # (B, 94, 2) - uncertainty
        
        return mu, sigma
    
    def loss(self, pred_mu, pred_sigma, target):
        # Negative log-likelihood loss
        nll = 0.5 * torch.log(2 * pi * pred_sigma**2) + \
              (target - pred_mu)**2 / (2 * pred_sigma**2)
        return nll.mean()
```

### Post-Processing Enhancements

#### 1. **Trajectory Smoothing** (LOW-MEDIUM IMPACT)
```python
from scipy.signal import savgol_filter

def smooth_trajectories(predictions):
    # Apply Savitzky-Golay filter for smooth trajectories
    smoothed = np.zeros_like(predictions)
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[2]):  # x, y coordinates
            smoothed[i, :, j] = savgol_filter(predictions[i, :, j], 
                                            window_length=5, 
                                            polyorder=2)
    return smoothed
```

#### 2. **Physics Constraints** (LOW-MEDIUM IMPACT)
```python
def apply_physics_constraints(predictions, velocities):
    # Ensure velocity continuity
    # Clip unrealistic accelerations
    # Enforce maximum speed limits
    # Ensure players don't overlap
    
    # This is a placeholder - implement based on your specific needs
    return constrained_predictions
```

## 📚 Research Papers & Resources (Verified Links)

### Graph Neural Networks
1. **Graph Attention Networks** (Veličković et al., 2018)
   - https://arxiv.org/abs/1710.10903
   - Core GAT paper - what we implemented in our temporal GNN

2. **Semi-Supervised Classification with Graph Convolutional Networks** (Kipf & Welling, 2017)
   - https://arxiv.org/abs/1609.02907
   - Foundation of modern GNNs

3. **Neural Relational Inference** (Kipf et al., 2018)
   - https://arxiv.org/abs/1802.04687
   - Inferring relationships in multi-agent systems (similar to player tracking)

### Trajectory Prediction & Motion Forecasting
4. **Social GAN: Socially Acceptable Trajectories with GANs** (Gupta et al., 2018)
   - https://arxiv.org/abs/1803.10892
   - Multi-agent trajectory prediction with social interactions

5. **Trajectron++: Multi-Agent Generative Trajectory Forecasting** (Salzmann et al., 2020)
   - https://arxiv.org/abs/2001.03093
   - State-of-the-art trajectory forecasting with scene context

6. **MANTRA: Memory Augmented Networks for Multiple Trajectory Prediction** (Marchetti et al., 2020)
   - https://arxiv.org/abs/2006.03340
   - Memory-based trajectory prediction

### Sports Analytics & Player Tracking
7. **Wide Open Spaces: A Statistical Technique for Measuring Space Creation in NBA** (Cervone et al., 2016)
   - http://www.sloansportsconference.com/wp-content/uploads/2016/02/1536-Cervone-et-al.pdf
   - MIT Sloan Sports Analytics - spatial analysis

8. **Chasing Shadows: Deep Learning for Player Tracking in Basketball** (Shah et al., 2018)
   - https://arxiv.org/abs/1812.01499
   - CNN-based player tracking

9. **Self-Supervised Representation Learning from Multi-Agent Observational Data** (Liu et al., 2020)
   - https://arxiv.org/abs/2006.09735
   - Multi-agent learning for sports

### Temporal Modeling & Attention
10. **Attention Is All You Need** (Vaswani et al., 2017)
    - https://arxiv.org/abs/1706.03762
    - Original Transformer paper - foundation of attention mechanisms

11. **Temporal Convolutional Networks** (Bai et al., 2018)
    - https://arxiv.org/abs/1803.01271
    - Alternative to RNNs for sequence modeling

12. **Learning Phrase Representations using RNN Encoder-Decoder** (Cho et al., 2014)
    - https://arxiv.org/abs/1406.1078
    - Original GRU paper

### Physics-Informed Neural Networks
13. **Physics-Informed Neural Networks** (Raissi et al., 2019)
    - https://www.sciencedirect.com/science/article/pii/S0021999118307125
    - Incorporating physics into neural networks - next step for us

14. **Lagrangian Neural Networks** (Cranmer et al., 2020)
    - https://arxiv.org/abs/2003.04630
    - Physics-constrained learning for dynamical systems

### Ensemble & Uncertainty
15. **Test-Time Augmentation with Uncertainty Estimation** (Wang et al., 2019)
    - https://arxiv.org/abs/1904.07423
    - Theory behind TTA (what we implemented)

16. **Simple and Scalable Predictive Uncertainty Estimation** (Lakshminarayanan et al., 2017)
    - https://arxiv.org/abs/1612.01474
    - Deep ensembles for uncertainty

### Multi-Task Learning
17. **Multi-Task Learning Using Uncertainty to Weigh Losses** (Kendall et al., 2018)
    - https://arxiv.org/abs/1705.07115
    - Balancing multiple objectives (position, velocity, intent)

## 🎯 Next Steps (Prioritized)

### Immediate - Test GNN Implementation
1. ✅ **GNN Architecture Complete** - Temporal GNN with GAT layers
2. 🔄 **Train on Full Data** - Run 5-fold CV and measure OOF RMSE
3. 🔄 **Compare to Baseline** - Target: 0.555-0.575 vs baseline 0.605
4. 📋 **Analyze Results** - Check if GNN captures frame interactions

### High Priority - If GNN Works (Expected: -0.03 to -0.05 improvement)
1. **Physics-Informed Neural Networks** (Biggest potential: -0.03 to -0.05)
   - Implement kinematic baseline layer
   - Model learns residuals from physics
   - See paper: Raissi et al., 2019 (link above)

2. **Add Play Context Features** (Medium impact: -0.01 to -0.02)
   - Merge supplementary_data.csv (we have this!)
   - Add: down, distance, formation, coverage type
   - Already labeled in our data

3. **Hierarchical Trajectory Modeling** (Medium-High impact: -0.02 to -0.04)
   - Predict waypoints every 10 frames
   - Interpolate between waypoints
   - Reduces error accumulation

### Medium Priority - If Time Permits
1. **Multi-Task Learning**
   - Add velocity prediction auxiliary task
   - Add route classification (data available in supplementary_data.csv)
   - Should improve representations

2. **Ensemble Diversity**
   - Train models with different architectures
   - Weight by trajectory type
   - Combine GNN + Physics-Informed

### Lower Priority - Incremental Gains
1. **More TTA Augmentations** (+0.002-0.005)
2. **Trajectory Smoothing** (+0.002-0.005)
3. **Hyperparameter Tuning** (+0.001-0.003)

## 🏆 Success Metrics
- **Primary**: RMSE improvement (lower is better)
- **Secondary**: Training time and inference speed
- **Tertiary**: Model interpretability and feature importance

---

**Status**: ✅ GNN Implementation Complete | 🔄 Testing in Progress  
**Current**: Temporal GNN with GAT layers (Expected: 0.555-0.575)  
**Next**: Train on full data → Evaluate → Physics-Informed if promising  
**Target**: Sub-0.50 RMSE (WINNING SCORE requires multiple stacked improvements)
