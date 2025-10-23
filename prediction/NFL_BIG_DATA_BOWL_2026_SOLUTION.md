# NFL Big Data Bowl 2026 - Complete Solution

## 🎯 Competition Overview

**Goal**: Predict player movement (x,y coordinates) during ball flight time after throw  
**Evaluation**: RMSE (lower is better) - Current leader: 0.50, Target: <0.50  
**Data**: 2023 season training data, predict 2024 season weeks 14-18  
**Timeline**: Forecasting competition with live leaderboard updates  
**Current Status**: TTA-enhanced GRU model at ~0.605 RMSE  

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

## 🚀 New Approach: Physics-Informed Residual Trajectory Network (PIRTN)

Goal: materially beat ~0.60 RMSE by combining a strong kinematic prior with learned residuals and interaction/context features.

- Physics prior: steered-kinematics simulator from last pre-throw frame towards `(ball_land_x, ball_land_y)` with speed cap, acceleration cap, and turn-rate limits. Produces per-frame baseline paths for every player.
- Residual learner: models predict small corrections to the physics prior rather than absolute positions. Start with GPU CatBoost (fast, robust); optional GRU/attention head for sequence refinement if time permits.
- Interaction signals: lightweight graph features (ally/opponent proximity, relative velocity) computed at the throw frame.
- Context: merge `analytics/data/supplementary_data.csv` (down, distance, formation, coverage, route_of_targeted_receiver, etc.).
- Hierarchical decoding: predict sparse waypoints every ~10 frames and interpolate, reducing error accumulation for longer flights.
- TTA/smoothing: optional small gains via horizontal flip and temporal smoothing at inference.

Expected impact: kinematic prior (~0.62→0.60), context/interaction (-0.01 to -0.02), hierarchical decoding (-0.01), residual learner capacity (-0.01). Combined: target 0.54-0.57; with tuning, aim <0.52.

### Feature Set (current)
- Kinematics (throw frame): `x, y, s, a, dir, o` plus components
  - `velocity_x, velocity_y, acceleration_x, acceleration_y`
  - Parallel/perpendicular projections to ball vector: `velocity_parallel, velocity_perpendicular, acceleration_parallel, acceleration_perpendicular`
  - Energy/momentum/body: `speed_squared, accel_magnitude, momentum_x, momentum_y, kinetic_energy, height_inches, bmi`
- Formation & field geometry
  - Team shape and relative placement: `team_centroid_{x,y}, team_width, team_length, rel_centroid_{x,y}`
  - Formation bearing to landing point: `formation_bearing_{sin,cos}`
- Geometric endpoint features (inspired by the new leading notebook)
  - Deterministic endpoint and path descriptors:
    - `geo_time_to_endpoint, geo_endpoint_{x,y}, geo_vector_{x,y}, geo_distance`
    - Required motion to reach endpoint: `geo_required_vx, geo_required_vy, geo_required_ax, geo_required_ay`
    - Deviation/alignment to geometric path: `geo_velocity_error_{x,y}, geo_velocity_error, geo_alignment`
- Interaction (GNN-lite, last pre-throw frame)
  - Weighted neighbor aggregates (ally and opponent):
    - Displacements/relative velocities: `gnn_ally/opp_{dx,dy,dvx,dvy}_mean`
    - Counts and distances: `gnn_ally_cnt, gnn_opp_cnt, gnn_ally/opp_{dmin}` (and optional `gnn_d1..d3` if enabled)
- Temporal structure (player-level)
  - Lags: `x,y,velocity_x,velocity_y,s,a,...` for lags 1–5
  - Rolling stats: mean/std windows 3 and 5 for core motion vars
  - First differences: `velocity_{x,y}_delta, velocity_{parallel,perpendicular}_delta`
- Play context (if available; safe-encoded)
  - `down, yards_to_go, pass_length, offense_formation, team_coverage_type, team_coverage_man_zone, route_of_targeted_receiver, play_action, dropback_type`
- Prediction time indexing
  - `delta_frames, delta_t, waypoint_idx, is_waypoint`

### Implementation Notes
- Train residuals: target = ground_truth − physics_prior; clip to field after recomposition.
- Role awareness: include `player_role` dummies; the targeted receiver and nearest defenders may receive stronger corrections.
- Robust paths: auto-detect Kaggle vs local data roots; use GPU CatBoost when available.

### Notes vs leader approach
- We incorporate their geometric endpoint intuition directly as model features while retaining a physics prior. This reduces residual variance and improves stability.
- Additional opponent/mirroring and route features are compatible; we will add route clustering next (KMeans on short-window route stats) to further close the gap.

### Next Steps After First Submission
- Tune prior (v_max, a_max, turn_limit) per role; learn them by minimizing OOF RMSE.
- Add GRU/attention residual head with teacher forcing; multi-task (velocity) auxiliary loss.
- Expand interaction window with temporal attention or per-frame graph pooling.