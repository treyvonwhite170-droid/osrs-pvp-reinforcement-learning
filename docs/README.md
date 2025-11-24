# OSRS PvP Reinforcement Learning - Technical Documentation

This directory contains comprehensive technical documentation for the OSRS PvP Reinforcement Learning project.

## Documentation Index

### ✅ Completed Documentation

1. **[01-project-overview-architecture.md](01-project-overview-architecture.md)**
   - High-level system architecture
   - Component breakdown (ML system, simulation, contracts)
   - Data flow and communication protocol
   - Scalability and performance characteristics
   - Key design decisions

2. **[02-environment-system.md](02-environment-system.md)**
   - PvpEnv class deep dive
   - Observation space (120+ features)
   - Action space (11 action heads)
   - Action dependencies and masking
   - Episode lifecycle
   - Reward calculation (20+ components)
   - Advanced features (normalization, noise, full observability)

### 📋 Planned Documentation

3. **03-ppo-algorithm.md**
   - PPO mathematical foundation
   - Implementation details
   - Loss functions (policy, value, entropy)
   - Advantage estimation (GAE)
   - Clipping mechanism
   - Hyperparameters

4. **04-neural-network-architecture.md**
   - Policy network design
   - Actor-Critic architecture
   - Feature extraction
   - Autoregressive actions
   - Action head networks
   - TorchScript optimization

5. **05-training-pipeline.md**
   - Training workflow
   - Rollout collection
   - Buffer management
   - Learning updates
   - Callback system
   - Distributed training

6. **06-reward-system.md**
   - Reward components breakdown
   - Reward shaping principles
   - Reward scheduling
   - Impact on learning
   - Reward ablation studies

7. **07-self-play-strategy.md**
   - Self-play implementation
   - Past self-play (prioritized)
   - League system
   - Opponent selection
   - Exploiter training

8. **08-buffer-data-collection.md**
   - Buffer structure
   - GAE computation
   - Reward normalization
   - Novelty rewards
   - Batch generation

9. **09-simulation-rsps.md**
   - Java simulation architecture
   - RemoteEnvironmentServer
   - Environment implementations
   - Combat mechanics
   - Plugin system

10. **10-code-structure.md**
    - Key classes and their responsibilities
    - File organization
    - Import dependencies
    - Extension points

## Quick Reference

### Key Concepts

**Reinforcement Learning:**
- Agent learns through trial and error
- Receives observations, takes actions, gets rewards
- Goal: Maximize cumulative reward

**PPO (Proximal Policy Optimization):**
- On-policy RL algorithm
- Clips policy updates to prevent large changes
- Balances exploration vs exploitation

**Self-Play:**
- Agent trains by playing against itself
- Difficulty increases naturally as agent improves
- Creates diverse opponent pool from past versions

**Action Space:**
- MultiDiscrete: 11 categorical choices per tick
- Total combinations: 4×3×3×4×5×2×2×2×2×5×6 = 172,800
- Action masking reduces to ~100-1000 valid per state

**Observation Space:**
- 120+ continuous features
- Frame stacking for temporal context
- Partial observability for actor (realistic)
- Full observability for critic (training aid)

### File Locations

**Training:**
- Main script: `pvp-ml/pvp_ml/train.py`
- Configs: `pvp-ml/config/*.yml`
- Models: `pvp-ml/models/`

**Environment:**
- PvpEnv: `pvp-ml/pvp_ml/env/pvp_env.py`
- Contracts: `contracts/environments/NhEnv.json`

**PPO:**
- Algorithm: `pvp-ml/pvp_ml/ppo/ppo.py`
- Policy: `pvp-ml/pvp_ml/ppo/policy.py`
- Buffer: `pvp-ml/pvp_ml/ppo/buffer.py`

**Simulation:**
- Plugin: `simulation-rsps/ElvargServer/src/main/java/com/github/naton1/rl/ReinforcementLearningPlugin.java`
- Server: `simulation-rsps/ElvargServer/src/main/java/com/github/naton1/rl/RemoteEnvironmentServer.java`

### Common Commands

**Training:**
```bash
# Start training
train --preset PastSelfPlay --name my-experiment

# Continue training
train --preset PastSelfPlay --name my-experiment --continue-training

# Distributed training
train --preset PastSelfPlay --name my-experiment --distributed-rollouts --num-distributed-rollouts 10
```

**Evaluation:**
```bash
# Evaluate model
eval --model-path models/FineTunedNh

# Serve API
serve-api
```

**Simulation:**
```bash
# Start server
cd simulation-rsps/ElvargServer
./gradlew run
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                            │
│                                                             │
│  1. Collect Rollouts (N environments × M steps)            │
│     ├─ Get observations from simulation                    │
│     ├─ Policy predicts actions                             │
│     ├─ Execute actions in simulation                       │
│     └─ Store (obs, action, reward) in buffer               │
│                                                             │
│  2. Compute Advantages (GAE)                                │
│     ├─ Bootstrap truncated episodes                        │
│     ├─ Calculate TD errors                                 │
│     └─ Compute advantages & returns                        │
│                                                             │
│  3. Update Policy (PPO)                                     │
│     ├─ Sample minibatches from buffer                      │
│     ├─ Compute policy & value loss                         │
│     ├─ Backprop gradients                                  │
│     └─ Clip updates                                        │
│                                                             │
│  4. Update Normalizations                                   │
│     ├─ Update observation mean/std                         │
│     ├─ Update reward normalization                         │
│     └─ Update action normalization                         │
│                                                             │
│  5. Callbacks                                               │
│     ├─ Checkpoint model                                    │
│     ├─ Launch past self-play                               │
│     ├─ Evaluate vs baseline                                │
│     └─ Log metrics to TensorBoard                          │
│                                                             │
│  6. Repeat                                                  │
└─────────────────────────────────────────────────────────────┘
```

### Performance Metrics

**Achieved Results:**
- **99% win rate** in Last Man Standing
- **#1 ranked** in PvP Arena
- **10,000+ rating** (first player to achieve)

**Training Stats:**
- ~1000 rollouts for GeneralizedNh
- ~200 additional rollouts for FineTunedNh
- ~1-2 weeks on single GPU + 32 CPU cores
- ~10M environment steps total

**Model Stats:**
- ~500K parameters
- ~0.5ms inference time (TorchScript)
- ~50MB model size

## Contributing to Documentation

To add or update documentation:

1. Follow the existing format and style
2. Include code examples where relevant
3. Use clear section headers
4. Add diagrams for complex concepts
5. Cross-reference related documents

## Questions?

If you have questions about the codebase:
1. Check if it's covered in existing documentation
2. Look at the code comments
3. Review tests for usage examples
4. Open a GitHub issue

## Additional Resources

- [PPO Paper](https://arxiv.org/pdf/1707.06347.pdf)
- [OpenAI Five Paper](https://arxiv.org/abs/1912.06680) (self-play)
- [AlphaStar Paper](https://www.nature.com/articles/s41586-019-1724-z) (SC2)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
