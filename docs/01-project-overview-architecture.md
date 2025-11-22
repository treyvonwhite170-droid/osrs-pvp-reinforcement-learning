# OSRS PvP Reinforcement Learning - Project Overview & Architecture

## Executive Summary

This project trains an AI agent to master "No Honor" (NH) PvP combat in Old School RuneScape using deep reinforcement learning. The AI achieved:
- **~99% win rate** in Last Man Standing (LMS)
- **#1 rank** in PvP Arena
- **10,000+ rating** (first ever to achieve this)

The system combines three major components working together to enable learning.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING SYSTEM                          │
│                         (pvp-ml/)                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ PPO Trainer  │───▶│    Buffer    │───▶│ Neural Net   │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                                         │             │
│         │ Actions                      Observations│           │
│         ▼                                         ▼             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ TCP Socket (port 7070)
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    GAME SIMULATION                              │
│                   (simulation-rsps/)                            │
│  ┌────────────────────────────────────────────────────┐        │
│  │   RemoteEnvironmentServer (Java)                   │        │
│  │   - Receives actions via socket                    │        │
│  │   - Executes game ticks                            │        │
│  │   - Returns observations & rewards                 │        │
│  └────────────────────────────────────────────────────┘        │
│                              │                                  │
│  ┌────────────────────────────────────────────────────┐        │
│  │   Modified OSRS Private Server                     │        │
│  │   - Combat system                                  │        │
│  │   - Player movement                                │        │
│  │   - Item/inventory management                      │        │
│  │   - Prayer/spell system                            │        │
│  └────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │
┌─────────────────────────────────────────────────────────────────┐
│                  ENVIRONMENT CONTRACTS                          │
│                    (contracts/)                                 │
│  ┌────────────────────────────────────────────────────┐        │
│  │   NhEnv.json                                       │        │
│  │   - Defines 11 action categories                   │        │
│  │   - Defines 120+ observations                      │        │
│  │   - Specifies action dependencies                  │        │
│  └────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component 1: ML Training System (pvp-ml/)

**Purpose:** Implements the reinforcement learning algorithm that learns PvP strategies.

### Key Technologies:
- **Language:** Python 3.x
- **ML Framework:** PyTorch
- **RL Algorithm:** Proximal Policy Optimization (PPO)
- **Environment Interface:** OpenAI Gym-style API
- **Distributed Computing:** Ray (optional)
- **Monitoring:** TensorBoard

### Directory Structure:
```
pvp-ml/
├── pvp_ml/
│   ├── ppo/                    # PPO implementation
│   │   ├── ppo.py             # Main PPO class
│   │   ├── policy.py          # Neural network architecture
│   │   ├── buffer.py          # Experience replay buffer
│   │   ├── trainer.py         # Training loop orchestration
│   │   └── rollout_sampler.py # Data collection
│   ├── env/                    # Environment wrappers
│   │   ├── pvp_env.py         # Main PvP environment
│   │   ├── async_io_env.py    # Async environment base
│   │   └── remote_env_connector.py # Socket communication
│   ├── callback/               # Training callbacks
│   │   ├── checkpoint_callback.py
│   │   ├── eval_callback.py
│   │   ├── past_self_play_callback.py
│   │   └── ... (many more)
│   ├── util/                   # Utility functions
│   │   ├── contract_loader.py # Load JSON contracts
│   │   ├── schedule.py        # Hyperparameter scheduling
│   │   └── running_mean_std.py # Normalization
│   ├── train.py               # Main training script
│   ├── evaluate.py            # Model evaluation
│   └── api.py                 # Model serving API
├── config/                     # Training configurations
├── models/                     # Pre-trained models
│   ├── GeneralizedNh/
│   └── FineTunedNh/
└── test/                       # Integration & unit tests
```

### Core Responsibilities:
1. **Policy Learning:** Train neural network to choose optimal actions
2. **Environment Management:** Connect to and control game simulation
3. **Self-Play Orchestration:** Manage multiple AI opponents of varying skill
4. **Reward Shaping:** Calculate multi-component rewards from game state
5. **Checkpointing:** Save model versions for evaluation and rollback
6. **Monitoring:** Track metrics (win rate, loss, entropy, etc.)
7. **Distributed Training:** Scale across multiple CPUs/GPUs

---

## Component 2: Game Simulation (simulation-rsps/)

**Purpose:** Provides an accurate, fast simulation of OSRS PvP combat for training.

### Key Technologies:
- **Language:** Java 17
- **Base:** Elvarg RSPS (RuneScape Private Server)
- **Build System:** Gradle
- **Networking:** Custom socket server

### Directory Structure:
```
simulation-rsps/
└── ElvargServer/
    └── src/main/java/
        ├── com/github/naton1/rl/      # RL plugin code
        │   ├── ReinforcementLearningPlugin.java  # Main plugin
        │   ├── RemoteEnvironmentServer.java      # Socket server
        │   ├── RemoteEnvironmentPlayerBot.java   # Agent control
        │   ├── AgentBotLoader.java               # Load agents
        │   └── environment/                       # Environment implementations
        │       ├── nh/                           # NH environment
        │       └── dharok/                       # Dharok environment
        ├── com/elvarg/                # Original RSPS code
        │   ├── game/
        │   │   ├── entity/
        │   │   ├── content/combat/    # Combat mechanics
        │   │   ├── model/            # Game models
        │   │   └── plugin/           # Plugin system (added)
        │   └── net/                  # Networking
        └── gradle/
```

### Core Responsibilities:
1. **Game Simulation:** Execute OSRS combat mechanics accurately
   - Attack delays and cooldowns
   - Food/potion consumption timing
   - Prayer effects
   - Freeze/stun mechanics
   - Special attacks
   - Damage calculation with RNG
2. **State Exposure:** Convert game state to observations for ML system
3. **Action Execution:** Parse and execute AI actions
4. **Socket Server:** Handle concurrent connections from multiple training agents
5. **Debugging:** Log environment state for troubleshooting

### Key Modifications from Base RSPS:
- Added **plugin/event system** for cleaner code organization
- Implemented **RemoteEnvironmentServer** for API communication
- Created **AgentEnvironment** abstraction for different PvP modes
- Enhanced **combat accuracy** (critical for transfer to live game)
- Added **state serialization** for observations

---

## Component 3: Environment Contracts (contracts/)

**Purpose:** Define the interface between the ML system and game simulation.

### Structure:
```
contracts/
└── environments/
    ├── NhEnv.json       # No Honor environment
    └── DharokEnv.json   # Dharok's edge-style (partial)
```

### Contract Schema:

Each contract JSON file contains:

#### 1. **Actions Array**
Defines all possible actions the AI can take each game tick.

```json
{
  "actions": [
    {
      "id": "attack",
      "description": "Attack style",
      "actions": [
        { "id": "no_op_attack", "description": "No-op attack" },
        { "id": "mage_attack", "description": "Mage attack" },
        { "id": "ranged_attack", "description": "Ranged attack" },
        { "id": "melee_attack", "description": "Melee attack" }
      ]
    },
    ...
  ]
}
```

Each action can have **dependencies:**
```json
{
  "id": "melee_special_attack",
  "dependencies": {
    "require_all": ["melee_attack"]  // Only valid if melee_attack chosen
  }
}
```

#### 2. **Observations Array**
Defines all state information the AI can observe.

```json
{
  "observations": [
    {
      "id": "player_health_percent",
      "description": "Player's health percent",
      "partial": true  // Opponent can't see this
    },
    {
      "id": "player_using_melee",
      "description": "Player using melee",
      "partial": false  // Opponent can see this
    },
    {
      "id": "absolute_attack_level",
      "description": "Absolute attack level",
      "constant": true  // Never changes during episode
    }
  ]
}
```

**Observation Types:**
- `partial: true` - Private info (own HP, inventory)
- `partial: false` - Public info (visible gear, prayer)
- `constant: true` - Fixed for episode (base stats, loadout)

### Why Contracts?

1. **Decoupling:** ML and simulation can evolve independently
2. **Type Safety:** Ensures both sides agree on data format
3. **Documentation:** Self-documenting interface
4. **Validation:** Can validate inputs/outputs against schema
5. **Extensibility:** Easy to add new environments

---

## Data Flow: Step-by-Step

Let's trace what happens when the AI takes one action in training:

### 1. **Observation** (Python → Java)
```python
# pvp_env.py calls reset or step
obs, info = await self.reset_async()
```
↓
```python
# remote_env_connector.py sends TCP message
await connector.send(action="reset", body={...})
```
↓
```java
// RemoteEnvironmentServer.java receives message
handleRequest(connection, request)
```
↓
```java
// Creates/resets environment
AgentEnvironment env = createEnvironment()
ObservationResponse obs = env.reset()
```
↓
```java
// Serializes observations to JSON
return { "obs": [0.5, 1.0, ...], "actionMasks": [[true, false, ...]] }
```
↓
```python
# Python receives and processes
obs = np.array(response['obs'], dtype=np.float32)
```

### 2. **Decision** (Python)
```python
# Policy network chooses action
action, log_prob, entropy, value = ppo.predict(
    obs, action_masks, deterministic=False
)
# action = [2, 0, 1, 0, 1, 0, 0, 1, 0, 3, 2]  # 11 action heads
```

### 3. **Execution** (Python → Java)
```python
# Send action to simulation
response = await env.step_async(action)
```
↓
```java
// Execute action in game
StepResponse response = agentEnvironment.step(action)
```
↓
```java
// Game tick happens
player.performAction(action)
combat.process()
movement.process()
```

### 4. **Result** (Java → Python)
```java
// Return new state
return {
  "obs": [...],           // New observations
  "reward": 0.0,          // Calculated in Python
  "done": false,          // Episode ended?
  "meta": {               // Extra info for reward
    "damageDealt": 15,
    "damageReceived": 0,
    "playerHealth": 85,
    ...
  }
}
```
↓
```python
# Python calculates reward
reward = self._generate_reward(response, info)
# reward = 0.05  (from dealing 15 damage)
```

### 5. **Storage** (Python)
```python
# Store experience in buffer
buffer.add(obs, action, reward, next_obs, done)
```

### 6. **Learning** (Python)
```python
# After N steps collected, train
for batch in buffer.generate_batches(batch_size=256):
    loss = compute_ppo_loss(batch)
    loss.backward()
    optimizer.step()
```

---

## Communication Protocol

### Socket Protocol (TCP on port 7070)

#### Message Format:
```json
{
  "action": "login|logout|reset|step",
  "body": { ... }
}
```

#### 1. Login
**Request:**
```json
{
  "action": "login",
  "body": {
    "agentType": "NhEnv"
  }
}
```

**Response:**
```json
{
  "success": true
}
```

#### 2. Reset
**Request:**
```json
{
  "action": "reset",
  "body": {
    "target": "baseline",
    "training": true,
    "deathMatch": true,
    "resetParams": {
      "episodeId": "uuid-here",
      "agent": "env-id-0"
    }
  }
}
```

**Response:**
```json
{
  "obs": [0.5, 1.0, 0.0, ...],  // 120+ floats
  "actionMasks": [
    [true, true, true, false],   // Attack options
    [true, false, true],         // Melee type
    ...                          // 11 total
  ],
  "meta": {
    "episodeTicks": 0,
    "playerHealth": 99,
    ...
  }
}
```

#### 3. Step
**Request:**
```json
{
  "action": "step",
  "body": {
    "action": [2, 0, 1, 0, 1, 0, 0, 1, 0, 3, 2]  // 11 integers
  }
}
```

**Response:**
```json
{
  "obs": [...],
  "actionMasks": [...],
  "terminalState": "WON",  // if episode ended
  "meta": {
    "episodeTicks": 147,
    "damageDealt": 15,
    "damageReceived": 0,
    "playerHealth": 85,
    "targetHealth": 60,
    ...
  }
}
```

#### 4. Logout
**Request:**
```json
{
  "action": "logout",
  "body": {}
}
```

---

## Scalability & Performance

### Distributed Training Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Training Machine (GPU)                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  PPO Trainer                                           │  │
│  │  - Gradient computation                                │  │
│  │  - Neural network updates                              │  │
│  │  - Model checkpointing                                 │  │
│  └────────────────────────────────────────────────────────┘  │
│                           │                                   │
│                           │ Ray RPC                           │
│                           ▼                                   │
└──────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐ ┌────────▼───────┐ ┌────────▼───────┐
│  Rollout CPU 1 │ │  Rollout CPU 2 │ │  Rollout CPU N │
│  - 10 envs     │ │  - 10 envs     │ │  - 10 envs     │
│  - Inference   │ │  - Inference   │ │  - Inference   │
│  - Data collect│ │  - Data collect│ │  - Data collect│
└────────┬───────┘ └────────┬───────┘ └────────┬───────┘
         │                  │                  │
         ▼                  ▼                  ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ RSPS Server 1  │ │ RSPS Server 2  │ │ RSPS Server N  │
│ (localhost)    │ │ (localhost)    │ │ (localhost)    │
└────────────────┘ └────────────────┘ └────────────────┘
```

### Performance Characteristics

**Single Machine (CPU):**
- ~10-20 environments in parallel
- ~100-200 steps/second total
- ~3-5 hours per 100 rollouts

**Distributed (Ray cluster):**
- ~100+ environments in parallel
- ~1000+ steps/second total
- ~30-60 minutes per 100 rollouts

**Bottlenecks:**
1. **Simulation speed:** Java tick processing (600ms/tick in-game)
2. **Network I/O:** Socket communication overhead
3. **Python GIL:** Async I/O helps but still limiting

**Optimizations:**
1. **Async I/O:** Non-blocking socket communication
2. **TorchScript:** JIT compile policy for faster inference
3. **Batch inference:** Process multiple envs together
4. **CPU offloading:** Rollouts on CPU, training on GPU

---

## Key Design Decisions

### 1. **Why Custom Simulation vs. Live Game?**

**Advantages:**
- ✅ **Speed:** No network latency, instant ticks
- ✅ **Control:** Deterministic RNG seeds for reproducibility
- ✅ **Scale:** Run 100+ parallel fights
- ✅ **Safety:** No risk of account bans
- ✅ **Customization:** Add debugging, logging, state access

**Challenges:**
- ❌ **Accuracy:** Must match live game mechanics perfectly
- ❌ **Maintenance:** Must update when game updates
- ❌ **Transfer:** Learned behavior must work on live game

### 2. **Why PPO Algorithm?**

**Alternatives considered:**
- DQN: Doesn't handle continuous/high-dim action spaces well
- A3C: Less sample efficient than PPO
- SAC: Designed for continuous actions (PvP is discrete)

**PPO chosen because:**
- ✅ Stable training (clipped objectives prevent large updates)
- ✅ Sample efficient (reuses data via multiple epochs)
- ✅ Works with discrete actions
- ✅ Proven in complex games (Dota 2, StarCraft II)

### 3. **Why Self-Play?**

**Without self-play:** AI only learns to beat fixed scripted opponents

**With self-play:**
- ✅ Curriculum learning (difficulty increases naturally)
- ✅ Diverse strategies (past versions have different playstyles)
- ✅ Robust policies (can't exploit single opponent)
- ✅ Continuous improvement (always faces challenge)

### 4. **Why Action Dependencies?**

**Problem:**
```
Attack: Melee
Melee Type: None  ← Invalid!
```

**Solution:**
```json
{
  "id": "no_melee_attack",
  "dependencies": {
    "require_none": ["melee_attack"]
  }
}
```

**Benefits:**
- ✅ Prevents invalid actions
- ✅ Reduces action space (faster learning)
- ✅ Encodes domain knowledge
- ✅ Cleaner than post-hoc filtering

---

## Monitoring & Debugging

### TensorBoard Metrics

**Training Metrics:**
- `train/loss` - Total loss
- `train/policy_gradient_loss` - Policy loss
- `train/value_loss` - Value function loss
- `train/entropy_loss` - Entropy (exploration)
- `train/clip_fraction` - How often updates clipped
- `train/explained_variance` - Value function accuracy
- `train/kl` - KL divergence (policy change)

**Environment Metrics:**
- `env/num_envs` - Active environments
- `env/num_self_play` - Self-play environments
- `env/num_baseline` - Baseline environments

**Evaluation Metrics:**
- `eval/win_rate` - Win rate vs baseline
- `eval/avg_episode_length` - Fight duration
- `eval/avg_reward` - Average reward

**Observation Metrics:**
- `observations/{name}_rollout_mean` - Mean value in rollout
- `observations/{name}_running_mean` - Running mean for normalization

**Reward Metrics:**
- `rewards/damage_dealt` - Total damage reward
- `rewards/win` - Win bonus
- `rewards/protected_correct_prayer` - Prayer reward
- ... (20+ reward components)

### Logging

**Python Logging:**
```python
logger.debug(f"Stepping environment {env_id}: {action}")
logger.info(f"Training rollout {rollout_num} complete")
logger.warning(f"Desync detected: {desync_ticks} ticks")
logger.error(f"Environment error: {error}")
```

**Log Files:**
- `logs/train-{timestamp}.log` - Training logs
- `experiments/{name}/traceback-dump.txt` - Error traces
- `tensorboard/{name}/` - TensorBoard events

### Debugging Tools

**1. Server Debug Tracker:**
```python
# Monitors simulation server health
ServerDebugTracker.run(host="localhost", port=7070)
```

**2. Buffer Saving:**
```python
# Save rollout data for analysis
SaveBufferCallback(experiment_name=name)
```

**3. Traceback Tracking:**
```python
# Aggregate error traces
track_tracebacks(path="traceback-dump.txt")
```

---

## Directory Structure (Complete)

```
osrs-pvp-reinforcement-learning/
├── pvp-ml/                          # Python ML system
│   ├── pvp_ml/
│   │   ├── ppo/                     # PPO implementation
│   │   ├── env/                     # Environment wrappers
│   │   ├── callback/                # Training callbacks
│   │   ├── util/                    # Utilities
│   │   ├── scripted/                # Scripted baseline
│   │   ├── train.py                 # Training script
│   │   ├── evaluate.py              # Evaluation script
│   │   └── api.py                   # Model serving API
│   ├── config/                      # YAML configs
│   ├── models/                      # Pre-trained models
│   ├── test/                        # Tests
│   ├── environment.yml              # Conda environment
│   └── README.md
├── simulation-rsps/                 # Java simulation
│   └── ElvargServer/
│       ├── src/main/java/
│       │   ├── com/github/naton1/rl/  # RL plugin
│       │   └── com/elvarg/            # Base RSPS
│       ├── build.gradle
│       └── README.md
├── contracts/                       # Environment contracts
│   └── environments/
│       ├── NhEnv.json
│       └── DharokEnv.json
├── assets/                          # Images/videos
└── README.md                        # Main README
```

---

## Next Steps

For detailed analysis of each component, see:
- `02-environment-system.md` - PvpEnv, contracts, observations
- `03-ppo-algorithm.md` - PPO implementation details
- `04-neural-network-architecture.md` - Policy network design
- `05-training-pipeline.md` - Training workflow
- `06-reward-system.md` - Reward calculation
- `07-self-play-strategy.md` - Self-play implementation
- `08-buffer-data-collection.md` - Experience replay
- `09-simulation-rsps.md` - Java simulation details
- `10-code-structure.md` - Key classes and functions
