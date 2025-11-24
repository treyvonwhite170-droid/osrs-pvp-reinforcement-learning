# Environment System - Deep Dive

## Overview

The environment system is the bridge between the reinforcement learning algorithm and the game simulation. It implements the OpenAI Gym interface, providing a standardized way for the AI to interact with the OSRS PvP simulation.

---

## Core Components

### 1. PvpEnv Class (`pvp_ml/env/pvp_env.py`)

The main environment class that wraps the game simulation.

#### Class Hierarchy
```
AsyncIoEnv[NDArray[np.float32], NDArray[np.int32]]
    └── PvpEnv
```

#### Key Attributes

```python
class PvpEnv(AsyncIoEnv):
    # Spaces
    action_space: spaces.MultiDiscrete      # Shape: (11,) with varying sizes
    observation_space: spaces.Box           # Shape: (frames, features)
    partial_observation_space: spaces.Box   # Actor sees this

    # Metadata
    meta: EnvironmentMeta                   # Loaded from contract
    action_dependencies: ActionDependencies  # Action constraints

    # Connection
    _remote_env_connector: RemoteEnvConnector  # Socket connection
    _logged_in: bool
    _closed: bool

    # Episode context
    _episode_context: _EpisodeContext | None
```

#### Initialization Parameters

**Core Parameters:**
```python
def __init__(
    self,
    env_name: str = "NhEnv",           # Environment type
    env_id: str = "PvpEnv",            # Unique identifier
    target: str = "baseline",          # Opponent type
    training: bool = False,            # Training vs eval mode

    # Rewards (Schedule types allow annealing)
    default_reward: Schedule[float] = ConstantSchedule(0.0),
    win_reward: Schedule[float] = ConstantSchedule(1.0),
    lose_reward: Schedule[float] = ConstantSchedule(-1.0),
    tie_reward: Schedule[float] = ConstantSchedule(-0.2),

    # Damage rewards
    damage_dealt_reward_scale: Schedule[float] = ConstantSchedule(0.0),
    damage_received_reward_scale: Schedule[float] = ConstantSchedule(0.0),

    # Prayer rewards
    protected_correct_prayer_reward: Schedule[float] = ConstantSchedule(0.0),
    attacked_correct_prayer_reward: Schedule[float] = ConstantSchedule(0.0),

    # Frame stacking
    stack_frames: int | list[int] = 1,

    # Network
    remote_environment_host: str = "localhost",
    remote_environment_port: int = 7070,

    # Advanced options
    include_target_obs_in_critic: bool = False,  # Full observability for critic
    noise_generator: NoiseGenerator | None = None,  # Add noise to observations
    ...
)
```

---

## Observation Space

### Structure

The observation space is a **Box** (continuous values) with shape `(stack_frames, num_features)`.

For NhEnv with `stack_frames=1`:
- Shape: `(1, 120+)` for actor
- Shape: `(1, 180+)` for critic (if `include_target_obs_in_critic=True`)

### Frame Stacking

**Purpose:** Give the AI "memory" of recent history

**Example:**
```python
stack_frames = [0, 1, 2]  # Current frame, 1 tick ago, 2 ticks ago

# Observation shape: (3, 120)
obs = np.array([
    [0.85, 1.0, 0.0, ...],  # Current tick
    [0.88, 1.0, 0.0, ...],  # 1 tick ago
    [0.91, 1.0, 0.0, ...],  # 2 ticks ago
])
```

**Implementation:**
```python
def __process_obs(self, obs: NDArray[np.float32], ...) -> NDArray[np.float32]:
    if self._episode_context.steps == 0:
        # Initialize frame history
        history_size = self._stack_frames[-1] + 1
        self._episode_context.frame_history = deque(
            [np.zeros_like(obs) for _ in range(history_size)],
            maxlen=history_size
        )

    # Add new frame at the front
    self._episode_context.frame_history.appendleft(obs)

    # Stack selected frames
    obs = np.stack(self._episode_context.frame_history)[self._stack_frames]
    return obs
```

### Observation Categories (NhEnv)

#### 1. **Combat State** (16 observations)
```python
# Player attack state
player_using_melee: bool       # Currently in melee stance
player_using_ranged: bool      # Currently in ranged stance
player_using_mage: bool        # Currently in mage stance
player_spec_equipped: bool     # Special weapon equipped
special_energy_percent: float  # 0.0 to 1.0

# Target attack state
target_using_melee: bool
target_using_ranged: bool
target_using_mage: bool
target_spec_equipped: bool
target_special_percent: float
```

#### 2. **Prayer State** (10 observations)
```python
# Player prayers
player_melee_prayer: bool
player_ranged_prayer: bool
player_magic_prayer: bool
player_smite_prayer: bool
player_redemption_prayer: bool

# Target prayers
target_melee_prayer: bool
target_ranged_prayer: bool
target_magic_prayer: bool
target_smite_prayer: bool
target_redemption_prayer: bool
```

#### 3. **Health & Resources** (partial, 8 observations)
```python
player_health_percent: float   # 0.0 to 1.0 (partial)
target_health_percent: float   # 0.0 to 1.0 (partial)

# Inventory (partial)
range_potion_doses: int        # 0-4
combat_potion_doses: int       # 0-4
super_restore_doses: int       # 0-40+
brew_doses: int                # 0-40+
food_count: int                # 0-20+
karambwan_count: int           # 0-20+
prayer_points: int             # 0-99+
```

#### 4. **Freeze/Movement** (7 observations)
```python
player_frozen_ticks: int       # Ticks remaining frozen
target_frozen_ticks: int
player_frozen_immunity_ticks: int
target_frozen_immunity_ticks: int
player_location_can_melee: bool
player_is_moving: bool
target_is_moving: bool
```

#### 5. **Levels** (partial, 10 observations)
```python
# Boosted levels (change with potions/brews)
strength_level: int
attack_level: int
defense_level: int
ranged_level: int
magic_level: int

# Base levels (constant)
absolute_strength_level: int
absolute_attack_level: int
absolute_defense_level: int
absolute_ranged_level: int
absolute_magic_level: int
```

#### 6. **Cooldowns** (partial, 8 observations)
```python
attack_cycle_ticks: int        # Ticks until can attack
food_cycle_ticks: int          # Ticks until can eat
potion_cycle_ticks: int        # Ticks until can drink
karambwan_cycle_ticks: int     # Ticks until can eat karambwan
food_attack_delay: int         # Attack delay from eating

target_attack_cycle_ticks: int
target_potion_cycle_ticks: int
```

#### 7. **Pending Damage** (partial, 3 observations)
```python
pending_damage_on_target: int  # Damage in flight to target
ticks_until_hit_on_target: int
ticks_until_hit_on_player: int
```

#### 8. **Recent Actions** (4 observations)
```python
player_just_attacked: bool     # Attacked this tick
target_just_attacked: bool
damage_on_player_tick: bool    # Took damage this tick
damage_on_target_tick: bool
tick_new_attack_damage: int    # Damage dealt this tick
```

#### 9. **Positioning** (3 observations)
```python
destination_to_target_distance: int
player_to_destination_distance: int
player_to_target_distance: int
```

#### 10. **Prayer Tracking** (14 observations)
```python
player_prayer_correct: bool    # Praying correct overhead
target_prayer_correct: bool

# Hit percentages (for predicting attack style)
target_melee_hit_percent: float
target_magic_hit_percent: float
target_ranged_hit_percent: float
player_melee_hit_percent: float
player_magic_hit_percent: float
player_ranged_hit_percent: float

# Prayer percentages
target_magic_prayer_percent: float
target_ranged_prayer_percent: float
target_melee_prayer_percent: float
player_magic_prayer_percent: float
player_ranged_prayer_percent: float
player_melee_prayer_percent: float
```

#### 11. **Gear Stats** (constant, 30+ observations)
```python
# Loadout information
is_enchanted_dragon_bolt: bool
is_melee_spec_ags: bool
is_blood_fury: bool
...

# Offensive stats
magic_accuracy: float
magic_strength: float
ranged_accuracy: float
ranged_strength: float
melee_accuracy: float
melee_strength: float

# Defensive stats (for each combat style)
magic_gear_ranged_defence: float
magic_gear_mage_defence: float
magic_gear_melee_defence: float
ranged_gear_ranged_defence: float
...
```

#### 12. **Game Modes** (constant, 2 observations)
```python
is_lms_restrictions: bool
is_pvp_arena_rules: bool
```

#### 13. **Special Mechanics** (6 observations)
```python
is_veng_active: bool
is_target_veng_active: bool
player_veng_cooldown_ticks: int
target_veng_cooldown_ticks: int
is_player_lunar_spellbook: bool
is_target_lunar_spellbook: bool
```

### Partial vs Full Observability

**Partial Observations** (actor sees these):
- Own health, inventory, cooldowns
- Opponent's visible state (gear, prayer)

**Full Observations** (critic sees these if enabled):
- Everything the actor sees
- PLUS opponent's partial observations

**Example:**
```python
if include_target_obs_in_critic:
    # Critic gets both perspectives
    obs = np.concatenate([
        player_obs,  # 120 features
        target_obs[partially_observable_indices]  # +60 features
    ])  # Total: 180 features
```

**Why?**
- Actor must make decisions with limited info (realistic)
- Critic can use full info to better estimate value (training aid)

---

## Action Space

### Structure

The action space is **MultiDiscrete** with 11 action heads:

```python
action_space = spaces.MultiDiscrete([4, 3, 3, 4, 5, 2, 2, 2, 2, 5, 6])
```

### Action Heads

#### 1. **Attack Style** (4 options)
```
0: no_op_attack     # Don't attack
1: mage_attack      # Cast a spell
2: ranged_attack    # Shoot projectile
3: melee_attack     # Melee attack
```

#### 2. **Melee Attack Type** (3 options)
```
0: no_melee_attack      # Only valid if not melee
1: basic_melee_attack   # Normal attack
2: melee_special_attack # Special attack
```
**Dependency:** `melee_attack` must be chosen in head 1

#### 3. **Ranged Attack Type** (3 options)
```
0: no_ranged_attack
1: basic_ranged_attack
2: ranged_special_attack
```
**Dependency:** `ranged_attack` must be chosen in head 1

#### 4. **Mage Attack Type** (4 options)
```
0: no_mage_attack
1: use_ice_spell      # Ice Barrage (freeze)
2: use_blood_spell    # Blood Barrage (heal)
3: use_magic_spec     # Volatile Nightmare Staff spec
```
**Dependency:** `mage_attack` must be chosen in head 1

#### 5. **Potion** (5 options)
```
0: no_potion
1: use_brew            # Saradomin brew
2: use_restore_potion  # Super restore
3: use_combat_potion   # Super combat
4: use_ranged_potion   # Ranging potion
```

#### 6. **Food** (2 options)
```
0: dont_eat_food
1: eat_primary_food
```

#### 7. **Karambwan** (2 options)
```
0: dont_karambwan
1: eat_karambwan  # Combo eat with food
```

#### 8. **Vengeance** (2 options)
```
0: dont_use_veng
1: use_veng
```

#### 9. **Gear Swap** (2 options)
```
0: no_gear        # Stay in current gear
1: use_tank_gear  # Swap to defensive gear
```
**Dependency:** `no_op_attack` must be chosen (can't attack while swapping)

#### 10. **Movement** (5 options)
```
0: dont_move
1: move_next_to_target    # Melee distance
2: move_under_target      # Tile stack
3: move_to_farcast_tile   # Distance for ranged/mage
4: move_diagonal_to_target # Diagonal positioning
```

#### 11. **Farcast Distance** (6 options)
```
0: no_op_farcast   # Not farcasting
1: farcast_2_tiles
2: farcast_3_tiles
3: farcast_4_tiles
4: farcast_5_tiles
5: farcast_6_tiles
6: farcast_7_tiles
```
**Dependency:** `move_to_farcast_tile` must be chosen in head 10

### Action Dependencies

**Implementation:**
```python
action_dependencies = {
    # Action head index
    1: {  # Melee attack type
        # Specific action index
        0: {  # no_melee_attack
            'require_none': [(0, 3)]  # Require NOT melee_attack
        },
        1: {  # basic_melee_attack
            'require_all': [(0, 3)]   # Require melee_attack
        },
        2: {  # melee_special_attack
            'require_all': [(0, 3)]
        }
    },
    8: {  # Gear swap
        0: {  # no_gear
            'require_none': [(0, 0)]  # Require NOT no_op_attack
        },
        1: {  # use_tank_gear
            'require_all': [(0, 0)]   # Require no_op_attack
        }
    }
}
```

**Dependency Types:**
- `require_all`: ALL conditions must be true
- `require_any`: AT LEAST ONE condition must be true
- `require_none`: NO condition can be true

### Action Masking

In addition to dependencies, invalid actions are masked based on game state:

```python
# Examples of masked actions:
- Can't eat food if inventory empty
- Can't use special attack if energy < 50%
- Can't cast vengeance if on cooldown
- Can't use range attack if no ammo
```

**Masking Process:**
1. Environment provides raw masks (game state)
2. Policy applies dependency masks (action logic)
3. Combined mask ensures only valid actions
4. Policy samples from valid actions only

---

## Episode Lifecycle

### 1. **Reset** (Start New Episode)

```python
obs, info = await env.reset_async(
    seed=None,
    options={
        'trained_steps': 10000,
        'trained_rollouts': 50,
        'agent': 'main-agent'
    }
)
```

**What happens:**
1. Login to simulation server (if not logged in)
2. Send reset request with configuration
3. Simulation spawns both players with random gear
4. Return initial observation and action masks

**Response:**
```python
obs = np.array([...])  # Shape: (stack_frames, features)
info = {}
```

### 2. **Step** (Take Action)

```python
action = np.array([2, 0, 1, 0, 1, 0, 0, 1, 0, 3, 2])  # 11 integers

obs, reward, terminated, truncated, info = await env.step_async(action)
```

**What happens:**
1. Send action to simulation
2. Simulation executes game tick
3. Calculate reward from meta information
4. Return new state

**Response:**
```python
obs = np.array([...])        # New observation
reward = 0.05                # Calculated reward
terminated = False           # Episode ended naturally (win/loss)?
truncated = False            # Episode cut off (timeout/error)?
info = {
    'meta': {...},           # Game state details
    'rewards': {...},        # Reward breakdown
    'id': 'env-0',
    'target': 'baseline',
    'episode_id': 'uuid'
}
```

### 3. **Close** (Cleanup)

```python
await env.close_async()
```

**What happens:**
1. Logout from simulation server
2. Close socket connections
3. Clean up resources

---

## Reward Calculation

The reward is calculated in `__generate_reward()` method with **20+ components**.

### Reward Structure

```python
def __generate_reward(self, response, info) -> float:
    reward = 0.0

    # 1. Default step reward
    reward += self._default_reward.value(trained_rollouts)

    # 2. Custom reward function
    reward += self._custom_reward_fn.value(trained_rollouts, **meta)

    # 3. Damage rewards
    if reward_on_damage_generated:
        damage_dealt = meta['damageGeneratedOnTargetScale']
        damage_received = meta['damageGeneratedOnPlayerScale']
    else:
        damage_dealt = meta['damageDealt']
        damage_received = meta['damageReceived']

    reward += damage_dealt_reward_scale * damage_dealt
    reward += damage_received_reward_scale * damage_received  # Negative

    # 4. Smite rewards
    if meta.get('hitWithSmite'):
        reward += smite_multiplier * damage_dealt_reward_scale * damage_dealt

    # 5. Healing rewards
    if 'playerHealedScale' in meta:
        reward += -damage_received_reward_scale * meta['playerHealedScale']

    # 6. Safe penalty (eating too high HP)
    if 'eatAtFoodScale' in meta:
        reward -= safe_penalty.value(eat_at_food_scale=meta['eatAtFoodScale'])

    # 7. Prayer rewards
    if meta['protectedPrayer']:
        reward += protected_correct_prayer_reward * attack_speed_scale
    else:
        reward += protected_wrong_prayer_reward * attack_speed_scale

    if meta['hitOffPrayer']:
        reward += attacked_correct_prayer_reward * attack_speed_scale
    else:
        reward += attacked_wrong_prayer_reward * attack_speed_scale

    # 8. Freeze rewards
    if target_frozen_ticks_increased:
        reward += freeze_reward * ticks_increased

    # 9. Stat boost rewards
    reward += (strength_level_scale - 1) * strength_level_scale_reward

    # 10. Wasted food penalty
    if 'wastedFoodScale' in meta:
        reward += damage_received_reward_scale * meta['wastedFoodScale'] * multiplier

    # 11. Terminal rewards
    if terminated:
        if response['terminalState'] == 'WON':
            reward += win_reward.value(trained_rollouts)

            # Bonus for target having food left
            if reward_target_food_on_death:
                reward += damage_dealt_reward_scale * meta['targetRemainingFoodScale']

        elif response['terminalState'] == 'LOST':
            reward += lose_reward.value(trained_rollouts)

            # Penalty for dying with food left
            if penalize_food_on_death:
                reward += damage_received_reward_scale * meta['remainingFoodScale']

        elif response['terminalState'] == 'TIED':
            reward += tie_reward.value(trained_rollouts)

    return reward
```

### Reward Examples

**Good Actions:**
```python
# Hit for 30 damage with AGS spec
reward = 0.01 * 30 = +0.30

# Win the fight
reward = +1.0

# Hit through wrong prayer
reward = +0.05

# Freeze opponent
reward = +0.02 * 10_ticks = +0.20
```

**Bad Actions:**
```python
# Take 25 damage
reward = -0.01 * 25 = -0.25

# Lose the fight
reward = -1.0

# Eat at 90 HP (wasted 5 HP)
reward = -0.001 * 5 = -0.005

# Die with 10 food left
reward = -0.01 * 10 * 20_hp_per_food = -2.0
```

**Typical Episode Reward:**
```
Win: +1.0 (terminal) + 0.5 (damage) + 0.2 (prayers) = +1.7
Loss: -1.0 (terminal) - 0.3 (damage) - 0.1 (mistakes) = -1.4
```

---

## Advanced Features

### 1. **Observation Normalization**

```python
if normalize_observations:
    # Track running mean/std
    meta.running_observation_stats.update(obs)

    # Normalize
    normalized_obs = (obs - mean) / sqrt(var + 1e-8)
    normalized_obs = clip(normalized_obs, -10, 10)
```

**Why?**
- Neural networks train better with normalized inputs
- Different observations have different scales (HP: 0-99, ticks: 0-1000)
- Running statistics adapt over training

### 2. **Noise Injection**

```python
if noise_generator:
    noise_generator.add_noise(obs, trained_rollouts)
```

**Example:**
```python
# Add Gaussian noise to observations
obs += np.random.normal(0, 0.01, size=obs.shape)
```

**Why?**
- Regularization (prevents overfitting)
- Robustness (handles imperfect observations)
- Exploration (encourages diverse behaviors)

### 3. **Critic Full Observability**

```python
if include_target_obs_in_critic and training:
    # Critic sees opponent's private info
    full_obs = np.concatenate([
        actor_obs,
        target_obs[partial_indices]
    ])
```

**Why?**
- Actor learns with limited info (like human)
- Critic has full info for better value estimates
- Improves training stability

### 4. **Desync Detection**

```python
# Check if simulation fell behind
desynced_ticks = meta['episodeTicks'] - self._episode_context.steps

if desync_change > desync_tick_threshold:
    raise _TruncateException("Desynced by {desynced_ticks} ticks")
```

**Causes:**
- Network lag
- Simulation crash
- Processing too slow

**Handling:**
- Truncate episode (don't count as loss)
- Log for debugging
- Retry on next reset

---

## Environment Contracts

### Contract Loader

```python
# pvp_ml/util/contract_loader.py

def load_environment_contract(env_name: str) -> EnvironmentMeta:
    contract_path = f"contracts/environments/{env_name}.json"
    with open(contract_path) as f:
        contract = json.load(f)

    return EnvironmentMeta(
        actions=parse_actions(contract['actions']),
        observations=parse_observations(contract['observations'])
    )
```

### EnvironmentMeta

```python
@dataclass
class EnvironmentMeta:
    actions: list[ActionCategory]
    observations: list[Observation]

    def get_action_space(self) -> spaces.MultiDiscrete:
        sizes = [len(cat.actions) for cat in self.actions]
        return spaces.MultiDiscrete(sizes)

    def get_observation_space(self) -> spaces.Box:
        num_obs = len(self.observations)
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_obs,),
            dtype=np.float32
        )

    def get_partially_observable_indices(self) -> list[int]:
        return [
            i for i, obs in enumerate(self.observations)
            if obs.partial
        ]

    def get_action_dependency_config(self) -> ActionDependencies:
        dependencies = {}
        for head_idx, action_cat in enumerate(self.actions):
            for action_idx, action in enumerate(action_cat.actions):
                if action.dependencies:
                    if head_idx not in dependencies:
                        dependencies[head_idx] = {}
                    dependencies[head_idx][action_idx] = action.dependencies
        return dependencies
```

---

## Performance Optimizations

### 1. **Async I/O**

```python
# All network operations are async
class AsyncIoEnv:
    async def step_async(self, action):
        ...

    async def reset_async(self):
        ...

    async def close_async(self):
        ...
```

**Benefits:**
- Non-blocking socket operations
- Process multiple environments concurrently
- Better CPU utilization

### 2. **Connection Pooling**

```python
# Reuse connections
class RemoteEnvConnector:
    def __init__(self):
        self._reader = None
        self._writer = None

    async def send(self, action, body):
        if not self._writer:
            await self._connect()
        # Reuse existing connection
        ...
```

### 3. **Batch Processing**

```python
# Process multiple environments together
class AsyncIoVecEnv:
    async def step_async(self, actions):
        # Send all actions concurrently
        tasks = [
            env.step_async(action)
            for env, action in zip(self.envs, actions)
        ]
        return await asyncio.gather(*tasks)
```

---

## Common Patterns

### Pattern 1: Episode Loop

```python
env = PvpEnv(env_name="NhEnv", target="baseline")

obs, info = await env.reset_async()
done = False
total_reward = 0

while not done:
    # Get action from policy
    action = policy.predict(obs)

    # Step environment
    obs, reward, terminated, truncated, info = await env.step_async(action)
    total_reward += reward
    done = terminated or truncated

print(f"Episode reward: {total_reward}")
```

### Pattern 2: Vectorized Environments

```python
envs = AsyncIoVecEnv([
    lambda: PvpEnv(env_id=f"env-{i}", target="baseline")
    for i in range(10)
])

obs = await envs.reset_async()

for _ in range(256):  # Collect 256 steps
    actions = policy.predict(obs)
    obs, rewards, dones, infos = await envs.step_async(actions)
    buffer.add(obs, actions, rewards, dones)
```

### Pattern 3: Self-Play

```python
# Environment 0 vs Environment 1
envs = AsyncIoVecEnv([
    lambda: PvpEnv(env_id="0", target="1"),  # Plays against env 1
    lambda: PvpEnv(env_id="1", target="0"),  # Plays against env 0
])

# Both controlled by same policy
obs = await envs.reset_async()
actions = policy.predict(obs)  # Same policy controls both
obs, rewards, dones, infos = await envs.step_async(actions)
```

---

## Summary

The environment system:
1. **Abstracts** game complexity into standard RL interface
2. **Provides** 120+ observations of game state
3. **Supports** 11 action heads with dependencies
4. **Calculates** multi-component rewards
5. **Handles** async communication with simulation
6. **Implements** frame stacking, normalization, noise
7. **Enables** partial observability and full observability modes

This design allows the PPO algorithm to focus on learning without worrying about game-specific details.
