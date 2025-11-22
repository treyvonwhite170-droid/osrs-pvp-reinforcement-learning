# PPO (Proximal Policy Optimization) - Deep Dive

## Overview

PPO is the core reinforcement learning algorithm that trains the OSRS PvP AI. It's an **on-policy** algorithm that learns a policy (what actions to take) and a value function (how good is this state) simultaneously.

**Key Characteristics:**
- **On-policy:** Learns from data collected by current policy
- **Actor-Critic:** Learns both policy and value function
- **Clipped updates:** Prevents destructively large policy changes
- **Sample efficient:** Reuses data via multiple epochs

---

## Mathematical Foundation

### 1. **Policy Gradient Objective**

The goal is to maximize expected cumulative reward:

```
J(θ) = E[∑_{t=0}^T γ^t r_t]
```

Where:
- `θ` = policy parameters (neural network weights)
- `γ` = discount factor (0.99)
- `r_t` = reward at time t

### 2. **Advantage Function**

The advantage tells us how much better an action is than average:

```
A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
```

Where:
- `Q(s, a)` = expected return from taking action a in state s
- `V(s)` = expected return from state s (average over all actions)

**Interpretation:**
- `A > 0` → Action better than average
- `A < 0` → Action worse than average
- `A = 0` → Action exactly average

### 3. **PPO Clipped Objective**

```
L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
```

Where:
- `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` (probability ratio)
- `ε` = clip coefficient (0.2)
- `A_t` = advantage

**Why clipping?**
- Prevents policy from changing too much
- Stabilizes training
- Avoids catastrophic forgetting

### 4. **Value Function Loss**

```
L^VF(θ) = E[(V_θ(s_t) - V_target)^2]
```

Where:
- `V_target = A_t + V_θ_old(s_t)` (returns)

### 5. **Entropy Bonus**

```
L^ENT(θ) = E[-∑_a π_θ(a|s) log π_θ(a|s)]
```

**Purpose:** Encourage exploration by keeping policy spread out

### 6. **Total Loss**

```
L(θ) = L^CLIP(θ) + c_1 * L^VF(θ) + c_2 * L^ENT(θ)
```

Where:
- `c_1` = value loss coefficient (0.5)
- `c_2` = entropy coefficient (0.0 - 0.01, annealed)

---

## Implementation Details

### PPO Class (`pvp_ml/ppo/ppo.py`)

```python
class PPO:
    def __init__(
        self,
        policy_params: PolicyParams,
        meta: Meta,
        device: str = "cpu",
        trainable: bool = True,
    ):
        # Neural network
        self._policy = Policy(**policy_params)
        self._policy.to(device)

        # Optimizer
        self._optimizer = optim.Adam(
            self._policy.parameters(),
            lr=3e-4,
            eps=1e-5  # Numerical stability
        )

        # Normalization stats
        self.meta = meta  # Contains running_observation_stats
```

### Training Loop (`learn` method)

```python
def learn(
    self,
    buffer: Buffer,
    num_updates: int = 5,
    batch_size: int = 64,
    clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    entropy_coef: float = 0.0,
    max_grad_norm: float = 0.5,
):
    # Set learning rate
    for param_group in self._optimizer.param_groups:
        param_group['lr'] = learning_rate

    # Multiple epochs over same data
    for epoch in range(num_updates):
        # Shuffle data each epoch
        for batch in buffer.generate_batches(batch_size):
            # 1. Forward pass
            observations = batch.observations
            if self.meta.normalized_observations:
                observations = normalize(observations)

            _, new_log_probs, entropies, new_values, _ = self._policy(
                observations,
                batch.action_masks,
                input_actions=batch.actions
            )

            # 2. Compute losses
            old_log_probs = batch.old_log_prob
            advantages = batch.advantages

            # Normalize advantages (important!)
            if normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy loss (clipped)
            ratio = th.exp(new_log_probs - old_log_probs)
            surrogate1 = ratio * advantages
            surrogate2 = th.clamp(ratio, 1-clip_coef, 1+clip_coef) * advantages
            policy_loss = -th.mean(th.min(surrogate1, surrogate2))

            # Value loss (MSE)
            value_loss = th.nn.functional.mse_loss(
                new_values.squeeze(),
                batch.returns
            )

            # Entropy loss (negative because we want to maximize)
            entropy_loss = -th.mean(entropies.sum(dim=1))

            # Total loss
            loss = policy_loss + vf_coef * value_loss + entropy_coef * entropy_loss

            # 3. Backward pass
            loss.backward()

            # 4. Gradient clipping
            grad_norm = th.nn.utils.clip_grad_norm_(
                self._policy.parameters(),
                max_grad_norm
            )

            # 5. Optimizer step
            self._optimizer.step()
            self._optimizer.zero_grad()

    # Update normalization stats
    self.meta.running_observation_stats.update(buffer.observations)
```

---

## Generalized Advantage Estimation (GAE)

GAE computes advantages using a balance between bias and variance.

### Formula

```
A_t = ∑_{l=0}^∞ (γλ)^l δ_{t+l}
```

Where:
- `δ_t = r_t + γV(s_{t+1}) - V(s_t)` (TD error)
- `λ` = GAE lambda (0.95)
- `γ` = discount factor (0.99)

### Implementation (`buffer.py`)

```python
def _compute_returns_and_advantage(self, ppo: PPO):
    # Get value of last state
    last_values = ppo.predict(self.last_step_obs, ...)

    last_gae_lam = 0
    for step in reversed(range(self.buffer_size)):
        if step == self.buffer_size - 1:
            next_non_terminal = 1.0 - self.last_step_dones
            next_values = last_values
        else:
            next_non_terminal = 1.0 - self.episode_starts[step + 1]
            next_values = self.values[step + 1]

        # TD error
        delta = (
            self.rewards[step]
            + self.gamma * next_values * next_non_terminal
            - self.values[step]
        )

        # GAE
        last_gae_lam = (
            delta
            + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        )

        self.advantages[step] = last_gae_lam

    # Returns = advantages + values
    self.returns = self.advantages + self.values
```

### Why GAE?

**Pure Monte Carlo** (`λ = 1`):
- Low bias (unbiased estimate)
- High variance (noisy)

**Pure TD** (`λ = 0`):
- High bias (relies on value function)
- Low variance (smooth)

**GAE** (`λ = 0.95`):
- Balanced bias-variance tradeoff
- Generally works best in practice

---

## Key PPO Features

### 1. **Clipping Mechanism**

```python
ratio = th.exp(new_log_probs - old_log_probs)

# Unclipped objective
surrogate1 = ratio * advantages

# Clipped objective
surrogate2 = th.clamp(ratio, 1-clip_coef, 1+clip_coef) * advantages

# Take minimum (pessimistic bound)
policy_loss = -th.mean(th.min(surrogate1, surrogate2))
```

**Visualization:**

```
Advantage > 0 (good action):
  ratio < 1-ε: Use ratio (increase probability)
  1-ε ≤ ratio ≤ 1+ε: Use ratio
  ratio > 1+ε: Use 1+ε (clip, don't increase too much)

Advantage < 0 (bad action):
  ratio < 1-ε: Use 1-ε (clip, don't decrease too much)
  1-ε ≤ ratio ≤ 1+ε: Use ratio
  ratio > 1+ε: Use ratio (decrease probability)
```

**Effect:**
- Limits how much policy can change in one update
- Prevents catastrophic updates
- Stabilizes training

### 2. **Multiple Epochs**

Unlike vanilla policy gradient, PPO reuses data:

```python
# Collect data with old policy
buffer = collect_rollouts(old_policy)

# Update policy multiple times on same data
for epoch in range(num_updates):  # e.g., 5 epochs
    for batch in buffer.generate_batches():
        update_policy(batch)
```

**Constraint:**
- Clipping ensures policy doesn't diverge from data distribution
- Allows sample reuse (more efficient)

### 3. **Advantage Normalization**

```python
if normalize_advantages and len(advantages) > 1:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**Why?**
- Scales advantages to roughly [-1, 1]
- Stabilizes learning across different reward scales
- Makes hyperparameters more transferable

### 4. **Value Function Clipping** (not used in this project)

Some PPO implementations also clip value function updates:

```python
# Clip value predictions
clipped_values = old_values + th.clamp(
    new_values - old_values,
    -clip_coef,
    clip_coef
)

# Clipped value loss
value_loss = th.max(
    (new_values - returns)**2,
    (clipped_values - returns)**2
).mean()
```

This project uses simple MSE loss instead (simpler, works well).

---

## Hyperparameters

### Learning Hyperparameters

```python
# Optimizer
learning_rate: 3e-4        # Adam learning rate
eps: 1e-5                  # Adam epsilon (numerical stability)

# PPO
clip_coef: 0.2            # Clip coefficient (ε)
value_coef: 0.5           # Value loss weight (c_1)
entropy_coef: 0.0-0.01    # Entropy weight (c_2, annealed)
max_grad_norm: 0.5        # Gradient clipping
num_updates: 1-5          # Epochs per rollout
batch_size: 256           # Minibatch size
normalize_advantages: True # Normalize advantages

# GAE
gamma: 0.99               # Discount factor
gae_lambda: 0.95          # GAE lambda

# Data collection
num_rollout_steps: 256    # Steps per environment
num_envs: 10              # Parallel environments
```

### Hyperparameter Scheduling

Many hyperparameters can be scheduled (annealed):

```python
from pvp_ml.util.schedule import LinearSchedule, ConstantSchedule

# Example: Decay entropy coefficient
entropy_coef = LinearSchedule(
    start_value=0.01,
    end_value=0.001,
    end_timestep=500  # 500 rollouts
)

# Example: Constant learning rate
learning_rate = ConstantSchedule(3e-4)

# Get current value
current_entropy = entropy_coef.value(current_rollout)
```

**Common schedules:**
- Entropy: High early (explore) → Low later (exploit)
- Learning rate: Sometimes decayed over time
- Clip coefficient: Usually constant

---

## Training Metrics

### Logged Metrics

**Loss Metrics:**
```python
train/loss                    # Total loss
train/policy_gradient_loss    # Policy loss
train/value_loss              # Value function loss
train/entropy_loss            # Entropy loss
```

**Policy Metrics:**
```python
train/clip_fraction          # How often updates are clipped
train/kl                     # KL divergence (policy change)
train/explained_variance     # How well value function predicts returns
```

**Gradient Metrics:**
```python
train/grad_norm             # Gradient norm before clipping
train/learning_rate         # Current learning rate
```

### Interpreting Metrics

**Clip Fraction:**
- 0.0: Policy not changing much (might be stuck)
- 0.2-0.3: Healthy (some updates clipped)
- >0.5: Policy changing a lot (might be unstable)

**KL Divergence:**
- <0.01: Small policy change (safe)
- 0.01-0.05: Moderate change
- >0.1: Large change (potential instability)

**Explained Variance:**
- 1.0: Value function perfectly predicts returns
- 0.5-0.9: Good prediction
- <0.5: Poor value function

---

## Gradient Accumulation

For large batches that don't fit in memory:

```python
accumulated_gradients = 0
for batch in buffer.generate_batches(batch_size):
    loss = compute_loss(batch)
    loss = loss / grad_accum  # Scale loss
    loss.backward()           # Accumulate gradients

    accumulated_gradients += 1
    if accumulated_gradients == grad_accum:
        # Update after accumulating grad_accum batches
        optimizer.step()
        optimizer.zero_grad()
        accumulated_gradients = 0
```

**Effective batch size** = `batch_size * grad_accum`

**Example:**
- `batch_size=128, grad_accum=4` → Effective batch size of 512
- Uses less memory than `batch_size=512` directly
- Equivalent gradient updates

---

## Distributed Training

PPO can scale across multiple machines:

### Architecture

```
Central Learner (GPU):
  - Stores master policy
  - Performs gradient updates
  - Broadcasts updated policy

Rollout Workers (CPUs):
  - Receive current policy
  - Collect experience in parallel
  - Send data back to learner
```

### Implementation

```python
# Distributed rollout sampler
sampler = DistributedRolloutSampler(
    experiment_name="my-experiment",
    num_tasks=10,           # 10 parallel workers
    cpus_per_rollout=4      # 4 CPUs each
)

# Collect data (distributed)
buffer = sampler.sample_rollouts(
    ppo=ppo,
    n_steps=256,
    n_envs=10
)

# Train on aggregated data (centralized)
ppo.learn(buffer)
```

**Speedup:**
- 10x faster data collection with 10 workers
- Training time unchanged (single GPU)
- Overall: ~5-7x faster (communication overhead)

---

## PPO vs Other Algorithms

### Comparison

| Algorithm | On/Off Policy | Sample Efficiency | Stability | Complexity |
|-----------|---------------|-------------------|-----------|------------|
| DQN       | Off-policy    | High              | Moderate  | Low        |
| A3C       | On-policy     | Low               | Low       | Low        |
| PPO       | On-policy     | Moderate          | High      | Moderate   |
| SAC       | Off-policy    | High              | High      | High       |
| TD3       | Off-policy    | High              | High      | High       |

### Why PPO for this project?

✅ **Stable:** Clipping prevents destructive updates
✅ **Discrete actions:** Works well with categorical actions
✅ **Proven:** Success in Dota 2, StarCraft II
✅ **Simple:** Easier to debug than off-policy methods
✅ **Self-play compatible:** On-policy works well with self-play

❌ **Sample efficiency:** Not as efficient as off-policy methods
❌ **Exploration:** Can get stuck in local optima

---

## Common Issues & Solutions

### Issue 1: Policy Not Learning

**Symptoms:**
- Win rate stuck at ~50% (random)
- Value loss not decreasing
- Explained variance near 0

**Solutions:**
- Check reward scale (too small/large?)
- Increase learning rate
- Reduce clip coefficient (allow larger updates)
- Check observation normalization
- Verify environment is working correctly

### Issue 2: Training Instability

**Symptoms:**
- Win rate oscillating wildly
- Loss exploding
- KL divergence > 0.1

**Solutions:**
- Decrease learning rate
- Increase clip coefficient (restrict updates)
- Reduce batch size
- Add gradient clipping
- Check for NaN values in rewards/observations

### Issue 3: Slow Learning

**Symptoms:**
- Takes many rollouts to improve
- Gradual but steady progress

**Solutions:**
- Increase number of environments
- Tune reward shaping
- Use reward normalization
- Increase entropy coefficient (more exploration)
- Use curriculum learning (easier opponents first)

### Issue 4: Overfitting to Opponent

**Symptoms:**
- High win rate vs one opponent
- Poor generalization to new opponents

**Solutions:**
- Use self-play
- Train against diverse opponents
- Use past self-play
- Add observation noise
- Regularize (entropy, dropout)

---

## Advanced PPO Techniques (Used in This Project)

### 1. **Reward Normalization**

```python
# Normalize rewards by running std of cumulative rewards
cumulative_rewards = compute_cumulative_rewards(rewards, gamma)
reward_normalizer.update(cumulative_rewards)
normalized_rewards = rewards / sqrt(reward_normalizer.var)
```

**Effect:**
- Stabilizes learning across different reward scales
- Helps with sparse rewards
- Improves value function learning

### 2. **Observation Normalization**

```python
# Track running mean/std of observations
obs_normalizer.update(observations)
normalized_obs = (obs - obs_normalizer.mean) / sqrt(obs_normalizer.var)
normalized_obs = clip(normalized_obs, -10, 10)
```

**Effect:**
- Prevents feature scale issues
- Faster learning
- Better gradient flow

### 3. **Bootstrapping Truncated Episodes**

```python
# If episode truncated (not truly terminal), bootstrap
if truncated:
    terminal_obs = info['terminal_observation']
    terminal_value = ppo.predict(terminal_obs, return_values=True)
    reward += gamma * terminal_value
```

**Why?**
- Truncation != episode actually ended
- Don't want to treat it as terminal state
- Bootstrap from where it left off

### 4. **Action Normalization**

```python
# Normalize one-hot encoded actions (for autoregressive)
action_normalizer.update(actions)
normalized_actions = (actions - action_normalizer.mean) / sqrt(action_normalizer.var)
```

**Effect:**
- Stabilizes autoregressive action heads
- Better gradient flow between action heads

---

## Summary

PPO is the core algorithm that:
1. **Collects** experience from environments
2. **Computes** advantages using GAE
3. **Updates** policy to maximize advantages (clipped)
4. **Updates** value function to predict returns
5. **Regularizes** with entropy bonus
6. **Repeats** for multiple epochs

Key innovations:
- **Clipping** for stability
- **Multiple epochs** for sample efficiency
- **Normalization** for faster learning
- **GAE** for bias-variance balance

The combination of PPO with self-play, reward shaping, and careful hyperparameter tuning enabled the AI to reach superhuman performance in OSRS PvP.
