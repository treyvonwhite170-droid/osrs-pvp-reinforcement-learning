# Neural Network Architecture - Deep Dive

## Overview

The neural network is the "brain" of the AI agent. It maps observations (game state) to actions (what to do) and estimates the value (how good is this position).

**Architecture Type:** Actor-Critic
- **Actor:** Chooses actions (policy)
- **Critic:** Evaluates states (value function)

---

## High-Level Architecture

```
Input: Stacked Observations (frames × features)
    ↓
┌───────────────────────────────────────────────────┐
│         Feature Extractor (Optional)              │
│  - Flatten frame-stacked input                    │
│  - Optional shared/separate layers                │
└───────────────────────────────────────────────────┘
    ↓                                  ↓
┌─────────────────────┐    ┌─────────────────────┐
│      ACTOR          │    │      CRITIC         │
│  (Policy Network)   │    │  (Value Network)    │
├─────────────────────┤    ├─────────────────────┤
│ Hidden Layers       │    │ Hidden Layers       │
│ [128, 128, 128]     │    │ [64, 64]            │
│         ↓           │    │         ↓           │
│ 11 Action Heads     │    │ Value Head          │
│ (Autoregressive)    │    │ (Single neuron)     │
└─────────────────────┘    └─────────────────────┘
    ↓                                  ↓
Actions [11 integers]         Value [1 float]
```

---

## Policy Network (`pvp_ml/ppo/policy.py`)

### Policy Class

```python
class Policy(nn.Module):
    def __init__(
        self,
        max_sequence_length: int,      # Frame stack size
        actor_input_size: int,          # Features per frame (actor)
        critic_input_size: int,         # Features per frame (critic)
        action_head_sizes: list[int],   # [4, 3, 3, 4, 5, 2, 2, 2, 2, 5, 6]
        feature_extractor_config: MlpConfig = MlpConfig(),
        share_feature_extractor: bool = False,
        actor_config: MlpConfig = [128, 128, 128],
        critic_config: MlpConfig = [64, 64],
        action_head_configs: MlpConfig | list[MlpConfig] = None,
        action_dependencies: ActionDependencies = {},
        autoregressive_actions: bool = True,
        normalize_autoregressive_actions: bool = True,
    ):
        super(Policy, self).__init__()

        # Feature extractors
        if share_feature_extractor:
            # Single shared feature extractor
            self.feature_extractor = create_mlp(...)
        else:
            # Separate feature extractors
            self.actor_feature_extractor = create_mlp(...)
            self.critic_feature_extractor = create_mlp(...)

        # Actor and Critic
        self.actor = Actor(...)
        self.critic = Critic(...)

    def forward(
        self,
        x: th.Tensor,              # Shape: (batch, frames, features)
        action_masks: th.Tensor,   # Shape: (batch, sum(action_sizes))
        sample_deterministic: Optional[th.Tensor] = None,
        input_actions: Optional[th.Tensor] = None,
    ):
        # Flatten frame-stacked input
        x = x.reshape(x.size(0), -1)  # (batch, frames * features)

        # Feature extraction
        if self.share_feature_extractor:
            actor_features = critic_features = self.feature_extractor(x)
        else:
            actor_features = self.actor_feature_extractor(x[..., :actor_input_size])
            critic_features = self.critic_feature_extractor(x[..., :critic_input_size])

        # Actor forward
        actions, log_probs, entropy, probs = self.actor(
            actor_features,
            action_masks,
            sample_deterministic,
            input_actions
        )

        # Critic forward
        values = self.critic(critic_features)

        return actions, log_probs, entropy, values, probs
```

---

## Actor Network

### Architecture

```
Input: Features (hidden_size from feature extractor or raw features)
    ↓
┌────────────────────────────────────────────────┐
│          Actor Hidden Layers                   │
│  - Default: [128, 128, 128] with ReLU          │
│  - Orthogonal initialization (gain=√2)         │
└────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────┐
│         11 Action Heads (Autoregressive)       │
│                                                 │
│  Head 0: Attack Style [4 options]              │
│     Input: hidden                               │
│     ↓                                          │
│  Head 1: Melee Type [3 options]                │
│     Input: hidden + one_hot(head_0)            │
│     ↓                                          │
│  Head 2: Ranged Type [3 options]               │
│     Input: hidden + one_hot(head_0, head_1)    │
│     ↓                                          │
│  ...                                           │
│     ↓                                          │
│  Head 10: Prayer [6 options]                   │
│     Input: hidden + one_hot(head_0...head_9)   │
└────────────────────────────────────────────────┘
    ↓
Actions: [a_0, a_1, ..., a_10]  # 11 integers
Log Probs: sum(log π(a_i | s, a_<i))
Entropy: [H_0, H_1, ..., H_10]  # Per-head entropy
```

### Actor Class

```python
class Actor(nn.Module):
    def __init__(
        self,
        input_size: int,
        action_head_sizes: list[int],
        config: MlpConfig,
        action_dependencies: ActionDependencies,
        autoregressive_actions: bool = True,
        normalize_autoregressive_actions: bool = True,
    ):
        super(Actor, self).__init__()

        # Shared hidden layers
        self.hidden = create_mlp(config, input_size)

        # Action normalization stats (for autoregressive)
        self.register_buffer("action_mean", th.zeros(sum(action_head_sizes)))
        self.register_buffer("action_var", th.ones(sum(action_head_sizes)))
        self.register_buffer("action_count", th.tensor([1e-4]))

        # Create action heads
        self.heads = nn.ModuleList([
            self._create_head(i, head_size)
            for i, head_size in enumerate(action_head_sizes)
        ])

        self.action_dependencies = action_dependencies

    def _create_head(self, index: int, head_size: int) -> nn.Module:
        # Input includes previous actions if autoregressive
        autoregressive_size = sum(self.action_head_sizes[:index]) if self.autoregressive_actions else 0
        head_input_size = hidden_size + autoregressive_size

        # Head MLP + linear output
        mlp = create_mlp(head_config, head_input_size)
        head = nn.Linear(mlp_output_size, head_size)

        return nn.Sequential(mlp, head)

    def forward(
        self,
        x: th.Tensor,
        flattened_action_masks: th.Tensor,
        sample_deterministic: Optional[th.Tensor] = None,
        input_actions: Optional[th.Tensor] = None,
    ):
        # Process hidden layers
        actor_hidden = th.relu(self.hidden(x))

        actions = []
        log_probs = []
        one_hot_actions = []
        entropy = []

        # Process each action head autoregressively
        for i, head in enumerate(self.heads):
            # Build input for this head
            current_input = actor_hidden

            if self.autoregressive_actions and i > 0:
                # Concatenate previous actions (one-hot encoded)
                prev_actions = th.cat(one_hot_actions, dim=-1)
                if self.normalize_autoregressive_actions:
                    prev_actions = self._normalize(prev_actions)
                current_input = th.cat([current_input, prev_actions], dim=-1)

            # Get action mask for this head
            action_mask = action_masks[i]

            # Apply dependency mask
            dependency_mask = self._get_action_dependency_mask(actions, i, ...)
            mask = action_mask & dependency_mask

            # Ensure at least one action is valid (use no-op)
            no_action_mask = ~mask.any(dim=-1)
            mask[no_action_mask, 0] = True

            # Compute logits
            logits = head(current_input)

            # Mask invalid actions
            masked_logits = logits - ((~mask) * 1e8)
            probs = th.softmax(masked_logits, dim=-1)

            # Sample or use given action
            if input_actions is None:
                if sample_deterministic[i]:
                    action = probs.argmax(dim=-1)
                else:
                    action = th.multinomial(probs, 1).squeeze(-1)
            else:
                action = input_actions[:, i].long()

            # Store
            actions.append(action)
            one_hot_actions.append(
                th.nn.functional.one_hot(action, self.action_head_sizes[i])
            )
            log_probs.append(self._log_prob(probs, action))
            entropy.append(self._entropy(probs))

        # Combine
        combined_actions = th.stack(actions, dim=1)
        combined_log_probs = th.stack(log_probs, dim=1).sum(dim=1)
        combined_entropy = th.stack(entropy, dim=1)

        return combined_actions, combined_log_probs, combined_entropy, None
```

### Autoregressive Actions

**Key Idea:** Each action head sees previous action choices.

**Example:**
```python
# Head 0: Attack Style
input_0 = hidden
logits_0 = head_0(input_0)
action_0 = sample(logits_0)  # e.g., "melee_attack"

# Head 1: Melee Attack Type
input_1 = concat(hidden, one_hot(action_0))  # Knows we chose melee
logits_1 = head_1(input_1)
action_1 = sample(logits_1)  # e.g., "melee_special_attack"

# Head 2: Ranged Attack Type
input_2 = concat(hidden, one_hot(action_0, action_1))  # Knows melee chosen
logits_2 = head_2(input_2)
action_2 = sample(logits_2)  # Will be masked to "no_ranged_attack"
```

**Benefits:**
- Action heads can condition on previous choices
- More expressive than independent heads
- Naturally enforces dependencies

**Cost:**
- Sequential (can't parallelize fully)
- More complex training

### Action Normalization

```python
def _normalize(self, actions: th.Tensor) -> th.Tensor:
    mean = self.action_mean[..., :actions.shape[-1]]
    var = self.action_var[..., :actions.shape[-1]]

    actions = (actions - mean) / th.sqrt(var + 1e-8)
    actions = th.clamp(actions, -5, 5)

    return actions

def update_action_normalization(self, actions: th.Tensor) -> None:
    # Convert to one-hot
    one_hot_actions = [
        th.nn.functional.one_hot(actions[..., i], size)
        for i, size in enumerate(self.action_head_sizes)
    ]
    actions = th.cat(one_hot_actions, dim=-1).float()

    # Update running stats
    batch_mean = actions.mean(dim=0)
    batch_var = actions.var(dim=0)
    self._update_from_moments(batch_mean, batch_var, len(actions))
```

**Why normalize?**
- One-hot vectors are sparse (mostly zeros)
- Normalization helps gradient flow
- Stabilizes training

---

## Critic Network

### Architecture

```
Input: Features (hidden_size from feature extractor or raw features)
    ↓
┌────────────────────────────────────────────────┐
│          Critic Hidden Layers                  │
│  - Default: [64, 64] with ReLU                 │
│  - Orthogonal initialization (gain=√2)         │
└────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────┐
│             Value Head                         │
│  - Linear layer (hidden → 1)                   │
│  - Orthogonal initialization (gain=1)          │
└────────────────────────────────────────────────┘
    ↓
Value: scalar (expected return)
```

### Critic Class

```python
class Critic(nn.Module):
    def __init__(self, input_size: int, config: MlpConfig):
        super(Critic, self).__init__()

        # Hidden layers
        self.hidden = create_mlp(config, input_size)

        # Value head
        self.head = nn.Linear(hidden_size, 1)

        # Initialize weights
        self.hidden.apply(partial(init_weights, gain=np.sqrt(2)))
        self.head.apply(partial(init_weights, gain=1))

    def forward(self, x: th.Tensor) -> th.Tensor:
        critic_hidden = th.relu(self.hidden(x))
        value = self.head(critic_hidden)
        return value.squeeze(-1)  # (batch,) instead of (batch, 1)
```

---

## Feature Extractor (Optional)

### Purpose

Process frame-stacked observations before actor/critic:

```python
# Without feature extractor:
input → actor/critic directly

# With feature extractor:
input → feature_extractor → actor/critic
```

### Shared vs Separate

**Shared Feature Extractor:**
```python
features = feature_extractor(x)
actor_output = actor(features)
critic_output = critic(features)
```

**Benefits:**
- Fewer parameters
- Faster training
- Shared representation

**Drawbacks:**
- Actor and critic compete for features
- Less flexible

**Separate Feature Extractors:**
```python
actor_features = actor_feature_extractor(x_actor)
critic_features = critic_feature_extractor(x_critic)
```

**Benefits:**
- Independent learning
- Critic can have full observability while actor has partial
- More flexible

**Drawbacks:**
- More parameters
- Slower training

**This project uses:** Separate feature extractors (default empty)

---

## Weight Initialization

### Orthogonal Initialization

```python
def init_weights(module, gain=1):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
```

**Layers and gains:**
- Hidden layers: `gain = √2` (compensates for ReLU)
- Actor output heads: `gain = 0.01` (small initial logits)
- Critic value head: `gain = 1` (standard)

**Why orthogonal?**
- Preserves gradient magnitudes
- Prevents vanishing/exploding gradients
- Works well with deep networks

---

## Inference Optimization (TorchScript)

### Compilation

```python
# Training mode
self._policy = Policy(...)

# Inference mode (compile with TorchScript)
self._eval_policy = th.jit.freeze(th.jit.script(self._policy))
```

**Benefits:**
- ~2-3x faster inference
- Can be exported/deployed without PyTorch
- Optimized graph execution

**Limitations:**
- Must be pure PyTorch (no Python control flow)
- Harder to debug
- Compilation overhead

### Usage

```python
with th.inference_mode():  # Disable gradients
    actions, log_probs, entropy, values, _ = self._eval_policy(
        obs, action_masks, deterministic=False
    )
```

---

## Model Size

### Parameter Count

For NhEnv with default configuration:

**Actor:**
```
Hidden: [128, 128, 128]
  Layer 1: (120 → 128): 120 * 128 = 15,360
  Layer 2: (128 → 128): 128 * 128 = 16,384
  Layer 3: (128 → 128): 128 * 128 = 16,384

Action Heads: 11 heads
  Head 0: (128 → 4): 512
  Head 1: (128 + 4 → 3): 396
  Head 2: (128 + 7 → 3): 405
  ...
  Head 10: (128 + 33 → 6): 966

Total Actor: ~70,000 parameters
```

**Critic:**
```
Hidden: [64, 64]
  Layer 1: (180 → 64): 180 * 64 = 11,520
  Layer 2: (64 → 64): 64 * 64 = 4,096

Value Head: (64 → 1): 64

Total Critic: ~16,000 parameters
```

**Total Model: ~90,000 parameters** (very small by modern standards!)

**Model file size:** ~350 KB (compressed)

---

## Training vs Inference Mode

### Training Mode

```python
policy.train()

# Has gradients
# Uses dropout (if any)
# Updates batch norm stats (if any)
```

### Evaluation Mode

```python
policy.eval()

# No gradients
# No dropout
# Fixed batch norm stats
# Faster inference
```

### Inference Mode

```python
with th.inference_mode():
    # Even faster (special PyTorch mode)
    # No gradient tracking overhead
    # Can use TorchScript compiled model
```

---

## Custom Extensions

The PPO class supports **extensions** for additional outputs:

### WinRateExtension

```python
class WinRateExtension(ModelExtension):
    """Predicts probability of winning from current state."""

    def __init__(self, input_size: int, max_sequence_length: int):
        # MLP to predict win probability
        self.net = nn.Sequential(
            nn.Linear(max_sequence_length * input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def run_extension(self, obs: th.Tensor) -> th.Tensor:
        obs_flat = obs.reshape(obs.size(0), -1)
        win_prob = self.net(obs_flat)
        return win_prob

    def learn(self, buffer: Buffer, meta: Meta, summary_writer):
        # Train on actual episode outcomes
        for batch in buffer.generate_batches(256):
            win_labels = ...  # 1 if won, 0 if lost
            win_pred = self.run_extension(batch.observations)
            loss = F.binary_cross_entropy(win_pred, win_labels)
            loss.backward()
            optimizer.step()
```

### Usage

```python
# Add extension
ppo.register_extension("win_rate", WinRateExtension(...))

# Get predictions
actions, log_probs, entropy, values, probs, extensions = ppo.predict(
    obs, action_masks, extensions=["win_rate"]
)

win_prob = extensions[0]  # Probability of winning
```

---

## Memory Usage

### Per-Environment Memory

**Observations:**
```
Buffer: (256 steps) × (120 features) × 4 bytes = 123 KB
```

**Actions:**
```
Buffer: (256 steps) × (11 heads) × 4 bytes = 11 KB
```

**Values, Rewards, etc:**
```
Buffer: (256 steps) × 5 arrays × 4 bytes = 5 KB
```

**Total per environment: ~140 KB**

### Training Batch Memory

**Batch size: 256**

**Observations:**
```
(256 batch) × (3 frames) × (120 features) × 4 bytes = 369 KB
```

**Gradients:**
```
~90,000 parameters × 4 bytes × 2 (weights + gradients) = 720 KB
```

**Total training: ~2-3 MB** (very small!)

**GPU memory:** ~200 MB for model + training (can fit on tiny GPUs)

---

## Architecture Variants

### Variant 1: Shared Feature Extractor

```python
policy_kwargs = {
    'share_feature_extractor': True,
    'feature_extractor_config': [256, 256],
}
```

**Effect:**
- Single feature extractor shared by actor and critic
- Faster, fewer parameters
- Used when actor and critic need same info

### Variant 2: Larger Networks

```python
policy_kwargs = {
    'actor_config': [256, 256, 256, 256],
    'critic_config': [128, 128, 128],
}
```

**Effect:**
- More capacity for complex patterns
- Slower training
- Risk of overfitting

### Variant 3: Per-Head Configs

```python
policy_kwargs = {
    'action_head_configs': [
        [64, 64],  # Head 0: Large (important)
        [],        # Head 1: Linear (simple)
        [32],      # Head 2: Small
        ...
    ]
}
```

**Effect:**
- Customize complexity per action type
- More flexibility
- Harder to tune

### Variant 4: No Autoregressive

```python
policy_kwargs = {
    'autoregressive_actions': False,
}
```

**Effect:**
- Parallel action head computation (faster)
- Less expressive (heads independent)
- Simpler training

---

## Best Practices

### 1. **Start Simple**

Begin with default architecture:
- Separate feature extractors (empty)
- Actor: [128, 128, 128]
- Critic: [64, 64]
- Autoregressive actions

### 2. **Monitor Gradient Norms**

```python
for name, param in policy.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: {grad_norm}")
```

Large gradients → Exploding (reduce LR, clip more)
Small gradients → Vanishing (increase LR, check initialization)

### 3. **Check Value Function**

```python
# Should predict returns well
explained_variance = 1 - var(returns - values) / var(returns)
```

High EV (>0.8) → Good value function
Low EV (<0.5) → Critic not learning (increase training epochs, check LR)

### 4. **Regularization**

- Entropy bonus (exploration)
- Gradient clipping (stability)
- Observation normalization (convergence)
- Weight decay (overfitting)

### 5. **Hyperparameter Tuning**

Tune in this order:
1. Learning rate (most important)
2. Network size (if underfitting/overfitting)
3. Batch size (memory/speed tradeoff)
4. Number of epochs (sample efficiency)
5. Entropy coefficient (exploration)

---

## Summary

The neural network architecture:
1. **Processes** frame-stacked observations
2. **Extracts** features (optional shared/separate)
3. **Actor** outputs 11 autoregressive action distributions
4. **Critic** outputs single value estimate
5. **Optimized** for inference with TorchScript
6. **Extensible** with custom model extensions

**Key innovations:**
- Autoregressive actions (action dependencies)
- Action normalization (stable training)
- Separate actor/critic feature extractors (flexibility)
- TorchScript compilation (fast inference)
- Small architecture (efficient, fast training)

This architecture balances **expressiveness** (can learn complex policies) with **efficiency** (fast training and inference), enabling the AI to reach superhuman performance.
