# Flappy Bird DQN - Quick Reference Cheat Sheet

## 30-Second Summary

**What:** An AI that learns to play Flappy Bird using Deep Q-Learning
**How:** By trying actions, observing results, storing memories, and learning patterns from past experiences
**Result:** A neural network that predicts which actions are good in which situations

---

## The 4 Main Components

### 1. **DQN Network** (`dqn.py`)
```
State [12 numbers] → Linear(12→128) + ReLU → Linear(128→2) → Q-values [2 numbers]
                     "What's in the world?"  "What do I know?"   "How good is each action?"
```
**What it does:** Predicts: "In this situation, should I jump (value=5) or not jump (value=3)?"

### 2. **Replay Memory** (`experience_replay.py`)
```
Stores: [(state, action, new_state, reward, terminated), ...]
        [   "What I saw", "What I did", "What happened", "Score change", "Game over?"]
```
**What it does:** Bank of past experiences. Randomly sample from it to train.

### 3. **Optimizer** (PyTorch Adam)
```
Calculates: How wrong is the network?
Updates: Network weights to be more right
```
**What it does:** Makes the network learn from mistakes.

### 4. **Training Loop** (`agent.py`)
```
for each episode:
  for each step:
    see state → pick action → get reward → store experience
    if have enough memories:
      sample random batch → calculate loss → update network → decay exploration
```
**What it does:** Orchestrates the entire learning process.

---

## Core Equation: Bellman Backup

The formula that makes everything work:

```
Target Q-value = Immediate Reward + Discount * Best Future Q-value

Q(s,a) = r + γ × max(Q(s', a'))

r = 1 for surviving, -1 for crashing
γ = 0.99 (future is almost as important as now)
```

**In code:**
```python
target_q = reward + 0.99 * target_dqn(new_state).max()
```

---

## Hyperparameters Explained (1 sentence each)

| Name | Value | Purpose |
|------|-------|---------|
| `replay_memory_size` | 1M | How many past experiences to remember |
| `mini_batch_size` | 64 | How many experiences to learn from at once |
| `epsilon_init` | 1.0 | Start 100% random (explore) |
| `epsilon_decay` | 0.9999 | Slowly reduce randomness each step |
| `epsilon_min` | 0.01 | Never drop below 1% random (keep exploring) |
| `learning_rate_a` | 0.00025 | How aggressive to update network |
| `discount_factor_g` | 0.99 | How much to care about future (99%) vs now (1%) |
| `network_sync_rate` | 5000 | Copy policy → target every 5000 steps (stability) |
| `fc1_nodes` | 128 | Hidden layer size (bigger = more complex patterns) |

---

## Training Process (Step by Step)

```
1️⃣  RESET: New game starts
    state = [bird_y, bird_vel, pipe_x, ...]

2️⃣  DECIDE: Should I jump?
    if random() < epsilon:
        action = random (explore: "try something new")
    else:
        action = policy_dqn(state).argmax() (exploit: "use what I know")

3️⃣  EXECUTE: Take action in game
    new_state, reward, terminated = env.step(action)

4️⃣  REMEMBER: Store this experience
    memory.append((state, action, new_state, reward, terminated))

5️⃣  LEARN: If enough memories, train
    batch = memory.sample(64)  # Random experiences
    
    # Calculate what we WANT to predict
    target_q = reward + 0.99 * target_dqn(new_state).max()
    
    # Calculate what we ACTUALLY predict  
    current_q = policy_dqn(state)[action_taken]
    
    # How wrong?
    loss = (current_q - target_q)²
    
    # Update network to reduce loss
    backprop(loss)

6️⃣  STABILIZE: Every 5000 steps
    target_dqn = copy(policy_dqn)

7️⃣  EXPLORE LESS: Decay epsilon
    epsilon = epsilon * 0.9999

8️⃣  REPEAT: Continue until bird crashes, start new episode
```

---

## The Two Networks Explained

```
POLICY NETWORK (The Fast Learner)
├─ Gets trained every step
├─ Updates weights constantly
├─ Can be unstable (moving target)
└─ Used to choose actions

TARGET NETWORK (The Stable Reference)
├─ Gets copied from policy every 5000 steps
├─ Doesn't change much
├─ Provides stable "ground truth"
└─ Used to calculate target values

Why both?
If you use same network for both:
  - You're learning from something that keeps changing
  - Like chasing a ball that moves every time you get close
  - Unstable! → Doesn't converge

With two networks:
  - Policy learns from fixed target
  - Target occasionally updates to match policy
  - Stable! → Converges to good solution
```

---

## Common Values & What They Mean

```python
# What you'll see in code

state          # [0.5, -0.1, 10, 5, ...]     Bird position, velocity, pipes
action         # 0 or 1                       0=nothing, 1=jump
reward         # 1.0 or -1.0                  Positive=good, negative=bad
epsilon        # starts at 1.0, goes to 0.01  Probability of random action
loss           # starts at ~10, goes to ~0.1  How wrong the network is
episode_reward # accumulates during episode   Total score for this game

# Network outputs
q_values       # [3.5, 2.1]                   Action 0 looks better
best_action    # argmax([3.5, 2.1]) = 0       Pick the higher Q-value
```

---

## Key Concepts

### Epsilon-Greedy
```
Pick random action with probability ε (explore)
Pick best known action with probability (1-ε) (exploit)

When ε = 1.0:  Always explore (try everything)
When ε = 0.5:  Half explore, half exploit
When ε = 0.01: Mostly exploit (use learned knowledge)
```

### Experience Replay
```
❌ Without: Learn from recent experiences only → Correlated, unstable
✅ With:    Learn from random past experiences → Decorrelated, stable
```

### Batch Learning
```
❌ Update after 1 experience: Noisy, unstable
✅ Update after 64 experiences: Smoother, more stable
   (Averaging reduces noise)
```

---

## State Space Visualization

```
What the agent sees (12 numbers):

[bird_y, bird_velocity, 
 next_pipe_x, next_pipe_top, next_pipe_bottom,
 pipe2_x, pipe2_top, pipe2_bottom,
 pipe3_x, pipe3_top, pipe3_bottom,
 ... more pipe info ...]

Example values:
[0.5,     -0.1,           
 10,      5,         15,
 20,      4,         14,
 30,      6,         16,
 ...]

Network learns:
"If (bird_y=0.5, bird_v=-0.1, next_pipe=10)
  then action_1 (jump) has Q-value = 7.2
  and action_0 (nothing) has Q-value = 2.1
  → Pick action 1!"
```

---

## Training Progress Markers

```
Episode 1-10:
  ❌ Bird crashes immediately
  📊 Loss very high (~5-10)
  🔄 Not enough data to train
  
Episode 50:
  ⚠️  Bird survives 5-10 frames
  📊 Loss decreasing (~2-3)
  ✅ Can start training
  
Episode 500:
  😊 Bird survives 30-50 frames
  📊 Loss much lower (~0.5)
  🧠 Learning good strategies
  
Episode 5000:
  🎉 Bird survives 100+ frames!
  📊 Loss very low (~0.1)
  🤖 Network is confident in decisions
```

---

## Quick Troubleshooting

| Problem | Reason | Fix |
|---------|--------|-----|
| Loss doesn't decrease | Learning rate too low | Increase `learning_rate_a` |
| Loss bounces wildly | Learning rate too high | Decrease `learning_rate_a` |
| Agent explores too much | Epsilon decays too slow | Increase `epsilon_decay` (e.g., 0.9995) |
| Agent doesn't explore | Epsilon decays too fast | Decrease `epsilon_decay` (e.g., 0.999) |
| Training unstable | Target network updates too freq. | Increase `network_sync_rate` |
| Training slow | Batch too small | Increase `mini_batch_size` |

---

## One-Liner Explanations

- **State**: What the bird sees right now
- **Action**: What the bird can do (jump or not)
- **Reward**: Score for this action (+1 good, -1 bad)
- **Q-value**: "How good is this action in this state?"
- **Episode**: One complete game (until crash)
- **Batch**: 64 random experiences to learn from
- **Loss**: How wrong the network is
- **Epsilon**: "Should I try something new (random) or use what I learned?"
- **Policy Network**: The learner (gets updated constantly)
- **Target Network**: The reference (gets updated occasionally)

---

## Formula Cheat Sheet

```
Epsilon decay:
  ε_new = max(ε_old × 0.9999, 0.01)

Bellman equation:
  Q(s,a) = r + γ × max(Q(s',a'))

Neural network forward pass:
  output = fc2(ReLU(fc1(input)))

MSE Loss:
  loss = (predicted - target)²

Update rule:
  weights = weights - learning_rate × gradient
```

---

## Files Quick Map

```
hyperparameters.yml
  └─ Contains all configurations
     └─ agent.py loads this at startup

dqn.py
  └─ Defines DQN class
     └─ agent.py creates policy_dqn and target_dqn from this

experience_replay.py
  └─ Defines ReplayMemory class
     └─ agent.py creates memory object from this

agent.py (the conductor)
  ├─ Loads hyperparameters.yml
  ├─ Creates DQN networks ← dqn.py
  ├─ Creates ReplayMemory ← experience_replay.py
  ├─ Runs training/evaluation loop
  └─ Saves model weights
```

---

## How to Read the Code

**Start with this order:**
1. Read `dqn.py` (5 min) - Simple neural network
2. Read `experience_replay.py` (3 min) - Simple buffer
3. Read `agent.py` `__init__` (2 min) - Setup
4. Read `agent.py` `run()` (10 min) - Main loop
5. Read `agent.py` `optimize()` (5 min) - Learning step
6. Read `hyperparameters.yml` (2 min) - Settings

**Total: ~27 minutes to understand the whole thing!**

---

## Quick Run Commands

```bash
# Train the agent
python agent.py flappybird --train

# Evaluate trained agent (watch it play)
python agent.py flappybird
```

During training, the progress logs and graphs save to `runs/` folder.

---

**Remember:** This is just **trial and error at scale**. The network tries things, sees results, remembers them, and gradually learns what works. No magic—just math! ✨

