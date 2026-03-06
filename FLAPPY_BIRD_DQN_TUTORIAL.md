# Flappy Bird DQN - Complete Step-by-Step Tutorial

## Table of Contents
1. [Project Overview](#project-overview)
2. [Reinforcement Learning Basics](#reinforcement-learning-basics)
3. [Deep Q-Network (DQN) Theory](#deep-q-network-theory)
4. [Project Architecture](#project-architecture)
5. [Component Breakdown](#component-breakdown)
6. [Training Loop Walkthrough](#training-loop-walkthrough)
7. [Hyperparameters Explained](#hyperparameters-explained)
8. [How Everything Connects](#how-everything-connects)

---

## Project Overview

**What is this project?**
This is an AI system that **trains a computer to play Flappy Bird automatically** using a machine learning technique called **Deep Q-Learning**.

**The Goal:**
Instead of writing explicit rules like "jump when you see a pipe," we let the AI learn from experience. The agent interacts with the game, tries different actions, observes the results, and gradually learns which actions lead to the best outcomes.

**Why is this useful?**
This approach works for ANY game or problem where:
- You have states (observations about the world)
- You have actions you can take
- You receive rewards/penalties based on your choices
- You want to maximize total reward over time

---

## Reinforcement Learning Basics

### The Core Concept: Agent-Environment Interaction

```
┌─────────────────────────────────────────────────────┐
│                                                       │
│  Agent (AI) ←──────── Reward, New State ────────┐   │
│      ↓                                          │    │
│   Action ──────→ Environment (Flappy Bird) ──→ Game Dynamics
│      ↓                                          │    │
│   "Should I jump?"      (Process action)        │    │
│                                                  │    │
│   Observation Space ←───────────────────────────┘    │
│   • Bird's Y position                                │
│   • Bird's velocity                                  │
│   • Pipe positions                                   │
│   • Distance to pipes                                │
│                                                       │
└─────────────────────────────────────────────────────┘
```

### State: What the Agent Sees
A **state** is the current situation. In Flappy Bird, the state might be:
- `[bird_y, bird_velocity, next_pipe_x, next_pipe_y_top, next_pipe_y_bottom, ...]`
- This is a vector of 12 numbers (according to the Flappy Bird environment)

### Action: What the Agent Can Do
The agent has **2 possible actions**:
- `0` = Do nothing (let gravity pull the bird down)
- `1` = Jump (flap wings to go up)

### Reward: Feedback on Performance
The environment gives a reward signal:
- `+1` for surviving each frame
- `-1` for crashing into a pipe or ground
- The agent wants to **maximize total reward** over an episode

### Episode: One Complete Game
- **Start:** Game begins, bird is in starting position
- **Loop:** Agent takes actions → environment responds with new state and reward
- **End:** Agent crashes (episode terminates)

---

## Deep Q-Network (Theory)

### The Q-Value Concept

A **Q-value** is the expected total future reward if you take a specific action in a specific state.

$$Q(state, action) = \text{Expected sum of future rewards}$$

**Example:**
- State = "Bird is about to hit a pipe"
- Action = "Jump"
- Q-value might be -100 (jumping won't save you)

**The Goal of Training:**
Learn accurate Q-values for all (state, action) pairs, then the optimal strategy is simple:
```
In any state, just pick the action with the highest Q-value!
```

### The Bellman Equation (Core Formula)

This is the fundamental equation of Q-Learning:

$$Q(s, a) = R + \gamma \cdot \max(Q(s', a'))$$

**Breaking it down:**
- `Q(s, a)` = Q-value for taking action `a` in state `s`
- `R` = Immediate reward you got
- `s'` = The next state you ended up in
- `a'` = Any possible action in the next state
- `γ (gamma)` = Discount factor (0.0 to 1.0) - how much you care about future rewards vs immediate rewards
  - If γ = 1.0: Future is as important as present
  - If γ = 0.5: Future is worth half as much as present
  - If γ = 0.0: Only immediate reward matters

**Intuition:**
The true value of taking an action is: the immediate reward + the best future value you can achieve

### Why Use a Neural Network?

With a simple table, you could store Q-values like:
```
State: [0.1, 0.5, 0.2, ...] → Action 0: Q-value 10.5, Action 1: Q-value 15.2
State: [0.1, 0.5, 0.3, ...] → Action 0: Q-value 10.6, Action 1: Q-value 15.3
... millions of rows needed!
```

**Problem:** With continuous states (floating point numbers), there are infinite possible states. A table is impossible!

**Solution:** Use a **neural network** to approximate Q-values:
```
Input: [bird_y, bird_velocity, pipe_x, ...] (the state)
      ↓
Neural Network (learns patterns)
      ↓
Output: [Q-value for action 0, Q-value for action 1]
```

### Why Two Networks? (Policy vs Target)

**Policy Network:** Used to choose actions during training
- Gets updated frequently based on experience
- Can be unstable (moving target)

**Target Network:** Used to calculate target Q-values
- Updated less frequently (every N steps)
- Provides stable reference points
- Prevents the network from "chasing its own tail"

**Analogy:**
- **Policy Network** = Student trying to learn
- **Target Network** = Textbook that doesn't change (stable reference)
- If the textbook changes every time you read it, you'll get confused!

---

## Project Architecture

### File Structure and Relationships

```
Flappybird/
├── agent.py              ← Main orchestrator (ties everything together)
├── dqn.py               ← Neural network architecture
├── experience_replay.py ← Memory system
└── hyperparameters.yml  ← Configuration file

Flow:
agent.py
  ├─ Loads hyperparameters.yml
  ├─ Creates DQN networks (from dqn.py)
  ├─ Creates ReplayMemory (from experience_replay.py)
  └─ Runs training loop:
     ├─ Gets state from environment
     ├─ Uses policy_dqn to pick action
     ├─ Stores transition in memory
     ├─ Samples from memory
     └─ Uses memory to train target_dqn
```

---

## Component Breakdown

### 1. DQN Neural Network (`dqn.py`)

**Purpose:** Approximate Q-values for state-action pairs

```python
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()
        
        # Input layer: takes state vector
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        
        # Output layer: outputs Q-values for each action
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU activation = rectified linear unit
        return self.fc2(x)        # Output Q-values (no activation)
```

**Architecture Breakdown:**

```
Input (12 numbers representing state)
    ↓
Dense Layer 1: 12 neurons → 128 neurons [ReLU activation]
    ↓
Dense Layer 2: 128 neurons → 2 neurons [No activation]
    ↓
Output: [Q-value for action 0, Q-value for action 1]
```

**Why No Activation on Output?**
Q-values can be negative (bad actions) or positive (good actions), so we don't restrict them.

**Why ReLU in Hidden Layer?**
ReLU introduces non-linearity, allowing the network to learn complex patterns. Without it, stacking layers would be equivalent to one layer.

### 2. Experience Replay (`experience_replay.py`)

**Purpose:** Store and sample experiences to break correlated learning

**What it stores:**
Each transition = `(state, action, new_state, reward, terminated)`

```python
class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)  # Limited size buffer
    
    def append(self, transition):
        self.memory.append(transition)      # Add experience
    
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)  # Random sampling
```

**Why is this important?**

**Without Replay Memory:**
- Episode 1: Bird crashes → learns "don't do that"
- Episode 2: Bird crashes differently → learns "don't do that either"
- All experiences are consecutive and correlated
- Network gets confused (unstable)

**With Replay Memory:**
- Store all transitions in a buffer
- When training, randomly sample 64 transitions from the buffer
- These transitions are from different times/episodes
- **Breaks correlation** → network learns more stable patterns

**Analogy:**
- Without replay: Studying only recent notes (biased)
- With replay: Randomly sampling from all your notes (balanced)

### 3. Agent (`agent.py`)

**Purpose:** Orchestrate the training process

**Main Components:**

#### a) Initialization
```python
def __init__(self, hyperparameter_set):
    # Load hyperparameters from YAML file
    # Initialize networks, memory, optimizer
```

#### b) Epsilon-Greedy Exploration
```python
if random.random() < epsilon:
    action = env.action_space.sample()  # Random action (explore)
else:
    action = policy_dqn(state).argmax() # Best known action (exploit)
```

**Why random actions?**
- If you only do what you know is best, you'll never discover better strategies
- Need to **explore** to find new knowledge

**Epsilon (ε) decay:**
- Start with ε = 1.0 (always explore)
- Gradually decay to ε = 0.01 (mostly exploit)
- Early training: Try lots of random things
- Late training: Use learned knowledge

#### c) Optimization (The Learning Step)

```python
def optimize(self, mini_batch, policy_dqn, target_dqn):
    # Calculate target Q-values using Bellman equation
    target_q = rewards + gamma * target_dqn(new_states).max() * (1 - terminations)
    
    # Calculate predicted Q-values from policy network
    current_q = policy_dqn(states).gather(actions)
    
    # Calculate difference (loss)
    loss = MSE(current_q, target_q)
    
    # Update policy network to reduce loss
    optimizer.backward()
    optimizer.step()
```

**What's happening:**
1. **Target:** Where we WANT the network to predict (using Bellman equation)
2. **Prediction:** What the network currently predicts
3. **Loss:** How wrong we are
4. **Update:** Adjust network weights to be more right

---

## Training Loop Walkthrough

### Step-by-Step Training Process

```
Initialize:
  ├─ policy_dqn (main network)
  ├─ target_dqn (copy of policy_dqn)
  ├─ replay_memory (empty buffer)
  ├─ epsilon = 1.0 (start exploring)
  └─ optimizer = Adam(learning_rate=0.00025)

For each episode:
  ├─ Reset environment (bird back at start)
  ├─ Get initial state
  │
  └─ For each step in episode (until bird crashes):
     │
     ├─ DECIDE ACTION:
     │  ├─ If random < epsilon: Pick random action (explore)
     │  └─ Else: Use policy_dqn to pick best action
     │
     ├─ INTERACT:
     │  └─ Execute action in environment → get reward + new_state
     │
     ├─ STORE EXPERIENCE:
     │  └─ memory.append((state, action, new_state, reward, terminated))
     │
     ├─ IF ENOUGH MEMORY:
     │  ├─ Sample 64 random transitions from memory
     │  ├─ Calculate target Q-values (using target_dqn)
     │  ├─ Calculate predicted Q-values (using policy_dqn)
     │  ├─ Compute loss (MSE between target and predicted)
     │  ├─ Backprop and update policy_dqn weights
     │  └─ Decay epsilon
     │
     ├─ SYNC NETWORKS:
     │  └─ Every N steps: target_dqn = policy_dqn (copy weights)
     │
     └─ Continue
```

### Concrete Example: One Training Step

**Suppose in one transition:**
- State: [0.5, -0.1, 10, 5, 15]  (bird position, velocity, pipes position)
- Action: 1 (jump)
- Reward: +1 (survived this frame)
- New State: [0.6, 0.2, 9, 5, 15]  (bird moved up, pipes moved left)
- Terminated: False (bird didn't crash)

**Using Bellman Equation:**
```
We want policy_dqn to output Q-values such that:

target_q = reward + γ * max(target_dqn(new_state))
target_q = 1 + 0.99 * max([Q for action 0, Q for action 1])
target_q = 1 + 0.99 * 5.0  (assuming best future Q-value is 5.0)
target_q = 1 + 4.95 = 5.95
```

**Current Prediction:**
```
policy_dqn([0.5, -0.1, 10, 5, 15]) → [2.0, 3.0]
We took action 1, so current_q = 3.0
```

**Loss:**
```
loss = (3.0 - 5.95)² = 8.62
Network is too pessimistic! Thinks jumping is worse than it is.
```

**Update:**
```
Backprop adjusts weights so that:
policy_dqn([0.5, -0.1, 10, 5, 15]) → [?, 4.5] (closer to 5.95)
```

---

## Hyperparameters Explained

### From `hyperparameters.yml`

```yaml
flappybird:
  env_id: FlappyBird-v0                    # Environment name
  replay_memory_size: 1000000              # Max experiences to store
  mini_batch_size: 64                      # Experiences to sample per update
  epsilon_init: 1.0                        # Start exploring 100%
  epsilon_decay: 0.9999                    # Decay factor per step
  epsilon_min: 0.01                        # Never go below 1% random
  network_sync_rate: 5000                  # Copy policy→target every 5000 steps
  learning_rate_a: 0.00025                 # Gradient descent step size
  discount_factor_g: 0.99                  # How much to value future rewards
  stop_on_reward: 100000                   # Stop episode if reward exceeds this
  fc1_nodes: 128                           # Hidden layer size
```

### Understanding Each Parameter

| Parameter | Value | Meaning | Why This Value? |
|-----------|-------|---------|-----------------|
| **replay_memory_size** | 1,000,000 | Store up to 1M experiences | Large enough to have diverse samples, but fits in RAM |
| **mini_batch_size** | 64 | Sample 64 experiences per update | Balance: large enough for stable loss, small enough for frequent updates |
| **epsilon_init** | 1.0 | 100% random actions initially | Start completely exploratory |
| **epsilon_decay** | 0.9999 | Multiply by 0.9999 each step | Very slow decay (10k steps ≈ 90% original) |
| **epsilon_min** | 0.01 | Don't go below 1% random | Always try some random actions (avoid local optima) |
| **learning_rate_a** | 0.00025 | Small gradient step | Small learning rate = stable training |
| **discount_factor_g** | 0.99 | Weight future ≈ 99% of present | Future rewards matter almost as much as immediate |
| **network_sync_rate** | 5000 | Update target network every 5000 steps | Balance: frequent enough to progress, infrequent enough for stability |
| **fc1_nodes** | 128 | Hidden layer has 128 neurons | Large enough for complex patterns, small enough to train fast |

### Quick Reference Formulas

```
After step N:
  epsilon = max(epsilon_init * (epsilon_decay ^ N), epsilon_min)
  epsilon = max(1.0 * (0.9999 ^ N), 0.01)

After 1000 steps:  epsilon ≈ 0.90
After 10000 steps: epsilon ≈ 0.37
After 100000 steps: epsilon ≈ 0.0000003 (capped at 0.01)
```

---

## How Everything Connects

### The Complete Training Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                          AGENT.RUN()                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Initialize Components                         │
│                                                                  │
│  policy_dqn = DQN(12, 2, 128)        ← From dqn.py             │
│  target_dqn = copy of policy_dqn                                │
│  memory = ReplayMemory(1M)           ← From experience_replay.py│
│  epsilon = 1.0                                                   │
│  optimizer = Adam(lr=0.00025)                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    FOR EACH EPISODE
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Reset Game Environment                         │
│           state = [bird_y, bird_v, pipe_x, ...]                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    WHILE GAME NOT OVER
                              ↓
        ┌───────────────────────────────────────┐
        │      1. SELECT ACTION (Epsilon-Greedy)│
        │                                       │
        │  if random() < ε:                     │
        │    action = random choice             │
        │  else:                                │
        │    ~~~~~~~~~~~~~~~~~~~~~~~~~          │
        │    Q-values = policy_dqn(state)      │ ← Uses DQN
        │    action = argmax(Q-values)         │
        │  ~~~~~~~~~~~~~~~~~~~~~~~~~           │
        └───────────────────────────────────────┘
                              ↓
        ┌───────────────────────────────────────┐
        │    2. TAKE ACTION IN ENVIRONMENT      │
        │                                       │
        │  reward, new_state = env.step(action)│
        │                                       │
        │  Example:                             │
        │  action = 1 (jump)                   │
        │  reward = +1 (survived frame)        │
        │  new_state = [bird_y', bird_v', ...] │
        └───────────────────────────────────────┘
                              ↓
        ┌───────────────────────────────────────┐
        │    3. STORE IN REPLAY MEMORY          │
        │                                       │
        │  memory.append(                       │
        │    (state, action, new_state,        │
        │     reward, terminated)               │
        │  )                                     │ ← Uses ReplayMemory
        │                                       │
        │  Buffer now has [1, 2, 3, ..., N]    │
        └───────────────────────────────────────┘
                              ↓
        IF memory has enough samples (>64)
                              ↓
        ┌───────────────────────────────────────┐
        │   4. TRAIN POLICY NETWORK             │
        │                                       │
        │  mini_batch = memory.sample(64)      │ ← Sample from memory
        │                                       │
        │  BELLMAN EQUATION:                    │
        │  ─────────────────────────────────────│
        │  target_Q = reward +                  │
        │             γ × max(target_dqn())    │
        │                                       │
        │  For new_states batch:                │
        │  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     │
        │  future_Q = target_dqn(new_states)   │ ← Uses target network
        │  max_future_Q = future_Q.max()       │
        │  target_Q = rewards + 0.99 × max_future│
        │  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     │
        │                                       │
        │  PREDICTION:                          │
        │  ─────────────────────────────────────│
        │  current_Q = policy_dqn(states)      │ ← Uses policy network
        │  current_Q = select actions used     │
        │                                       │
        │  LOSS:                                │
        │  ─────────────────────────────────────│
        │  loss = MSE(current_Q, target_Q)     │
        │                                       │
        │  UPDATE:                              │
        │  ─────────────────────────────────────│
        │  optimizer.zero_grad()                │
        │  loss.backward() ← Backpropagation   │
        │  optimizer.step() ← Update weights   │
        │                                       │
        │  Decay epsilon:                       │
        │  ε = max(ε × 0.9999, 0.01)          │
        └───────────────────────────────────────┘
                              ↓
        IF step_count > 5000
                              ↓
        ┌───────────────────────────────────────┐
        │    5. SYNC TARGET NETWORK             │
        │                                       │
        │  target_dqn = copy(policy_dqn)       │
        │  step_count = 0                       │
        │                                       │
        │  (Stabilize learning by giving       │
        │   target a stable reference)          │
        └───────────────────────────────────────┘
                              ↓
                          Continue...
```

### Module Interaction Diagram

```
┌──────────────────────────────────┐
│    hyperparameters.yml           │
│  (All configuration values)      │
└──────────────┬───────────────────┘
               │ (loaded by)
               ↓
    ┌──────────────────────┐
    │   Agent.__init__()   │ ← agent.py
    └──────────────────────┘
               │
        ┌──────┼──────┐
        ↓      ↓      ↓
    ┌───────────────────┐    ┌──────────────┐    ┌──────────────────────┐
    │  DQN Networks     │    │ Replay Memory│    │  Optimizer (Adam)    │
    │  (from dqn.py)    │    │(from exp...) │    │  Training Loop       │
    │                   │    │              │    │  Visualization       │
    │ policy_dqn ███    │    │  memory ███  │    │                      │
    │ target_dqn ███    │    │              │    │  agent.run()         │
    │                   │    │  .append()   │    │  agent.optimize()    │
    │ forward(state)    │    │  .sample()   │    │  agent.save_graph()  │
    │  → Q-values       │    │              │    │                      │
    └───────────────────┘    └──────────────┘    └──────────────────────┘
```

---

## Summary: The Big Picture

### What Happens During Training

```
Episode 1:
  Step 1: State=[...], Action=random, Reward=+1, Store in memory
  Step 2: State=[...], Action=random, Reward=+1, Store in memory
  Step 3: State=[...], Action=random, Reward=-1 (CRASH), Episode ends
  → Network hasn't seen enough experiences, doesn't train much

Episode 50:
  Memory has ~1000 experiences
  Step 1: State=[...], Action=best, Reward=+1, Store in memory
  Step 2: Sample 64 experiences, Train network
  Step 3: loss=0.8, update weights
  Step 10: Sample 64 experiences, Train network
  Step 11: loss=0.5, policies improving!
  Step 50: State=[...], Action=best, Reward=-1 (CRASH)

Episode 500:
  Memory has ~500k experiences (very diverse)
  Training is stable (sampling from diverse experiences)
  Loss keeps decreasing
  Agent is better at avoiding crashes
  High-value actions are clearly Q-valued

Episode 5000:
  Epsilon ≈ 0.01 (mostly exploiting, rarely exploring)
  Agent can survive many frames
  Best reward keeps increasing
  Model is saved when new best is found
```

### Key Insight

The agent learns **indirectly** through trial and error:
1. Take actions → see results
2. Store what happens in memory
3. Learn patterns from random samples of past
4. Update predictions about future rewards
5. These better predictions lead to better actions
6. repeat...

**You don't program the strategy. The network discovers it through experience!**

---

## Quick Reference: Variables and What They Mean

| Variable | Type | Meaning |
|----------|------|---------|
| `state` | Tensor(12) | Bird y, velocity, pipe positions, etc. |
| `action` | int | 0=nothing, 1=jump |
| `reward` | float | +1 for surviving, -1 for crashing |
| `new_state` | Tensor(12) | State after taking action |
| `terminated` | bool | Is the episode over? |
| `epsilon (ε)` | float | Probability of random action |
| `Q-value` | float | Expected future reward for (state, action) |
| `loss` | float | Error between prediction and target |
| `policy_dqn` | Network | Learns from experience (gets updated) |
| `target_dqn` | Network | Provides stable targets (updated slowly) |

---

## Next Steps: Understanding Code

1. **Read `dqn.py` first** → Simple, just a neural network
2. **Read `experience_replay.py` second** → Simple, just stores and samples
3. **Read `agent.py` last** → Ties everything together
4. **Trace through one episode** → Follow a single game with your understanding

## Exercises to Test Your Understanding

1. What would happen if you removed epsilon-greedy and only did exploitation?
2. What would happen if target_dqn updated every step instead of every 5000?
3. Why is randomness (epsilon) important early but not late?
4. How would adding a third action (turn left/right) change things?
5. Could this work for Chess? Why/why not?

---

**Happy Learning! 🚀**
