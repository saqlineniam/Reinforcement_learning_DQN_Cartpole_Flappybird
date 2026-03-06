# Flappy Bird DQN - Code Walkthrough with Annotations

## File 1: DQN Neural Network (`dqn.py`)

This is the **simplest file** - just a standard neural network.

```python
import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    A simple neural network that predicts Q-values.
    
    Input:  State vector (12 numbers)
    Output: Q-values for each action (2 numbers)
    
    The network learns: "In this state, this action is good/bad"
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        state_dim:   How many numbers in state (12 for Flappy Bird)
        action_dim:  How many actions possible (2: do nothing or jump)
        hidden_dim:  Size of middle layer (128 for Flappy Bird)
        """
        super(DQN, self).__init__()
        
        # Layer 1: Convert state (12 numbers) to hidden (128 numbers)
        # This is where the network learns patterns
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        
        # Layer 2: Convert hidden (128 numbers) to Q-values (2 numbers)
        # Output: One Q-value for each action
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        """
        What happens when you pass a state through the network
        
        Input x:  [batch_size, 12]        (e.g., [64, 12])
        Output:   [batch_size, 2]         (e.g., [64, 2])
        """
        # Pass through layer 1, apply ReLU activation
        # ReLU = max(0, x) - turns negatives to 0
        # This adds non-linearity (network can learn curves, not just straight lines)
        x = F.relu(self.fc1(x))
        
        # Pass through layer 2 (output layer)
        # NO activation here - Q-values can be any number
        # (positive for good actions, negative for bad actions)
        return self.fc2(x)


# Example usage:
if __name__ == '__main__':
    state_dim = 12      # Flappy Bird state has 12 numbers
    action_dim = 2      # 2 possible actions
    net = DQN(state_dim, action_dim)
    
    # Create dummy state (batch of 10 states, each with 12 numbers)
    state = torch.randn(10, state_dim)
    
    # Feed through network
    output = net(state)  # shape: [10, 2]
    
    # output[0] = [Q-value for action 0, Q-value for action 1]
    # The network is predicting: "In these states, which action is better?"
    print(output)
```

### Simple Explanation

```
Your Brain:      "In this situation, should I jump or not?"
DQN Network:     Processes sensory info → makes a decision
Output:          "Jump score = 7.5, Don't jump score = 3.2"
                 "Jump is better!"
```

---

## File 2: Experience Replay (`experience_replay.py`)

Stores memories so the agent can learn from diverse past experiences.

```python
from collections import deque
import random

class ReplayMemory:
    """
    A buffer that stores and replays experiences.
    
    Why? So we can learn from random samples instead of just recent events.
    Similar to studying from random past exams instead of just today's problems.
    """
    
    def __init__(self, maxlen, seed=None):
        """
        maxlen: Maximum number of experiences to store (1,000,000)
        seed:   For reproducible randomness (for debugging)
        """
        # deque = double-ended queue
        # When full, automatically removes oldest element when you add a new one
        self.memory = deque(maxlen=maxlen)
        
        if seed is not None:
            random.seed(seed)
    
    def append(self, transition):
        """
        Store one experience.
        
        transition = (state, action, new_state, reward, terminated)
        
        Example:
        state = [0.1, 0.5, 10, 5, ...]     (bird position, velocity, pipes)
        action = 1                           (jump)
        new_state = [0.2, 0.6, 9, 5, ...]  (after jumping)
        reward = 1                           (survived this frame)
        terminated = False                   (bird didn't crash)
        """
        self.memory.append(transition)
    
    def sample(self, sample_size):
        """
        Get a random sample of past experiences.
        
        sample_size: How many experiences to sample (64)
        
        Returns: List of 64 random transitions from the entire buffer
        
        Example:
        If buffer has [exp1, exp2, exp3, ..., exp10000]
        sample(64) might return [exp1, exp8732, exp345, exp6123, ...]
        
        This randomness is KEY! It breaks the correlation between
        consecutive experiences, making training more stable.
        """
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        """Return how many experiences are stored"""
        return len(self.memory)


# Visual Example:
"""
Step 1: Episode 1, Step 1
    transition = ([bird state], 1, [new state], +1, False)
    memory.append(transition)
    memory = [transition_1]
    
Step 2: Episode 1, Step 2
    transition = ([bird state], 0, [new state], +1, False)
    memory.append(transition)
    memory = [transition_1, transition_2]
    
...more steps...

Step 65: Episode 2, Step 15
    transition = ([bird state], 1, [new state], -1, True)
    memory.append(transition)
    memory has 64+ items now!
    
    Now we can sample:
    mini_batch = memory.sample(64)
    # Returns random mix: [transition_5, transition_25, transition_1, ...]
    # These are from different times/episodes, not consecutive!
```

---

## File 3: The Main Agent (`agent.py`)

This ties everything together. It's the "brain" that controls training.

### Initialization

```python
def __init__(self, hyperparameter_set):
    """
    Set up the agent with hyperparameters from the YAML file.
    """
    
    # Load hyperparameters from file
    # Example: hyperparameter_set = "flappybird"
    # Loads: learning_rate=0.00025, epsilon_decay=0.9999, etc.
    with open("Flappybird/hyperparameters.yml", "r") as file:
        all_hyperparameter_sets = yaml.safe_load(file)
        hyperparameters = all_hyperparameter_sets[hyperparameter_set]
    
    # Store each hyperparameter as an instance variable
    self.env_id = "FlappyBird-v0"        # Which game to play
    self.learning_rate_a = 0.00025       # How fast to learn
    self.discount_factor_g = 0.99        # How much to value future
    self.network_sync_rate = 5000        # Update target network every 5000 steps
    # ... more hyperparameters
```

### Main Training Loop

```python
def run(self, is_training=True, render=False):
    """
    The CORE of everything - the training/evaluation loop.
    
    is_training=True:  Train the agent, save model
    is_training=False: Load trained model, show it playing
    """
    
    # Create the game environment
    env = gym.make("FlappyBird-v0", render_mode=None)
    
    # Get dimensions
    num_actions = 2        # Jump or don't jump
    num_states = 12        # State vector size
    
    # Initialize data tracking
    reward_per_episode = []  # Track score for each episode
    
    if is_training:
        # ==== SETUP FOR TRAINING ====
        
        # Create the POLICY network (learns from experience)
        policy_dqn = DQN(num_states, num_actions, hidden_dim=128).to(device)
        
        # Create the TARGET network (provides stable targets)
        target_dqn = DQN(num_states, num_actions, hidden_dim=128).to(device)
        
        # Copy policy weights to target (start identical)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        
        # Create memory buffer (stores up to 1,000,000 experiences)
        memory = ReplayMemory(1000000)
        
        # Create optimizer (Adam - a smart version of gradient descent)
        optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=0.00025)
        
        # Start exploring 100% of the time
        epsilon = 1.0
        epsilon_history = []  # Track epsilon over time
        
        # Loss function (Mean Squared Error)
        loss_fn = nn.MSELoss()
    else:
        # ==== SETUP FOR EVALUATION ====
        # Load trained model
        policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
        policy_dqn.eval()  # Set to evaluation mode (don't update weights)
    
    
    # ==== MAIN TRAINING LOOP ====
    
    for episode in itertools.count():  # Loop forever (until stopped)
        
        # Start a new game
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float, device=device)
        
        terminated = False
        episode_reward = 0.0
        
        # ---- PLAY ONE EPISODE ----
        
        while not terminated:
            
            # ========== STEP 1: CHOOSE ACTION ==========
            
            if is_training and random.random() < epsilon:
                # EXPLORATION: Pick random action
                # This means: "Try something new to learn"
                action = env.action_space.sample()
            else:
                # EXPLOITATION: Pick best known action
                # This means: "Use what you've learned"
                with torch.no_grad():  # Don't track gradients (just evaluating)
                    # policy_dqn(state) returns Q-values: [Q for action 0, Q for action 1]
                    # .squeeze() removes batch dimension
                    # .argmax() picks the action with highest Q-value
                    q_values = policy_dqn(state.unsqueeze(0))  # Add batch dim
                    action = q_values.squeeze().argmax()
            
            # ========== STEP 2: TAKE ACTION ==========
            # Execute action in environment, get feedback
            new_state, reward, terminated, _, info = env.step(action.item())
            
            # Convert to tensors
            new_state = torch.tensor(new_state, dtype=torch.float, device=device)
            reward = torch.tensor(reward, dtype=torch.float, device=device)
            
            # Accumulate total reward for this episode
            episode_reward += reward.item()
            
            # ========== STEP 3: STORE EXPERIENCE ==========
            # Remember this for later training
            if is_training:
                memory.append((state, action, new_state, reward, terminated))
            
            # ========== STEP 4: TRAIN ON BATCH ==========
            # If we have enough memories, learn from them
            if is_training and len(memory) > 64:
                
                # Sample 64 random experiences
                mini_batch = memory.sample(64)
                
                # Call the optimization function
                self.optimize(mini_batch, policy_dqn, target_dqn, optimizer, loss_fn)
                
                # Decay epsilon (gradually stop exploring)
                epsilon = max(epsilon * 0.9999, 0.01)
                epsilon_history.append(epsilon)
                
                # Every 5000 steps, copy policy to target (stabilize)
                step_count += 1
                if step_count > 5000:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0
            
            # Move to next state
            state = new_state
        
        # ---- EPISODE ENDED (bird crashed) ----
        
        reward_per_episode.append(episode_reward)
        
        if is_training:
            # Save model if this is the best performance yet
            if episode_reward > best_reward:
                torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                best_reward = episode_reward
                print(f"New best reward: {episode_reward} at episode {episode}")
```

### The Optimization Step

```python
def optimize(self, mini_batch, policy_dqn, target_dqn, optimizer, loss_fn):
    """
    This is where the actual learning happens!
    
    Uses the Bellman Equation to update the network.
    """
    
    # Unpack batch into separate lists
    states, actions, new_states, rewards, terminations = zip(*mini_batch)
    
    # Convert to tensors and stack
    states = torch.stack(states)                    # [64, 12]
    actions = torch.stack(actions)                  # [64, 1]
    new_states = torch.stack(new_states)            # [64, 12]
    rewards = torch.stack(rewards)                  # [64]
    terminations = torch.tensor(terminations).float()  # [64]
    
    # ========== CALCULATE TARGET Q-VALUES ==========
    # This is what we WANT the network to predict
    
    with torch.no_grad():  # Don't need gradients for target
        # Get Q-values for all actions in next states from TARGET network
        next_q_values = target_dqn(new_states)  # [64, 2]
        
        # Pick the best Q-value for each next state
        max_next_q = next_q_values.max(dim=1)[0]  # [64]
        
        # Apply Bellman Equation:
        # Q(s,a) = R + γ * max(Q(s', a'))
        # Where (1-terminations) ensures we ignore future if episode ended
        target_q = rewards + (1 - terminations) * 0.99 * max_next_q
    
    # ========== CALCULATE CURRENT Q-VALUES ==========
    # This is what the POLICY network currently predicts
    
    # Get Q-values from policy network
    current_q_values = policy_dqn(states)  # [64, 2]
    
    # Extract Q-value for the action that was actually taken
    # gather(1, actions) selects the Q-value for the action taken
    current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()  # [64]
    
    # ========== CALCULATE LOSS ==========
    # How wrong is the network?
    
    loss = loss_fn(current_q, target_q)  # MSE between what it predicts vs what it should predict
    
    # ========== BACKPROPAGATION ==========
    # Update the network to reduce loss
    
    optimizer.zero_grad()   # Clear old gradients
    loss.backward()         # Calculate gradients (backprop)
    optimizer.step()        # Update weights along gradients
    
    # Result: policy_dqn has improved slightly!
```

---

## The Complete Data Flow (One Step)

```
Input:
  State = [0.5, -0.1, 10, 5, 15, ...]    (bird y, velocity, pipes pos)

↓ FORWARD PASS (deciding action)

Policy DQN:
  Input: [0.5, -0.1, 10, 5, 15, ...]
  Layer 1: Linear(12→128) + ReLU
           → [0.2, 0.0, 1.5, ... 128 values ...]
  Layer 2: Linear(128→2)
           → [3.5, 2.1]  ← Q-values for each action
  
  Action: argmax([3.5, 2.1]) = action 0 (don't jump)

↓ ENVIRONMENT

Environment: Execute action 0 (don't jump)
  State + Physics = New State
  Reward = +1 (bird survived)
  Terminated = False

↓ STORE

Memory: append((state, 0, new_state, +1, False))

↓ TRAINING (after 64 batch collected)

Target DQN:
  Input: new_state = [0.6, 0.0, 9, 5, 15, ...]
  Output: [4.2, 3.8]
  max(Q) = 4.2

Target Q-value = 1 + 0.99 * 4.2 = 5.158

Policy DQN (same inputs):
  Output: [3.5, 2.1]
  For action 0: Q_predicted = 3.5

Loss = (3.5 - 5.158)² = 2.73

Backprop adjusts weights so next time:
  Policy DQN([0.5, -0.1, 10, 5, 15, ...]) → [3.7, ?]
  Slightly higher Q-value for action 0!
```

---

## Key Concepts Summary

### The Three Networks

```
┌─────────────────────────────────────────────────────────┐
│ POLICY NETWORK (gets updated)                           │
│ • Used to pick actions during training                  │
│ • Gets trained every step                               │
│ • Can be unstable (keeps moving)                        │
│ • saved_model.pth is a copy of this                     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ TARGET NETWORK (stable reference)                       │
│ • Used to calculate target Q-values                     │
│ • Gets copied from policy every 5000 steps              │
│ • Stays stable (doesn't change every step)              │
│ • Provides "ground truth" for training                  │
└─────────────────────────────────────────────────────────┘
```

### Why Two Networks?

```
Analogy: Student taking an exam

❌ Bad: Textbook changes every time student reads it
           Student gets confused (unstable learning)

✅ Good: Textbook stays the same
         Student studies, takes exam, improves
         Then textbook updates to match their learning
         Repeat...
```

### Epsilon-Greedy Strategy

```
Early Training (ε = 1.0):
  Random Action:  100%
  Best Action:    0%
  → Agent explores EVERYTHING

Mid Training (ε = 0.5):
  Random Action:  50%
  Best Action:    50%
  → Agent tries new things AND uses learning

Late Training (ε = 0.01):
  Random Action:  1%
  Best Action:    99%
  → Agent uses what it learned (mostly)
```

---

## Common Questions

**Q: Why do we store experiences instead of training immediately?**
A: Because consecutive experiences are correlated. The bird is in similar states doing similar things. Training on random samples from memory breaks this correlation, leading to more stable learning.

**Q: Why do we need two networks?**
A: If we use the same network for both prediction and target, we're chasing a moving target. It's like trying to hit a ball that keeps moving. Two networks (one fast-updating, one slow-updating) provides stability.

**Q: What does "no_grad()" mean?**
A: It tells PyTorch "don't track gradients for this part." Gradients are needed for backprop, but when we're just evaluating (not training), we don't need them.

**Q: Why ReLU activation?**
A: Without activation, each layer would just be matrix multiplication. Stacking layers would be equivalent to one layer. ReLU adds non-linearity, allowing the network to learn curves and complex patterns.

**Q: What happens if learning rate is too high?**
A: Network bounces around, can't converge. Updates are too aggressive.
Too low? Network learns very slowly.
0.00025 is small but works for this problem.

---

## Visual Timeline

```
Episode 1:
  [Random action] → Crash early → No training (not enough memory)

Episode 10:
  [Mostly random] → Crash → Train on 10 experiences → Loss = 5.2

Episode 50:
  [Random 50%, best 50%] → Crash → Train on 1000 experiences → Loss = 3.1

Episode 100:
  [Random 20%, best 80%] → Crash → Train on 5000 experiences → Loss = 1.8
  
Episode 500:
  [Random 1%, best 99%] → Survives 50 frames → Loss = 0.3
  [Network learned good strategy!]

Episode 1000:
  [Mostly exploitation] → Survives 100+ frames → Loss = 0.1
  [Agent is expert]
```

---

**Remember:** The network doesn't have a rule book. It learns from *experience* what actions lead to good outcomes. This is the power of Deep Reinforcement Learning! 🎮🤖
