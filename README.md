# Deep Q-Learning Agent for FlappyBird and CartPole

Train and run reinforcement learning agents using Deep Q-Learning (DQN) for FlappyBird and CartPole environments. This project implements a complete DQN training pipeline with experience replay, target networks, and epsilon-greedy exploration.

## Overview

This project demonstrates the implementation of a Deep Q-Network (DQN) agent that learns to play games through reinforcement learning. The agent uses a neural network to approximate Q-values and an experience replay buffer to break correlations in training data. Two environments are supported: FlappyBird and CartPole.

## Features

- **Deep Q-Learning (DQN)**: Full implementation of DQN with target networks and experience replay
- **Multiple Environments**: Support for FlappyBird-v0 and CartPole-v1
- **Real-time Visualization**: Plots mean reward over episodes and epsilon decay during training
- **Model Checkpointing**: Automatically saves the best model based on episode reward
- **Configurable Hyperparameters**: YAML-based configuration for different environment setups
- **Training & Evaluation Modes**: Dedicated modes for training agents and running trained models
- **Detailed Logging**: Comprehensive training logs with timestamps and milestone rewards

## Requirements

The project requires the following Python packages:

```
torch>=1.9.0
gymnasium>=0.27.0
flappy-bird-gymnasium>=0.3.3
numpy>=1.21.0
matplotlib>=3.4.0
pyyaml>=5.4.0
```

**Python Version**: 3.8 or higher

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install torch gymnasium flappy-bird-gymnasium numpy matplotlib pyyaml
```

3. For GPU support (optional):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

The main entry point is `agent.py` in the `Flappybird` directory. The agent supports different hyperparameter sets configured in `hyperparameters.yml`.

### Basic Commands

#### CartPole Training

Train the agent on the CartPole-v1 environment:

```bash
python Flappybird/agent.py cartpole1 --train
```

#### CartPole Evaluation

Run a trained CartPole agent:

```bash
python Flappybird/agent.py cartpole1
```

#### FlappyBird Training

Train the agent on the FlappyBird-v0 environment:

```bash
python Flappybird/agent.py flappybird --train
```

#### FlappyBird Evaluation

Run a trained FlappyBird agent:

```bash
python Flappybird/agent.py flappybird
```

## Training

### Training Command

To train an agent, use the `--train` flag:

```bash
python Flappybird/agent.py <environment> --train
```

Replace `<environment>` with either `flappybird` or `cartpole1`.

### What Happens During Training

1. The agent interacts with the environment using an epsilon-greedy policy
2. Experiences (state, action, reward, next_state, done) are stored in a replay buffer
3. Mini-batches are sampled from the replay buffer for training
4. The policy network is optimized using the DQN loss function
5. The target network is periodically synced with the policy network
6. Training metrics (rewards, epsilon decay) are logged and visualized
7. The best model is saved whenever a new highest episode reward is achieved

### Output Files

Training produces the following files in the `runs/` directory:

- `<env>_<timestamp>_model.pth`: Trained model weights
- `<env>_<timestamp>_graph.png`: Training progress visualization
- `<env>_<timestamp>.log`: Training log with milestone rewards

## Running a Trained Model

To run a trained agent (evaluation mode):

```bash
python Flappybird/agent.py <environment>
```

During evaluation:
- The agent uses a greedy policy (always selects the action with highest Q-value)
- The environment is rendered by default for visualization
- No training updates occur

## Example Commands

### Training FlappyBird for 1000+ Episodes

```bash
python Flappybird/agent.py flappybird --train
```

Monitor progress in `runs/` directory for real-time graphs and logs.

### Evaluating a Trained CartPole Agent

```bash
python Flappybird/agent.py cartpole1
```

### Viewing Rendered Agent Performance

When running in evaluation mode, the environment automatically renders, allowing you to watch the trained agent play.

## Project Structure

```
.
├── Flappybird/
│   ├── agent.py                 # Main training/evaluation script
│   ├── dqn.py                   # DQN neural network implementation
│   ├── experience_replay.py      # Replay buffer implementation
│   ├── hyperparameters.yml       # Configuration for different environments
│   └── __pycache__/             # Python cache files
├── runs/                         # Training outputs (models, logs, graphs)
│   ├── FlappyBird-v0_*.pth       # Trained model checkpoints
│   ├── FlappyBird-v0_*.log       # Training logs
│   └── FlappyBird-v0_*.png       # Training progress graphs
├── test.py                       # Testing utilities
├── tes.ipynb                     # Jupyter notebook for experimentation
└── README.md                     # This file
```

### Key Components

- **agent.py**: Contains the Agent class that handles training and evaluation loops, network optimization, and logging
- **dqn.py**: Defines the DQN neural network architecture (2-layer fully connected network)
- **experience_replay.py**: Implements the ReplayMemory class for storing and sampling transitions
- **hyperparameters.yml**: YAML configuration file containing hyperparameters for `flappybird` and `cartpole1` environments

## Future Improvements

- [ ] Double DQN (DDQN) for reduced overestimation of Q-values
- [ ] Dueling DQN architecture with advantage and value streams
- [ ] Prioritized Experience Replay for more efficient learning
- [ ] Support for additional environments (Atari games, etc.)
- [ ] Distributed training across multiple GPUs
- [ ] Interactive web UI for monitoring training in real-time
- [ ] Support for continuous action spaces with policy gradient methods
- [ ] Tensorboard integration for advanced monitoring

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Author**: Your Name  
**Last Updated**: March 2026
