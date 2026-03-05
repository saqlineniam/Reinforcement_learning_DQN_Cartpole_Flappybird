import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import torch
from torch import nn
import yaml

from experience_replay import ReplayMemory
from dqn import DQN

import itertools
import argparse
from datetime import datetime, timedelta


import flappy_bird_gymnasium
import gymnasium as gym
import os

DATE_FORMAT = "%m-%d %H-%M-%S"

#directory to save training results
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok = True)

#agg to generate plots as image and save them to disk instead of displaying them interactively
matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class Agent:
    def __init__(self, hyperparameter_set):
        with open("Flappybird/hyperparameters.yml", "r") as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.env_id = hyperparameters["env_id"]
        self.learning_rate_a = hyperparameters["learning_rate_a"]
        self.discount_factor_g = hyperparameters["discount_factor_g"]
        self.network_sync_rate = hyperparameters["network_sync_rate"]
        self.replay_memory_size = hyperparameters["replay_memory_size"]
        self.mini_batch_size = hyperparameters["mini_batch_size"]
        self.epsilon_init = hyperparameters["epsilon_init"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.epsilon_min = hyperparameters["epsilon_min"]
        self.stop_on_reward = hyperparameters["stop_on_reward"]
        self.fc1_nodes = hyperparameters["fc1_nodes"]
        self.env_make_params = hyperparameters.get("env_make_params") or {}


        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        # Path to save training results
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.env_id}_{datetime.now().strftime(DATE_FORMAT)}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.env_id}_{datetime.now().strftime(DATE_FORMAT)}_model.pth")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{self.env_id}_{datetime.now().strftime(DATE_FORMAT)}_graph.png")


    def _find_latest_model(self):
        """Find the most recently saved model for this environment."""
        import glob
        models = glob.glob(os.path.join(RUNS_DIR, f"{self.env_id}*_model.pth"))
        if not models:
            return None
        return max(models, key=os.path.getctime)

    def run(self, is_training=True, render=False):
        # During evaluation, try to load the most recent model
        if not is_training:
            latest_model = self._find_latest_model()
            if latest_model:
                self.MODEL_FILE = latest_model
                print(f"Loading model: {latest_model}")
            else:
                print(f"No saved model found for {self.env_id}")
                return

        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"Training started at {start_time.strftime(DATE_FORMAT)} with hyperparameters: {self.__dict__}"
            print(log_message)
            with open(self.LOG_FILE, "a") as log_file:
                log_file.write(log_message + "\n")
        
        
        #env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
        env = gym.make(self.env_id, render_mode="human" if render else None, **self.env_make_params)

        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]
        reward_per_episode = []


        # create policy and target newtworks
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training:
            # initiate epsilon
            epsilon = self.epsilon_init

            # initiate replay memory
            memory = ReplayMemory(self.replay_memory_size)

            # create the target DQN and copy the weights from the policy DQN         
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # network policy optimizer
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr = self.learning_rate_a)

            # list to keep track of epsilon decay
            epsilon_history = []

            # trak number of step taken
            step_count = 0

            # track best reward
            best_reward = -9999999
        else:
            # load learned policy
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            # switch to evaluation mode
            policy_dqn.eval()


        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype = torch.float, device = device)

            terminated = False
            episode_reward = 0.0


            while (not terminated and episode_reward < self.stop_on_reward):

                # select action using epsilon-greedy policy
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype = torch.long, device = device)
                else:
                    # select best action according to policy DQN
                    with torch.no_grad():
                        # the state is unsqueezed to add a batch dimension before being passed through the network. The output is then squeezed to remove the batch dimension, and argmax is used to select the action with the highest Q-value.
                        action = policy_dqn(state.unsqueeze(dim = 0)).squeeze().argmax()

                # Execute the action in the environment
                new_state, reward, terminated, _, info = env.step(action.item())

                # accumulate rewards
                episode_reward += reward

                # convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype = torch.float, device = device)
                reward = torch.tensor(reward, dtype = torch.float, device = device)


                if is_training:
                    # Store the transition in the replay memory
                    memory.append((state, action, new_state, reward, terminated))

                    # Increment step count
                    step_count += 1
                
                # move to the next state
                state = new_state
            
            # keep track of reward per episode
            reward_per_episode.append(episode_reward)
            
            # save model when new best reward is obtained    
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"New best reward: {episode_reward:.2f} at episode {episode}"
                    print(log_message)
                    with open(self.LOG_FILE, "a") as log_file:
                        log_file.write(log_message + "\n")

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                # update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(reward_per_episode, epsilon_history)
                    last_graph_update_time = current_time
                    print(f"Graph saved at episode {episode}")
                
                # if enough experience has been collected, start optimizing the policy network
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # copy policy network to target network every N steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0


    def save_graph(self, reward_per_episode, epsilon_history):
        # create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # plot average rewards (y axis) vs episodes (X axis)
        mean_rewards = np.zeros(len(reward_per_episode))
        for x in range(len(reward_per_episode)):
            mean_rewards[x] = np.mean(reward_per_episode[max(0, x-99):x+1])
        
        ax1.plot(mean_rewards, label='Mean Reward (100 eps)')
        ax1.set_ylabel("Mean Reward")
        ax1.set_xlabel("Episode")
        ax1.grid(True)
        ax1.legend()

        # plot epsilon decay
        if len(epsilon_history) > 0:
            ax2.plot(epsilon_history, label='Epsilon')
        ax2.set_ylabel("Epsilon")
        ax2.set_xlabel("Step")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()

        # save plots
        fig.savefig(self.GRAPH_FILE, dpi=100)
        plt.close(fig)


    # optimize policy nrework
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim = 1)[0]
            '''The target Q-value is calculated using the Bellman equation.
            The term (1-terminations) ensures that if the episode has terminated, the future reward is not considered. The target DQN is used to compute the maximum Q-value for the next
            '''
        
        current_q = policy_dqn(states).gather(1, actions.unsqueeze(dim = 1)).squeeze()


        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate DQN agent")
    parser.add_argument('hyperparameters', help='Hyperparameter set name')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--render', action='store_true', help='Render environment during eval')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)




