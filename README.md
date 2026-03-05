# FlappyBird & CartPlay Reinforcement Learning Agent

This project contains reinforcement learning agents that can be trained to play FlappyBird and CartPlay using `agent.py`.

## Project Structure

Flappybird/
│
├── agent.py
├── models/
├── environments/
└── README.md


## Requirements

Make sure you have Python 3.8+ installed.

Install dependencies:

pip install -r requirements.txt

If you do not have a requirements file, you can install common dependencies manually:

pip install numpy pygame gym torch


## Running the Agent

Run `agent.py` from the Flappybird folder.

cd Flappybird


--------------------------------
Cartplay
--------------------------------

Train the model:

python Flabbybird/agent.py cartplay --train

Run the trained model:

python Flabbybird/agent.py cartplay


--------------------------------
Flappybird
--------------------------------

Train the model:

python Flappybird/agent.py flappybird --train

Run the trained model:

python Flabbybird/agent.py flappybird


## Training

Training mode allows the agent to learn by interacting with the environment.

--train

Example:

python Flabbybird/agent.py cartplay --train


## Models

Trained models are saved automatically and reused when running the agent without the --train flag.


## Example Workflow

1. Train the agent

python Flabbybird/agent.py flappybird --train

2. Run the trained agent

python Flabbybird/agent.py flappybird


## License

This project is open source and available under the MIT License.
