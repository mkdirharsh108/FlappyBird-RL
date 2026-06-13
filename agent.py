# https://github.com/markub3327/flappy-bird-gymnasium -> for more infor about the game 

import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMeamory
import itertools
import torch
import yaml
import torch.nn as nn
import torch.optim as optim




if torch.backends .mps.is_available():
    device ="mps"
elif torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"  # => runtime => change runtime => t4 group
    

class Agent():
    
    def __init__(self, param_set):
        self.param_set  = param_set
        
        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]
            
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        
        self.epsilon_init= params["epsilon_init"]
        self.epsilon_min= params["epsilon_min"]
        self.epsilon_decay= params["epsilon_decay"]
        
        self.replay_meamory_size= params["replay_meamory_size"]
        self.mini_batch_size= params["mini_batch_size"]
        
        self.reward_threshold= params["reward_threshold"]
        self.network_sync_rate= params["network_sync_rate"]
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        
        
    def run(self, is_training = True, render = False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)
        
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        
        policy_dqn = DQN(num_states, num_actions).to(device)

        
        if is_training:
            meamory = ReplayMeamory(self.replay_meamory_size)

        for episode in itertools.cout():
            state, _ = env.reset()
            episode_rewards = 0
            terminated = False
            
            while not terminated:
                action = env.action_space.sample()

                # Processing: terminated => done
                next_state, reward, terminated, _, _ = env.step(action)
                
                if is_training:
                    meamory.append((state, action,  new_state, reward, terminated))
                
                state = new_state
                episode_rewards += reward

            print(f"episodes={episode+1} with total rewards={episode_rewards}")
        # env.close() - > to auto stop 