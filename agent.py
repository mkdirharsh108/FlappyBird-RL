# https://github.com/markub3327/flappy-bird-gymnasium -> for more infor about the game  ->readme.md file

import argparse

import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMeamory
import itertools
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
import random
import os 



if torch.backends .mps.is_available():
    device ="mps"
elif torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"  # => runtime => change runtime => t4 group
    
RUNS_DIR = "runs"    
os.makedirs(RUNS_DIR, exist_ok = True)
    

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
        
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")
        
        
    def run(self, is_training = True, render = False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)
        
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        
        policy_dqn = DQN(num_states, num_actions).to(device)

        
        if is_training:
            meamory = ReplayMeamory(self.replay_meamory_size)
            epsilon = self.epsilon_init
            target_dqn = DQN(num_states, num_actions).to(device)
            # copy the wt & bias vals from policy
            target_dqn.load_state_dict(policy_dqn.state_dict())
            
            steps = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            best_reward = float("-inf")
            
        else:
            # best policy load = best model
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()
            

    
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            
            episode_rewards = 0
            terminated = False
            
            while (not terminated and episode_rewards < self.reward_threshold):
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()   # explore
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()  # exploit
            
                # Processing: terminated => done
                next_state, reward, terminated, _, _ = env.step(action.item())
                
                episode_rewards += reward
                
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                next_state = torch.tensor(next_state, dtype=torch.float, device=device)
                
                if is_training:
                    meamory.append((state, action,  next_state, reward, terminated))
                    steps+=1 
                
                state = next_state

            print(f"episodes={episode+1} with total rewards={episode_rewards} & epsilon={epsilon}")
            
            
            if is_training:
                # epsilon decay
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                
                if episode_rewards > best_reward:
                    log_msg = f"best reqard={episode_rewards} for episode={episode+1}"
                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")
                        
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward=episode_rewards 
                    
                
            if is_training and len(meamory) > self.mini_batch_size:
                # get sample
                mini_batch = meamory.sample(self.mini_batch_size)
                
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # sync the network
                if steps > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0
                    
            
        # env.close() - > to auto stop 
        
        
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # get batch of experiences
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch. stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch. stack(rewards)
        terminations = torch. tensor(terminations) . float().to(device)

        # calculate target Q-values - if terminations=true => zero
        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.gamma * target_dqn(next_states).max(dim=1) [0]

        # calculate y_pred i.e. Q-value from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # compute loss  
        loss = self. loss_fn(current_q, target_q)

        # optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
if __name__ == "__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='Name of the parameter set in YAML')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql. run(is_training=True)
    else:
        dql. run(is_training=False, render=True) 