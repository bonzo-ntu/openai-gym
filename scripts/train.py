from utils import fix
from agents import (
    PolicyGradientNetwork,
    PolicyGradientAgent
)
from typing import Dict
from tqdm.notebook import tqdm
import wandb
import numpy as np
import gymnasium as gym
from gym.wrappers import RecordVideo
from IPython import display
import torch
import torch.nn as nn
from utils import avg
import matplotlib.pyplot as plt


class trainer:
    def __init__(self, agent:nn.Module, config: Dict, seed: int=543):
        """
        Example of config:
            config={
                "agent": "PolicyGradientAgent",
                "phase": "train",
                "project": "lunalander",
                "reward": "discounted_future_reward",
                "learning_rate": 1e-3,
                "gamma": 0.99,
                "batch": 1,
                "episode": 5,
                "model_name": "",
                "model_path": "",
                "greedy": True,
                "info": "network 8x10x10x4"
            }
        """
        self.agent = agent
        self.config = config
        self.seed = seed
        match config["project"]:
            case "lunalander":
                self.env = gym.make('LunarLander-v2', render_mode='rgb_array')
                
        fix(self.env, seed)

    def model_name(self):
        c = self.config # 重新命名
        match self.config["agent"]:
            case "PolicyGradientAgent":
                agent = "PG"
        self.config["model_name"] = f'{agent} {c["info"]} lr={c["learning_rate"]} reward={c["reward"]} gamma={c["gamma"]} batch={c["batch"]} episode={c["episode"]}'
        self.config["model_path"] = f'./models/{self.config["model_name"]}.pt'
        return self.config["model_path"]
    
    def greedy_str(self):
        return "greedy" if self.config["greedy"] else "sample"

    def train(self, reward_func, wandb_log: bool=True, verbose: bool=False):
        self.config["phase"] = "train"

        if wandb_log:
            wandb.init(
                # set the wandb project where this run will be logged
                project = self.config['project'], 
                # track hyperparameters and run metadata
                config = self.config
            )

        self.agent.network.train()  # Switch network into training mode 
        

        gamma = self.config["gamma"]
        num_batch, episode_per_batch = self.config['batch'], self.config['episode']
        avg_total_rewards, avg_final_rewards = [], []

        prg_bar = tqdm(range(num_batch))
        for batch in prg_bar:

            log_probs, rewards = [], []
            total_rewards, final_rewards = [], []
            total_steps = []
            # collect trajectory
            for episode in range(episode_per_batch):
                
                state, info = self.env.reset(seed=self.seed)
                total_reward, total_step = 0, 0
                seq_rewards = []
                terminated, truncated = False, False
                
                while not (terminated or truncated):
                    action, log_prob = self.agent.sample(state)  # at, log(at|st)
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    

                    log_probs.append(log_prob)  # [log(a1|s1), log(a2|s2), ...., log(at|st)]
                    state = next_state
                    total_reward += reward
                    rewards.append(reward)  # 改這裡
                    seq_rewards.append(reward)
                    # ! 重要 ！
                    # 現在的reward 的implementation 為每個時刻的瞬時reward, 給定action_list : a1, a2, a3 ......
                    #                                                       reward :     r1, r2 ,r3 ......
                    # medium：將reward調整成accumulative decaying reward, 給定action_list : a1,                         a2,                           a3 ......
                    #                                                       reward :     r1+0.99*r2+0.99^2*r3+......, r2+0.99*r3+0.99^2*r4+...... ,r3+0.99*r4+0.99^2*r5+ ......
                    # boss : implement DQN

                else:
                    final_rewards.append(reward)
                    total_rewards.append(total_reward)
                    total_steps.append(len(seq_rewards))

                    rewards = reward_func(seq_rewards, rewards, gamma)
                    # n = len(seq_rewards)

                    # for i in range(2, n+1):
                    #     seq_rewards[-i] += gamma * (seq_rewards[-i+1])
                    # rewards[-len(seq_rewards):] = seq_rewards           
                

            # 紀錄訓練過程
            avg_total_reward = avg(total_rewards)
            avg_final_reward = avg(final_rewards)
            avg_episode_steps = avg(total_steps)
            avg_total_rewards.append(avg_total_reward)
            avg_final_rewards.append(avg_final_reward)
            prg_bar.set_description(f"\r Total reward: {avg_total_reward: 4.1f}, Final reward: {avg_final_reward: 4.1f}")

            # 更新網路
            rewards_mean, rewards_std = np.mean(rewards), np.std(rewards)
            rewards = (rewards - rewards_mean) / (rewards_std + 1e-9)  # 將 reward 正規標準化
            self.agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))
            
            if verbose:
                print("logs prob looks like ", torch.stack(log_probs).size())
                print("torch.from_numpy(rewards) looks like ", torch.from_numpy(rewards).size())

            if wandb_log:
                wandb.log({"mean rewards":rewards_mean, 
                           "std reward":rewards_std, 
                           "avg_total_reward": avg_total_reward, 
                           "avg_final_reward": avg_final_reward, 
                           "avg_episode_steps":avg_episode_steps}
                           )

        self.agent.save(self.model_name())

        return avg_total_reward, avg_final_reward, avg_episode_steps

    def test(self, reward_func, wandb_log: bool=True, verbose: bool=False, record_video: bool=True):
        self.config["phase"] = "test"
        if wandb_log:
            wandb.init(
                # set the wandb project where this run will be logged
                project = self.config['project'], 
                # track hyperparameters and run metadata
                config = self.config
            )

        
        self.agent.load(self.config["model_path"])
        self.agent.network.eval()  # Switch network into eval mode 
        

        if not record_video:
            img = plt.imshow(self.env.render())
        
        gamma = self.config["gamma"]
        greedy = self.config["greedy"]
        rewards = []
        total_rewards, final_rewards = [], []
        total_steps = []
        for episode in range(self.config["episode"]):
            print(episode)
            if not record_video:
                env = self.env
            else:
                trigger = lambda t: t%10 == 0
                env = RecordVideo(self.env, 
                            video_folder=f'./videos/{self.config["model_name"]} {self.greedy_str()} {episode}', 
                            episode_trigger=trigger)
                
            state, info = env.reset(seed=self.seed)
            seq_rewards = []
            terminated, truncated = False, False
            total_reward = 0
            while not (terminated or truncated):
                action, log_prob = self.agent.sample(state, greedy)
                next_state, reward, terminated, truncated, info = env.step(action)

                state = next_state
                total_reward += reward
                rewards.append(reward)  # 改這裡
                seq_rewards.append(reward)
                
                if verbose:
                    print(f'reward: {reward}, total_reward:{total_reward}, terminated:{terminated}, truncated:{truncated}')

                if not record_video:
                    img.set_data(self.env.render())
                    display.display(plt.gcf())
                    display.clear_output(wait=True)
            else:
                final_rewards.append(reward)
                total_rewards.append(total_reward)
                total_steps.append(len(seq_rewards))
                rewards = reward_func(seq_rewards, rewards, gamma)

            avg_total_reward = avg(total_rewards)
            avg_final_reward = avg(final_rewards)
            avg_episode_steps = avg(total_steps)
            rewards_mean, rewards_std = np.mean(rewards), np.std(rewards)
            rewards = (rewards - rewards_mean) / (rewards_std + 1e-9)  # 將 reward 正規標準化

            if verbose:
                print(f'reward: {reward}, total_reward:{total_reward}, terminated:{terminated}, truncated:{truncated}')
            
            if wandb_log:
                wandb.log({"mean rewards":rewards_mean, 
                           "std reward":rewards_std, 
                           "avg_total_reward": avg_total_reward, 
                           "avg_final_reward": avg_final_reward, 
                           "avg_episode_steps":avg_episode_steps}
                           )

            return total_rewards, final_rewards, total_steps