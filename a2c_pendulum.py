#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh


import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm
from typing import Tuple
import os
import yaml
import datetime
import json
import csv

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim1=128, hidden_dim2=64):
        """Initialize."""
        super(Actor, self).__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        self.fc1 = nn.Linear(in_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.mu = nn.Linear(hidden_dim2, out_dim)
        self.std = nn.Linear(hidden_dim2, out_dim)
        
        # Initialize weights
        initialize_uniformly(self.fc1)
        initialize_uniformly(self.fc2)
        initialize_uniformly(self.mu)
        initialize_uniformly(self.std)
        #############################
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        ############TODO#############
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mu = self.mu(x)
        std = F.softplus(self.std(x)) + 1e-3  # 確保標準差為正值
        
        dist = Normal(mu, std)
        action = dist.sample()
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden_dim1=128, hidden_dim2=64):
        """Initialize."""
        super(Critic, self).__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        self.fc1 = nn.Linear(in_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        
        # Initialize weights
        initialize_uniformly(self.fc1)
        initialize_uniformly(self.fc2)
        initialize_uniformly(self.fc3)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        #############################

        return value
    

class A2CAgent:
    """A2CAgent interacting with environment.

    Atribute:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args=None, is_test=False):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = getattr(args, 'num_episodes', 1000)
        
        # 設置網絡結構參數（如果有提供）
        self.actor_hidden_dim1 = getattr(args, 'actor_hidden_dim1', 128)
        self.actor_hidden_dim2 = getattr(args, 'actor_hidden_dim2', 64)
        self.critic_hidden_dim1 = getattr(args, 'critic_hidden_dim1', 128)
        self.critic_hidden_dim2 = getattr(args, 'critic_hidden_dim2', 64)
        
        # 設置保存相關參數
        self.save_per_epoch = getattr(args, 'save_per_epoch', 100)  # 每隔多少個epoch保存一次
        self.result_dir = getattr(args, 'result_dir', 'result-a2c_pendulum')
        
        # 創建實驗資料夾 (僅在非測試模式下)
        self.is_test_mode = is_test
        if not self.is_test_mode:
            timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.exp_dir = os.path.join(self.result_dir, f"exp_{timestr}")
            if not os.path.exists(self.exp_dir):
                os.makedirs(self.exp_dir, exist_ok=True)
            
            # 保存配置
            self.save_config(args)
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(
            obs_dim, 
            action_dim, 
            hidden_dim1=self.actor_hidden_dim1, 
            hidden_dim2=self.actor_hidden_dim2
        ).to(self.device)
        self.critic = Critic(
            obs_dim, 
            hidden_dim1=self.critic_hidden_dim1, 
            hidden_dim2=self.critic_hidden_dim2
        ).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            self.transition = [state, log_prob]

        return selected_action.clamp(-2.0, 2.0).cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition.extend([next_state, reward, done])

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        state, log_prob, next_state, reward, done = self.transition

        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        mask = 1 - done
        
        ############TODO#############
        # value_loss = ?
        next_state = torch.FloatTensor(next_state).to(self.device)
        
        # 計算當前狀態的值函數
        value = self.critic(state)
        
        # 計算目標值 (TD target)
        next_value = self.critic(next_state)
        target = reward + self.gamma * next_value * mask
        
        # MSE 損失函數計算值損失
        value_loss = F.mse_loss(value, target.detach())
        #############################

        # update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # advantage = Q_t - V(s_t)
        ############TODO#############
        # policy_loss = ?
        advantage = (target - value).detach()
        
        # 計算策略損失
        policy_loss = -log_prob * advantage
        
        # 如果使用熵正則化
        if self.entropy_weight != 0:
            _, dist = self.actor(state)
            entropy = dist.entropy().mean()
            policy_loss = policy_loss - self.entropy_weight * entropy
        #############################
        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return policy_loss.item(), value_loss.item()

    def train(self):
        """Train the agent."""
        self.is_test = False
        step_count = 0
        best_score = -float('inf')
        
        # 創建結果記錄文件
        result_file = os.path.join(self.exp_dir, 'training_results.csv')
        with open(result_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Score', 'Actor_Loss', 'Critic_Loss'])
        
        for ep in tqdm(range(1, int(self.num_episodes) + 1)): 
            actor_losses, critic_losses, scores = [], [], []
            state, _ = self.env.reset(seed=self.seed)
            score = 0
            done = False
            while not done:
                # 移除 render 以加速訓練
                # self.env.render()
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                state = next_state
                score += reward
                step_count += 1
                # W&B logging
                wandb.log({
                    "step": step_count,
                    "actor loss": actor_loss,
                    "critic loss": critic_loss,
                    }) 
                # if episode ends
                if done:
                    scores.append(score)
                    avg_actor_loss = np.mean(actor_losses)
                    avg_critic_loss = np.mean(critic_losses)
                    print(f"Episode {ep}: Total Reward = {score}")
                    
                    # 記錄結果
                    with open(result_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([ep, score, avg_actor_loss, avg_critic_loss])
                    
                    # W&B logging
                    wandb.log({
                        "episode": ep,
                        "return": score,
                        "avg_actor_loss": avg_actor_loss,
                        "avg_critic_loss": avg_critic_loss
                        })
                    
                    # 保存檢查點
                    if ep % self.save_per_epoch == 0:
                        self.save_checkpoint(ep)
                    
                    # 保存最佳模型
                    if score > best_score and not self.is_test_mode:
                        best_score = score
                        checkpoint = {
                            'actor_state_dict': self.actor.state_dict(),
                            'critic_state_dict': self.critic.state_dict(),
                            'actor_optimizer': self.actor_optimizer.state_dict(),
                            'critic_optimizer': self.critic_optimizer.state_dict(),
                            'episode': ep,
                            'score': best_score
                        }
                        torch.save(checkpoint, os.path.join(self.exp_dir, 'best_model.pt'))
                        print(f"Saved best model with score {best_score} at episode {ep}")
        
        # 訓練結束後保存最終模型
        if not self.is_test_mode:
            self.save_checkpoint(int(self.num_episodes))

    def load_checkpoint(self, checkpoint_path):
        """加載模型檢查點"""
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加載模型參數
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # 加載優化器參數（可選）
        if 'actor_optimizer' in checkpoint and 'critic_optimizer' in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            
        episode = checkpoint.get('episode', 0)
        score = checkpoint.get('score', None)
        
        print(f"Loaded checkpoint from episode {episode}" + 
              (f" with score {score}" if score is not None else ""))
        
        return episode, score

    def test(self, video_folder=None, checkpoint_path=None, num_episodes=1):
        """Test the agent."""
        self.is_test = True
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        if video_folder:
            # 如果沒有提供視頻文件夾，創建一個
            if not os.path.exists(video_folder):
                os.makedirs(video_folder, exist_ok=True)
                
            tmp_env = self.env
            self.env = gym.wrappers.RecordVideo(
                self.env, 
                video_folder=video_folder,
                episode_trigger=lambda x: True  # 錄制所有episode
            )
        
        scores = []
        for ep in range(num_episodes):
            state, _ = self.env.reset(seed=self.seed + ep)  # 使用不同的種子
            done = False
            score = 0
            steps = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward
                steps += 1

            scores.append(score)
            print(f"Test Episode {ep+1}/{num_episodes}: Score = {score}, Steps = {steps}")
        
        # 保存結果到CSV
        if video_folder:
            result_file = os.path.join(video_folder, 'test_results.csv')
            with open(result_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Score'])
                for i, score in enumerate(scores):
                    writer.writerow([i+1, score])
                    
            # 保存統計摘要
            summary_file = os.path.join(video_folder, 'test_summary.json')
            summary = {
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
                'median_score': float(np.median(scores)),
                'num_episodes': num_episodes
            }
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4)
            
            print(f"Test Summary: Mean Score = {summary['mean_score']:.2f} ± {summary['std_score']:.2f}")
            
            # 關閉記錄環境
            self.env.close()
            self.env = tmp_env
        
        return scores

    def save_config(self, args):
        """保存配置到 YAML 文件"""
        if self.is_test_mode:
            return  # 在測試模式下跳過保存配置
            
        config = {
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'discount_factor': self.gamma,
            'entropy_weight': self.entropy_weight,
            'seed': self.seed,
            'num_episodes': self.num_episodes,
            'actor_hidden_dim1': self.actor_hidden_dim1,
            'actor_hidden_dim2': self.actor_hidden_dim2,
            'critic_hidden_dim1': self.critic_hidden_dim1,
            'critic_hidden_dim2': self.critic_hidden_dim2,
            'save_per_epoch': self.save_per_epoch,
        }
        
        with open(os.path.join(self.exp_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
    def save_checkpoint(self, episode):
        """保存模型檢查點"""
        if self.is_test_mode:
            return  # 在測試模式下跳過保存檢查點
            
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'episode': episode,
        }
        checkpoint_path = os.path.join(self.exp_dir, f'checkpoint_ep{episode}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint at episode {episode} to {checkpoint_path}")

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-a2c-run")
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=float, default=5000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=1e-2) # entropy can be disabled by setting this to 0
    
    # 保存相關參數
    parser.add_argument("--save-per-epoch", type=int, default=100, help="保存頻率（每多少個回合保存一次）")
    parser.add_argument("--result-dir", type=str, default="result-a2c_pendulum", help="結果保存的基礎目錄")
    
    # 網絡相關參數
    parser.add_argument("--actor-hidden-dim1", type=int, default=128)
    parser.add_argument("--actor-hidden-dim2", type=int, default=64)
    parser.add_argument("--critic-hidden-dim1", type=int, default=128)
    parser.add_argument("--critic-hidden-dim2", type=int, default=64)
    
    args = parser.parse_args()
    
    # environment
    env = gym.make("Pendulum-v1")#, render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    
    # 僅在直接運行此檔案時初始化 wandb，sweep 時會由 sweep_a2c.py 初始化
    if not wandb.run:
        wandb.init(project="DLP-Lab7-A2C-Pendulum", name=args.wandb_run_name, save_code=True)
    
    agent = A2CAgent(env, args)
    agent.train()