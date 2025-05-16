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

def initialize_uniformly(layer, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    if isinstance(layer, nn.Linear):
        layer.weight.data.uniform_(-init_w, init_w)
        layer.bias.data.uniform_(-init_w, init_w)
    elif isinstance(layer, nn.Sequential):
        for module in layer:
            if isinstance(module, nn.Linear):
                module.weight.data.uniform_(-init_w, init_w)
                module.bias.data.uniform_(-init_w, init_w)
            elif isinstance(module, nn.Sequential):
                initialize_uniformly(module, init_w)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize."""
        super(Actor, self).__init__()
   
        
        ############TODO#############
        # Remeber to initialize the layer weights
        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.mu_layer = nn.Linear(64, out_dim)
        self.std_layer = nn.Linear(64, out_dim)
        initialize_uniformly(self.model)
        initialize_uniformly(self.mu_layer)
        initialize_uniformly(self.std_layer)
        
        #############################
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        ############TODO#############
        x = self.model(state)
        mu = self.mu_layer(x)
        # 使用softplus或exp確保標準差為正值
        std = F.softplus(self.std_layer(x)) + 1e-5  # 加上一個小值避免為零
        dist = Normal(mu, std)
        action = dist.sample()
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        initialize_uniformly(self.model)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = self.model(state)
        #############################

        return x
    
def discounted_rewards(rewards, dones, gamma):
    ret = 0
    discounted = []
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + ret * gamma * (1-done)
        discounted.append(ret)
    
    return discounted[::-1]


# state, log_prob, next_state, reward, done = process_memory(self.transition, self.gamma)
def process_memory(memory, gamma=0.99, device="cpu"):
    states = []
    log_probs = []
    next_states = []
    rewards = []
    dones = []


    for it in memory:
        state, log_prob, next_state, reward, done = it
        log_probs.append(log_prob)
        rewards.append(reward)
        states.append(state)
        next_states.append(next_state)
        dones.append(done)
    

    rewards = discounted_rewards(rewards, dones, gamma)
    log_probs = torch.stack(log_probs).view(-1, 1).to(device)
    states = torch.stack(states).to(device)
    next_states_arr    = np.stack(next_states, axis=0)  # shape = (16, 3)
    next_states_tensor = torch.from_numpy(next_states_arr).to(device)

    rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1).to(device)
    dones = torch.tensor(dones).view(-1, 1).to(device)
    return states, log_probs, next_states_tensor, rewards, dones

def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)


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

    def __init__(self, env: gym.Env, args=None):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes
        self.memory = []
        self.steps_on_memory = args.steps_on_memory
        self.max_grad_norm = args.max_grad_norm
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        print("obs_dim: ", obs_dim)
        print("action_dim: ", action_dim)
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

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
            self.memory.append(self.transition)
            self.transition = []

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        # state, log_prob, next_state, reward, done = self.transition
        # actions, rewards, states, next_states, dones = process_memory(self.transition, self.gamma)
        state, log_prob, next_state, reward, done = process_memory(self.memory, self.gamma, self.device)
        td_target = reward
        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        # mask = 1 - done
        
        ############TODO#############
        value = self.critic(state)
        value_loss = F.mse_loss(td_target, value)
        #############################

        # update value
        self.critic_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic_optimizer, self.max_grad_norm)
        self.critic_optimizer.step()
        


        # advantage = Q_t - V(s_t)
        ############TODO#############
        advantage = td_target - value.detach()  # 確保值不再參與梯度計算
        _, norm_dists = self.actor(state)
        entropy = norm_dists.entropy().mean()
        policy_loss = (-log_prob*advantage).mean() - entropy*self.entropy_weight

        #############################
        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        clip_grad_norm_(self.actor_optimizer, self.max_grad_norm)
        self.actor_optimizer.step()

        # 更新後清空記憶庫
        self.memory = []
        
        return policy_loss.item(), value_loss.item()

    def train(self):
        """Train the agent."""
        self.is_test = False
        step_count = 0
        
        for ep in tqdm(range(1, self.num_episodes)): 
            actor_losses, critic_losses, scores = [], [], []
            state, _ = self.env.reset(seed=self.seed)
            score = 0
            done = False
            while not done:
                # 移除渲染以提高訓練速度
                # self.env.render()  # 註解掉這行來提高速度
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                
                # 移除這行，因為在step方法中已經添加到memory
                # self.transition.extend([next_state, reward, done])
                
                if len(self.memory) < self.steps_on_memory and not done:
                    state = next_state
                    score += reward
                    continue

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
                    if ep % 100 == 0:
                        print(f"Episode {ep}: Total Reward = {score}")
                    # W&B logging
                    wandb.log({
                        "episode": ep,
                        "return": score
                        })

    def test(self, video_folder: str):
        """Test the agent."""
        self.is_test = True

        tmp_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        self.env = tmp_env

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
    parser.add_argument("--num-episodes", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=int, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--steps-on-memory", type=int, default=16)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    args = parser.parse_args()
    
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(project="DLP-Lab7-A2C-Pendulum", name=args.wandb_run_name, save_code=True)
    
    agent = A2CAgent(env, args)
    agent.train()