#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 2: PPO-Clip
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import random
from collections import deque
from typing import Deque, List, Tuple

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
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Mish(),
            nn.Linear(128, 128),
            nn.Mish()
        )
        self.fc_mean = nn.Linear(128, out_dim)
        self.fc_log_std = nn.Linear(128, out_dim)

        initialize_uniformly(self.model)
        initialize_uniformly(self.fc_mean)
        initialize_uniformly(self.fc_log_std)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        X = self.model(state)
        mean = self.fc_mean(X)
        log_std = self.fc_log_std(X)
        LOG_STD_MIN = -20  # Or -10, -5, a common choice for stability
        LOG_STD_MAX = 2    # Or 0,  a common choice for stability
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) # 非常重要！
        std = torch.exp(log_std) + 1e-5 # 1e-5 也可以考慮調整，例如 1e-6
        # std = torch.exp(log_std) + 1e-5
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.tanh(action) * 2.0
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Mish(),
            nn.Linear(128, 128),
            nn.Mish(),
            nn.Linear(128, 1),
        )
        initialize_uniformly(self.model)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        value = self.model(state)
        #############################

        return value
    
def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float) -> List:
    """Compute gae."""

    ############TODO#############
    gae_returns = []
    gae = 0
    
    # 從後向前計算GAE
    for r, m, v in zip(reversed(rewards), reversed(masks), reversed(values)):
        # 計算TD誤差
        delta = r + gamma * next_value * m - v
        
        # 更新GAE
        gae = delta + gamma * tau * m * gae
        
        # 將GAE插入到列表開頭
        gae_returns.insert(0, gae)
        
        # 更新next_value為當前值
        next_value = v
    #############################
    return gae_returns

# PPO updates the model several times(update_epoch) using the stacked memory. 
# By ppo_iter function, it can yield the samples of stacked memory by interacting a environment.
def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Get mini-batches."""
    batch_size = states.size(0)
    for _ in range(update_epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], values[rand_ids], log_probs[
                rand_ids
            ], returns[rand_ids], advantages[rand_ids]

class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        update_epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        # self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch
        self.max_env_step = args.max_env_step
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        obs_dim = env.observation_space.shape[0]
        self.obs_dim = obs_dim
        action_dim = env.action_space.shape[0]
        self.action_dim = action_dim
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.005)

        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state_tensor = torch.FloatTensor(state).to(self.device)
        action_from_actor, dist = self.actor(state_tensor)
        selected_action_tensor = dist.mean if self.is_test else action_from_actor

        if not self.is_test:
            value = self.critic(state_tensor)
            self.states.append(state_tensor)
            self.actions.append(selected_action_tensor)
            self.values.append(value)
            self.log_probs.append(dist.log_prob(selected_action_tensor))

        final_selected_action_numpy = selected_action_tensor.cpu().detach().numpy()
        
        if not self.is_test: # Log during training
            # For Pendulum-v1: state is (1,3), action is (1,1)
            wandb.log({
                "action_histogram": wandb.Histogram(final_selected_action_numpy.flatten()),
                "state_cos_theta_histogram": wandb.Histogram(state[:,0]),
                "state_sin_theta_histogram": wandb.Histogram(state[:,1]),
                "state_angular_velocity_histogram": wandb.Histogram(state[:,2]),
                "global_step": self.total_step
            })
            
        return final_selected_action_numpy

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float32, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, (1, -1)).astype(np.float32)
        reward = np.reshape(reward, (1, -1)).astype(np.float32)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1 - done).to(self.device))

        return next_state, reward, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state)

        advantages_gae = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.cat(self.states).view(-1, self.obs_dim)
        actions = torch.cat(self.actions)
        advantages_gae_raw = torch.cat(advantages_gae).detach()
        advantages_for_actor = (advantages_gae_raw - advantages_gae_raw.mean()) / (advantages_gae_raw.std() + 1e-8)
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        td_targets = advantages_gae_raw + values

        actor_losses, critic_losses = [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(
            update_epoch=self.update_epoch,
            mini_batch_size=self.batch_size,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            returns=td_targets,
            advantages=advantages_for_actor,
        ):
            # calculate ratios
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            # actor_loss
            ############TODO#############
            # actor_loss = ?
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
            policy_loss_elementwise = -torch.min(surr1, surr2) # 形狀: [mini_batch_size]

            # 更新的熵計算和 actor_loss 計算
            entropy = dist.entropy() 
            if entropy.ndim > 1 and self.action_dim > 1: 
                entropy_bonus = entropy.sum(dim=-1).mean() 
            else:
                entropy_bonus = entropy.mean()
            
            actor_loss = policy_loss_elementwise.mean() - self.entropy_weight * entropy_bonus # 確保 actor_loss 是純量
            #############################

            # critic_loss
            ############TODO#############
            # critic_loss = ?
            current_values = self.critic(state)
            critic_loss = F.mse_loss(return_, current_values)

            #############################
            
            # train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)

        return actor_loss, critic_loss

    def train(self):
        """Train the PPO agent."""
        self.is_test = False

        state, _ = self.env.reset(seed=self.seed)
        state = np.expand_dims(state, axis=0)

        actor_losses, critic_losses = [], []
        scores = []
        score = 0
        episode_count = 0
        # for ep in tqdm(range(1, self.num_episodes)):
        pbar = tqdm(total=self.max_env_step, desc="Training Progress")
        while self.total_step < self.max_env_step:
            score = 0
            for _ in range(self.rollout_len):
                self.total_step += 1
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward[0][0]

                # if episode ends
                if done[0][0]:
                    scores.append(score)
                    episode_count += 1
                    wandb.log({
                        "episode_score": score,
                        "episode": episode_count,
                        "global_step": self.total_step
                    })
                    state, _ = self.env.reset(seed=self.seed)
                    state = np.expand_dims(state, axis=0)
                    pbar.set_postfix({
                        'Episode': episode_count,
                        'Total Reward': f'{score:.2f}',
                    })
                    score = 0

            actor_loss, critic_loss = self.update_model(next_state)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            wandb.log({
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "global_step": self.total_step
            })
            pbar.update(self.rollout_len)

        pbar.close()
        # termination
        self.env.close()

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
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-ppo-run")
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--max-env-step", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=int, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=int, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=2000)  
    parser.add_argument("--update-epoch", type=float, default=64)
    args = parser.parse_args()
 
    # environment
    env = gym.make("Pendulum-v1")#, render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(project="DLP-Lab7-PPO-Pendulum", name=args.wandb_run_name, save_code=True)
    
    agent = PPOAgent(env, args)
    agent.train()