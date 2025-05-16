#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# PPO Pendulum Sweep: 超參數優化搜索
no = 14

import random
from collections import deque
from typing import Deque, List, Tuple, Dict, Any, Optional

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import os
from tqdm import tqdm
import csv
import glob


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


def mish(input):
    """Mish激活函數"""
    return input * torch.tanh(F.softplus(input))


class Mish(nn.Module):
    """Mish激活函數模組"""
    def __init__(self): super().__init__()
    def forward(self, input): return mish(input)


class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            Mish(),
            nn.Linear(128, 128),
            Mish()
        )
        self.fc_mean = nn.Linear(128, out_dim)
        self.fc_log_std = nn.Linear(128, out_dim)

        initialize_uniformly(self.model)
        initialize_uniformly(self.fc_mean)
        initialize_uniformly(self.fc_log_std)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        X = self.model(state)
        mean = self.fc_mean(X)
        log_std = self.fc_log_std(X)
        LOG_STD_MIN = -20
        LOG_STD_MAX = 2
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std) + 1e-5
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.tanh(action) * 2.0

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            Mish(),
            nn.Linear(128, 128),
            Mish(),
            nn.Linear(128, 1),
        )
        initialize_uniformly(self.model)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        value = self.model(state)

        return value
    
def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float) -> List:
    """Compute gae."""

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

    def __init__(self, env: gym.Env, config=None, result_dir="result-ppo_pendulum_sweep-14", use_wandb=True):
        """Initialize."""
        self.env = env
        self.gamma = config.gamma
        self.tau = config.tau
        self.batch_size = config.batch_size
        self.epsilon = config.epsilon
        self.rollout_len = config.rollout_len
        self.entropy_weight = config.entropy_weight
        self.seed = config.seed
        self.update_epoch = config.update_epoch
        self.max_env_step = config.max_env_step
        self.actor_lr = config.actor_lr
        self.critic_lr = config.critic_lr
        self.use_wandb = use_wandb
        
        # 創建結果目錄
        self.result_dir = result_dir

        # 創建結果目錄 - 使用流水編號
        self.result_dir = result_dir
        # 獲取當前已存在的實驗編號，並創建新的編號
        existing_dirs = glob.glob(f"{self.result_dir}/exp_*")
        if existing_dirs:
            last_exp_num = max([int(d.split('_')[-1]) for d in existing_dirs])
            exp_num = last_exp_num + 1
        else:
            exp_num = 1
        
        self.exp_dir = f"{self.result_dir}/exp_{exp_num}"
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 創建CSV文件記錄最佳模型表現
        self.csv_path = os.path.join(self.exp_dir, "best_model_records.csv")
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['episode', 'train_reward', 'avg_test_reward'])
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用設備: {self.device}")

        # networks
        obs_dim = env.observation_space.shape[0]
        self.obs_dim = obs_dim
        action_dim = env.action_space.shape[0]
        self.action_dim = action_dim
        print(f"狀態空間維度: {obs_dim}")
        print(f"動作空間維度: {action_dim}")
        
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

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
        
        # 記錄最佳回報
        self.best_reward = -float('inf')
        self.best_test_reward = -float('inf')

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
        
        if not self.is_test and self.use_wandb: # Log during training
            import wandb
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
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
            policy_loss_elementwise = -torch.min(surr1, surr2)

            # 更新的熵計算和 actor_loss 計算
            entropy = dist.entropy() 
            if entropy.ndim > 1 and self.action_dim > 1: 
                entropy_bonus = entropy.sum(dim=-1).mean() 
            else:
                entropy_bonus = entropy.mean()
            
            actor_loss = policy_loss_elementwise.mean() - self.entropy_weight * entropy_bonus

            # critic_loss
            current_values = self.critic(state)
            critic_loss = F.mse_loss(return_, current_values)
            
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
        
        all_rewards = []
        
        pbar = tqdm(total=self.max_env_step, desc="訓練進度")
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
                    all_rewards.append(score)
                    episode_count += 1
                    
                    if self.use_wandb:
                        import wandb
                        wandb.log({
                            "episode_score": score,
                            "episode": episode_count,
                            "global_step": self.total_step
                        })
                    
                    # 每25個回合測試模型性能
                    if episode_count % 25 == 0:
                        # 保存當前檢查點
                        self.save_checkpoint(episode_count, score)
                        
                        # 測試當前模型
                        test_env = gym.make("Pendulum-v1")
                        test_rewards = []
                        
                        # 記錄當前模式並切換到測試模式
                        current_is_test = self.is_test
                        self.is_test = True
                        
                        # 進行8次測試
                        for test_ep in range(8):
                            test_state, _ = self.env.reset(seed=self.seed + test_ep)
                            test_state = np.expand_dims(test_state, axis=0)
                            test_done = False
                            test_score = 0
                            
                            while not test_done:
                                test_action = self.select_action(test_state)
                                test_next_state, test_reward, test_done = self.step(test_action)
                                test_state = test_next_state
                                test_score += test_reward[0][0]
                            
                            test_rewards.append(test_score)
                        
                        # 恢復原來的模式
                        self.is_test = current_is_test
                        
                        # 關閉測試環境
                        test_env.close()
                        
                        # 計算平均測試獎勵
                        avg_test_reward = np.mean(test_rewards)
                        
                        if self.use_wandb:
                            import wandb
                            wandb.log({
                                "episode": episode_count,
                                "avg_test_reward": avg_test_reward,
                            })

                        # 如果是最佳測試性能，保存為最佳模型
                        if avg_test_reward > self.best_test_reward:
                            self.best_test_reward = avg_test_reward
                            print(f"發現新的最佳模型，平均測試獎勵: {avg_test_reward:.2f}, 回合: {episode_count}")
                            
                            # 儲存最佳模型
                            self.save_best_model(episode_count, score, avg_test_reward)
                            
                            # 更新CSV記錄
                            with open(self.csv_path, 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow([episode_count, score, avg_test_reward])
                    
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
            
            if self.use_wandb:
                import wandb
                wandb.log({
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "global_step": self.total_step
                })
            
            pbar.update(self.rollout_len)

        # 訓練結束時保存最後一個檢查點
        self.save_checkpoint(episode_count, score)
        
        pbar.close()
        # termination
        self.env.close()
        
        return all_rewards
    
    def save_checkpoint(self, episode, reward):
        """保存檢查點"""
        checkpoint_path = os.path.join(self.exp_dir, f"checkpoint_ep{episode}.pt")
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'episode': episode,
            'reward': reward,
            'gamma': self.gamma,
            'tau': self.tau,
            'epsilon': self.epsilon
        }, checkpoint_path)
        return checkpoint_path
    
    def save_best_model(self, episode, reward, test_reward):
        """保存最佳模型"""
        best_model_path = os.path.join(self.exp_dir, "best_model.pt")
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'episode': episode,
            'reward': reward,
            'test_reward': test_reward,
            'gamma': self.gamma,
            'tau': self.tau,
            'epsilon': self.epsilon
        }, best_model_path)
        return best_model_path
    
    def load_checkpoint(self, checkpoint_path=None):
        """加載檢查點"""
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                # 嘗試正常加載，不使用weights_only
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            except RuntimeError as e:
                print(f"加載檢查點時出錯: {e}")
                try:
                    # 如果需要使用weights_only，先添加安全全局變量
                    try:
                        import numpy as np
                        from torch.serialization import add_safe_globals
                        add_safe_globals(["numpy.core.multiarray.scalar"])
                        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                    except:
                        # 如果還是失敗，嘗試不使用weights_only
                        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                except Exception as e:
                    print(f"所有嘗試都失敗: {e}")
                    return False
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            print(f"已加載檢查點: {checkpoint_path}")
            return True
        return False
    
    def test(self, env=None, checkpoint_path=None, num_episodes=10):
        """測試代理"""
        self.is_test = True
        
        # 使用提供的環境或默認環境
        test_env = env if env is not None else self.env
        
        # 加載模型檢查點（如果提供）
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        all_scores = []
        
        for ep in range(num_episodes):
            state, _ = test_env.reset(seed=self.seed + ep)  # 不同的種子以獲得更穩定的評估
            state = np.expand_dims(state, axis=0)
            
            done = False
            episode_reward = 0
            
            step_count = 0
            while not done:
                step_count += 1
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                state = next_state
                # 直接累加原始獎勵，不進行任何縮放
                episode_reward += reward[0][0]
            
            # 將最終總獎勵添加到列表
            all_scores.append(episode_reward)
            print(f"測試回合 {ep+1}/{num_episodes}: 獎勵 = {episode_reward:.2f}, 步數 = {step_count}")
        
        avg_score = np.mean(all_scores)
        print(f"平均測試獎勵: {avg_score:.2f}")
        
        return all_scores


def seed_torch(seed):
    """設置PyTorch隨機種子"""
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_agent_with_config(use_wandb=True, test_mode=False):
    """訓練並測試代理，將測試結果作為超參數優化指標"""
    # 初始化 wandb 設置
    if use_wandb:
        import wandb
        run = wandb.init()
        # 從 wandb 獲取超參數配置
        config = wandb.config
    else:
        # 如果不使用wandb，使用默認配置
        config = argparse.Namespace(
            actor_lr=1e-4,
            critic_lr=1e-3,
            gamma=0.95,
            tau=0.95,
            batch_size=64,
            epsilon=0.2,
            rollout_len=2048,
            entropy_weight=0.01,
            update_epoch=10,
            max_env_step=200000 if not test_mode else 10000,
            seed=42
        )
    
    # 創建環境
    env = gym.make("Pendulum-v1")
    
    # 設置隨機種子
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    
    # 創建結果目錄
    result_dir = f"result-ppo_pendulum_sweep-{no}"
    os.makedirs(result_dir, exist_ok=True)
    
    # 創建 PPO 代理
    agent = PPOAgent(env, config, result_dir, use_wandb=use_wandb)
    
    # 訓練代理
    episode_rewards = agent.train()
    
    # 訓練結束後，執行最終測試
    # 找到最佳模型檢查點
    best_model_path = os.path.join(agent.exp_dir, "best_model.pt")
    
    if os.path.exists(best_model_path):
        # 創建測試環境
        test_env = gym.make("Pendulum-v1")
        
        # 創建測試代理
        test_agent = PPOAgent(test_env, config, result_dir, use_wandb=use_wandb)
        
        # 執行測試
        NUM_TEST_EPISODES = 5 if test_mode else 20  # 測試模式時減少測試回合
        print(f"\n===== 開始最終測試 - 評估超參數性能 =====")
        test_scores = test_agent.test(
            checkpoint_path=best_model_path,
            num_episodes=NUM_TEST_EPISODES
        )
        
        # 計算平均測試獎勵
        avg_test_reward = np.mean(test_scores)
        print(f"測試完成! 平均測試獎勵: {avg_test_reward:.2f}")
        
        # 記錄平均測試獎勵作為優化目標
        if use_wandb:
            wandb.log({"avg_test_reward": avg_test_reward})
        
        # 關閉測試環境
        test_env.close()
    else:
        print(f"未找到最佳模型檔案: {best_model_path}")
    
    # 關閉訓練環境
    env.close()
    
    return avg_test_reward if 'avg_test_reward' in locals() else -float('inf')


# 超參數搜索空間定義
sweep_config = {
    'method': 'bayes',  # 使用貝葉斯優化方法
    'metric': {
        'name': 'avg_test_reward',  # 優化測試平均回報值
        'goal': 'maximize'  # 目標是最大化回報
    },
    'parameters': {
        'actor_lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3
        },
        'critic_lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-2
        },
        'gamma': {
            'distribution': 'uniform',
            'min': 0.85,
            'max': 0.999
        },
        'tau': {
            'distribution': 'uniform',
            'min': 0.8,
            'max': 0.99
        },
        'batch_size': {
            'values': [32, 64, 128, 256]
        },
        'epsilon': {
            'values': [0.1, 0.2, 0.3]
        },
        'rollout_len': {
            'values': [1024, 2048, 4096]
        },
        'entropy_weight': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-1
        },
        'update_epoch': {
            'values': [5, 10, 15, 20]
        },
        'max_env_step': {
            'value': 200000
        },
        'seed': {
            'values': [42, 77, 123, 456, 789, 1000, 1234, 1456, 1789, 2000]
        }
    }
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100, help="要執行的 sweep 運行次數")
    parser.add_argument("--no-use-wandb", action="store_true", help="不使用 wandb 進行實驗追蹤")
    parser.add_argument("--test", action="store_true", help="以測試模式運行（減少回合數）")
    args = parser.parse_args()
    
    use_wandb = not args.no_use_wandb
    test_mode = args.test
    
    if use_wandb:
        import wandb
        # 初始化 sweep
        sweep_id = wandb.sweep(sweep_config, project=f"DLP-Lab7-PPO-Pendulum-Sweep-{no}")
        
        # 執行 sweep
        wandb.agent(sweep_id, lambda: train_agent_with_config(use_wandb=True, test_mode=test_mode), count=args.count)
    else:
        print("以無wandb模式運行單次訓練...")
        train_agent_with_config(use_wandb=False, test_mode=test_mode) 