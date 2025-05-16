#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# A2C Pendulum Sweep: 超參數優化搜索
no = 14

import random
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
from typing import Tuple, Optional, List, Dict, Any
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
    def __init__(self, state_dim, n_actions, activation=nn.Tanh):
        """初始化Actor網絡"""
        super().__init__()
        self.n_actions = n_actions
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, n_actions)
        )
        
        logstds_param = nn.Parameter(torch.full((n_actions,), 0.1))
        self.register_parameter("logstds", logstds_param)
    
    def forward(self, X):
        """順向傳播實現"""
        means = self.model(X)
        stds = torch.clamp(self.logstds.exp(), 1e-3, 50)
        
        return torch.distributions.Normal(means, stds)


class Critic(nn.Module):
    def __init__(self, state_dim, activation=nn.Tanh):
        """初始化Critic網絡"""
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, 1),
        )
    
    def forward(self, X):
        """順向傳播實現"""
        return self.model(X)


def process_memory(memory, device="cpu"):
    """處理記憶中的轉換，不再計算折扣獎勵，而是直接返回單步原始獎勵"""
    # 檢查記憶庫是否為空
    if not memory:
        raise ValueError("記憶庫為空，無法處理")
    
    states_np_list = []
    actions_tensor_list = []
    next_states_np_list = []
    raw_rewards_list = []
    dones_list = []
    
    for s_np, a_tensor, ns_np, r_float, d_bool in memory:
        states_np_list.append(s_np)
        actions_tensor_list.append(a_tensor)  # a_tensor 已經在設備上並已分離
        next_states_np_list.append(ns_np)
        raw_rewards_list.append(r_float)
        dones_list.append(d_bool)
    
    # 確保正確的數據轉換
    states_batch = torch.FloatTensor(np.array(states_np_list)).to(device)
    actions_batch = torch.stack(actions_tensor_list).to(device)  # 堆疊張量列表
    next_states_batch = torch.FloatTensor(np.array(next_states_np_list)).to(device)
    raw_rewards_batch = torch.FloatTensor(raw_rewards_list).view(-1, 1).to(device)
    dones_batch = torch.FloatTensor(dones_list).view(-1, 1).to(device)  # 確保為浮點型用於乘法
    
    return states_batch, actions_batch, next_states_batch, raw_rewards_batch, dones_batch


def clip_grad_norm_(module, max_grad_norm):
    """梯度裁剪函數"""
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)


class A2CAgent:
    """A2C代理與環境互動"""
    
    def __init__(self, env: gym.Env, config=None, result_dir="result-a2c_pendulum_sweep-5", use_wandb=True):
        """初始化代理"""
        self.env = env
        self.gamma = config.gamma
        self.entropy_weight = config.entropy_beta
        self.seed = config.seed
        self.actor_lr = config.actor_lr
        self.critic_lr = config.critic_lr
        self.num_episodes = config.num_episodes
        self.steps_on_memory = config.steps_on_memory
        # 使用config中的max_grad_norm（如果有），否則使用默認值
        self.max_grad_norm = getattr(config, 'max_grad_norm', 0.5)
        self.save_per_epoch = config.save_per_epoch
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
        # self.exp_dir = f"{self.result_dir}/seed_{self.seed}_lr_{self.actor_lr}_{self.critic_lr}_norm_{self.max_grad_norm}"
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
        action_dim = env.action_space.shape[0]
        print(f"狀態空間維度: {obs_dim}")
        print(f"動作空間維度: {action_dim}")
        
        # 使用新的網絡架構
        self.actor = Actor(obs_dim, action_dim, activation=Mish).to(self.device)
        self.critic = Critic(obs_dim, activation=Mish).to(self.device)
        
        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        # 存儲當前經驗的字典
        self.current_experience_for_memory = {"state_np": None, "action_tensor_device": None}
        
        # memory
        self.memory = []
        
        # total steps count
        self.total_step = 0
        
        # mode: train / test
        self.is_test = False
        
        # 記錄最佳回報
        self.best_reward = -float('inf')
        self.best_test_reward = -float('inf')
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """選擇動作，但不再預先計算log_prob"""
        state_torch = torch.FloatTensor(state).to(self.device)  # state是numpy格式
        dist = self.actor(state_torch)  # 獲取動作分佈
        
        if self.is_test:
            action_candidate_torch = dist.mean  # 測試時使用分佈的均值
        else:
            action_candidate_torch = dist.sample()  # 訓練時從分佈中採樣
        
        # 確保動作被正確裁剪並轉換為numpy格式
        action_np = action_candidate_torch.clamp(-2.0, 2.0).cpu().detach().numpy()
        
        if not self.is_test:
            # 確保state為numpy數組
            if isinstance(state, np.ndarray):
                state_copy = state.copy()  # 創建狀態的副本
            else:
                state_copy = np.array(state, dtype=np.float32)  # 確保是numpy數組
            
            # 儲存原始numpy格式的狀態和在設備上分離的動作張量
            self.current_experience_for_memory = {
                "state_np": state_copy,  # 原始numpy狀態的副本
                "action_tensor_device": action_candidate_torch.detach()  # 分離以避免計算圖問題
            }
        
        return action_np
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """執行動作並返回環境反饋，更新memory存儲格式"""
        next_state_np, reward_float, terminated, truncated, _ = self.env.step(action)
        done_bool = terminated or truncated
        
        if not self.is_test:
            # 獲取存儲的狀態和動作
            s_np = self.current_experience_for_memory["state_np"]
            a_tensor_dev = self.current_experience_for_memory["action_tensor_device"]
            
            # 將完整的轉換添加到memory中
            self.memory.append((s_np, a_tensor_dev, next_state_np, reward_float, done_bool))
        
        return next_state_np, reward_float, done_bool
    
    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """通過梯度下降更新模型，使用on-policy學習"""
        # 檢查記憶庫是否為空
        if not self.memory:
            print("警告: 記憶庫為空，跳過本次更新")
            return 0.0, 0.0  # 返回零損失
        
        # 1. 處理記憶數據以獲取批次
        states, actions, next_states, raw_rewards, dones = process_memory(
            self.memory, self.device
        )
        
        # 2. 計算Critic的TD(0)目標
        with torch.no_grad():
            next_values = self.critic(next_states)
            # V(s_t)的目標是r_t + gamma * V(s_{t+1}) * (1 - done_t)
            td_target = raw_rewards + self.gamma * next_values * (1 - dones)
        
        # 3. 更新Critic
        current_values = self.critic(states)
        value_loss = F.mse_loss(td_target, current_values)
        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        clip_grad_norm_(self.critic_optimizer, self.max_grad_norm)
        self.critic_optimizer.step()
        
        # 4. 計算優勢函數
        with torch.no_grad():
            advantage = td_target - current_values.detach()
        
        # 5. 更新Actor（On-Policy方式）
        dist = self.actor(states)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)  # 對多維動作求和
        entropy = dist.entropy().mean()
        
        actor_loss = (-log_probs * advantage).mean() - self.entropy_weight * entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor_optimizer, self.max_grad_norm)
        self.actor_optimizer.step()
        
        # 更新後清空記憶庫
        self.memory = []
        
        return actor_loss.item(), value_loss.item()

    
    def train(self):
        """訓練代理"""
        self.is_test = False
        step_count = 0
        all_rewards = []
        
        for ep in tqdm(range(1, int(self.num_episodes) + 1)):
            actor_losses, critic_losses, scores = [], [], []
            state, _ = self.env.reset() # train 不指定 seed，只在 test 時指定 seed
            score = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                
                # 如果記憶庫中沒有足夠的樣本且遊戲沒有結束，則繼續收集
                if len(self.memory) < self.steps_on_memory and not done:
                    state = next_state
                    score += reward
                    continue
                
                # 確保記憶庫不為空再更新模型
                if self.memory:
                    actor_loss, critic_loss = self.update_model()
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                else:
                    print("警告: 記憶庫為空，跳過更新")
                    # 繼續收集經驗而不進行更新
                    state = next_state
                    score += reward
                    continue
                
                state = next_state
                score += reward
                step_count += 1
                self.total_step += 1
                
                # 記錄訓練指標
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "total_step": self.total_step,
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss,
                    })
            
            # 如果回合結束
            if done:
                scores.append(score)
                all_rewards.append(score)
                # if ep % 100 == 0:
                    # print(f"回合 {ep}: 總獎勵 = {score}")
                
                # 記錄訓練指標
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "total_step": self.total_step,
                        "episode": ep,
                        "return": score,
                    })
                
                # 每100個回合測試模型性能
                if ep % 100 == 0 or ep == self.num_episodes:
                    # 保存當前檢查點
                    self.save_checkpoint(ep, score)
                    
                if ep % 25 == 0 or ep == self.num_episodes:
                    # 測試當前模型，注意：直接使用當前模型，不加載檢查點
                    # print(f"\n===== 測試回合 {ep} 的模型性能 =====")
                    test_env = gym.make("Pendulum-v1")
                    test_rewards = []
                    
                    # 記錄當前模式並切換到測試模式
                    current_is_test = self.is_test
                    self.is_test = True
                    
                    # 進行20次測試
                    for test_ep in range(8):
                        test_state, _ = self.env.reset()
                        test_done = False
                        test_score = 0
                        
                        step_count = 0
                        while not test_done:
                            test_action = self.select_action(test_state)
                            test_next_state, test_reward, test_done = self.step(test_action)
                            test_state = test_next_state
                            test_score += test_reward
                            step_count += 1
                        
                        test_rewards.append(test_score)
                        # print(f"測試回合 {test_ep+1}/20: 獎勵 = {test_score:.2f}, 步數 = {step_count}")
                    
                    # 恢復原來的模式
                    self.is_test = current_is_test
                    
                    # 關閉測試環境
                    test_env.close()
                    
                    # 計算平均測試獎勵
                    avg_test_reward = np.mean(test_rewards)
                    # print(f"回合 {ep} 平均測試獎勵: {avg_test_reward:.2f}")
                    if self.use_wandb:
                        import wandb
                        wandb.log({
                            "episode": ep,
                            "avg_test_reward": avg_test_reward,
                        })

                    # 如果是最佳測試性能，保存為最佳模型
                    if avg_test_reward > self.best_test_reward:
                        self.best_test_reward = avg_test_reward
                        best_model_path = os.path.join(self.exp_dir, "best_model.pt")
                        print(f"發現新的最佳模型，平均測試獎勵: {avg_test_reward:.2f}, 回合: {ep}")
                        
                        # 複製當前檢查點為最佳模型
                        self.save_best_model(ep, score, avg_test_reward)
                        
                        # 更新CSV記錄
                        with open(self.csv_path, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([ep, score, avg_test_reward])
        
        # 訓練結束時保存最後一個檢查點
        final_checkpoint_path = os.path.join(self.exp_dir, f"checkpoint_ep{self.num_episodes}.pt")
        self.save_checkpoint(self.num_episodes, score)
        
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
            'max_grad_norm': self.max_grad_norm,
            'gamma': self.gamma
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
            'max_grad_norm': self.max_grad_norm,
            'gamma': self.gamma
        }, best_model_path)
        return best_model_path
    
    def evaluate_model(self, checkpoint_path=None, num_episodes=20):
        """評估模型性能"""
        # 創建一個新的評估環境
        eval_env = gym.make("Pendulum-v1")
        
        # 保存原有的模型狀態
        # actor_state = self.actor.state_dict()
        # critic_state = self.critic.state_dict()
        
        # 加載檢查點
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        # 進行評估
        eval_rewards = self.test(eval_env, num_episodes=num_episodes)
        print(f"評估結束，評估獎勵: {eval_rewards}")

        self.is_test = False
        
        # 恢復原有模型狀態
        # self.actor.load_state_dict(actor_state)
        # self.critic.load_state_dict(critic_state)
        
        # 關閉評估環境
        eval_env.close()
        
        return eval_rewards
    
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
            # state, _ = test_env.reset(seed=self.seed + ep)  # 不同的種子以獲得更穩定的評估
            state, _ = self.env.reset(seed=self.seed + ep)
            
            done = False
            episode_reward = 0
            
            step_count = 0
            # 重要：在Pendulum環境中，reward範圍為[-16.2736044, 0]
            # 負值越大（越接近0）表示表現越好
            while not done:
                step_count += 1
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                state = next_state
                # 直接累加原始獎勵，不進行任何縮放
                episode_reward += reward
            
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
        # config = argparse.Namespace(
        #     actor_lr=0.0003,
        #     critic_lr=0.003,
        #     gamma=0.99,
        #     entropy_beta=0.01,
        #     num_episodes=10 if test_mode else 1000,  # 測試模式時減少回合數
        #     steps_on_memory=16,
        #     save_per_epoch=5 if test_mode else 100,  # 測試模式時更頻繁保存
        #     seed=42,
        #     max_grad_norm=0.5
        # )
        config = argparse.Namespace(
            actor_lr=3e-4,  # 較高的學習率
            critic_lr=3e-3,  # 較高的學習率
            gamma=0.95,      # 對Pendulum使用0.9的折扣因子
            entropy_beta=2.5e-4,  # 減小熵權重，增加利用率
            num_episodes=10 if test_mode else 1000,
            steps_on_memory=16,  # 合適的記憶步數
            save_per_epoch=5 if test_mode else 100,
            seed=42,
            max_grad_norm=10.0   # 較小的梯度裁剪閾值，增加穩定性
        )
    
    # 創建環境
    env = gym.make("Pendulum-v1")
    
    # 設置隨機種子
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    
    # 創建結果目錄
    result_dir = f"result-a2c_pendulum_sweep-{no}"
    os.makedirs(result_dir, exist_ok=True)
    
    # 創建 A2C 代理
    agent = A2CAgent(env, config, result_dir, use_wandb=use_wandb)
    
    # 訓練代理
    episode_rewards = agent.train()
    
    # 訓練結束後，執行最終測試
    # 找到最佳模型檢查點
    best_model_path = os.path.join(agent.exp_dir, "best_model.pt")
    
    if os.path.exists(best_model_path):
        # 創建測試環境
        test_env = gym.make("Pendulum-v1")
        
        # 創建測試代理
        test_agent = A2CAgent(test_env, config, result_dir, use_wandb=use_wandb)
        
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
            'distribution': 'log_uniform_values',
            'min': 0.85,
            'max': 0.999
        },
        'entropy_beta': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-1
        },
        'max_grad_norm': {
            'values': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]  # 測試不同的梯度裁剪值
        },
        'num_episodes': {
            'value': 1000  # 固定值，不要修改
        },
        'steps_on_memory': {
            'values': [8, 16, 32]  # 記憶的步數
        },
        'save_per_epoch': {
            'value': 100  # 每 100 個回合保存一次檢查點
        },
        'seed': {
            'values': [42, 77, 123, 456, 789, 1000, 1234, 1456, 1789, 2000, 2234, 2456, 2789, 3000, 3234, 3456, 3789, 4000, 4234, 4456, 4789, 5000]  # 多個隨機種子以提高穩定性
        }
    }
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500, help="要執行的 sweep 運行次數")
    parser.add_argument("--no-use-wandb", action="store_true", help="不使用 wandb 進行實驗追蹤")
    parser.add_argument("--test", action="store_true", help="以測試模式運行（減少回合數）")
    args = parser.parse_args()
    
    use_wandb = not args.no_use_wandb
    test_mode = args.test
    
    if use_wandb:
        import wandb
        # 初始化 sweep
        sweep_id = wandb.sweep(sweep_config, project=f"DLP-Lab7-A2C-Pendulum-Sweep-{no}")
        
        # 執行 sweep
        wandb.agent(sweep_id, lambda: train_agent_with_config(use_wandb=True, test_mode=test_mode), count=args.count)
    else:
        print("以無wandb模式運行單次訓練...")
        train_agent_with_config(use_wandb=False, test_mode=test_mode) 