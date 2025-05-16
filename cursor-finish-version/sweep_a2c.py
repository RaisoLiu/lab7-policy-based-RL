#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C with Wandb Sweep for hyperparameter optimization

import random
import gymnasium as gym
import numpy as np
import torch
import wandb
import argparse
import os
import yaml
from a2c_pendulum import A2CAgent, seed_torch

def train_agent_with_config():
    """訓練並測試代理，將測試結果作為超參數優化的指標"""
    # 初始化 wandb 設置
    run = wandb.init()
    
    # 從 wandb 獲取超參數配置
    config = wandb.config
    
    # 創建環境
    env = gym.make("Pendulum-v1")
    
    # 設置隨機種子
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    
    # 創建參數物件
    class Args:
        def __init__(self, config):
            self.actor_lr = config.actor_lr
            self.critic_lr = config.critic_lr
            self.discount_factor = config.discount_factor
            self.entropy_weight = config.entropy_weight
            self.num_episodes = config.num_episodes
            self.seed = config.seed
            # 網絡結構參數
            self.actor_hidden_dim1 = config.actor_hidden_dim1
            self.actor_hidden_dim2 = config.actor_hidden_dim2
            self.critic_hidden_dim1 = config.critic_hidden_dim1
            self.critic_hidden_dim2 = config.critic_hidden_dim2
            # 獎勵縮放和限制參數
            self.reward_scaling = getattr(config, 'reward_scaling', 0.1)
            self.reward_clamp_min = getattr(config, 'reward_clamp_min', -1.0)
            self.reward_clamp_max = getattr(config, 'reward_clamp_max', 1.0)
            # 保存相關參數
            self.save_per_epoch = getattr(config, 'save_per_epoch', 100)
            self.result_dir = "result-a2c_pendulum_sweep_reward_scaling_clamp"
    
    args = Args(config)
    
    # 創建並訓練代理
    agent = A2CAgent(env, args)
    agent.train()
    
    # 訓練結束後，執行測試
    # 找到最好的模型檢查點
    best_model_path = os.path.join(agent.exp_dir, 'best_model.pt')
    
    # 如果最佳模型不存在，使用最後一個檢查點
    if not os.path.exists(best_model_path):
        checkpoint_files = [f for f in os.listdir(agent.exp_dir) if f.startswith('checkpoint_ep') and f.endswith('.pt')]
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: int(x.split('_ep')[1].split('.')[0]))
            best_model_path = os.path.join(agent.exp_dir, checkpoint_files[-1])
    
    if os.path.exists(best_model_path):
        # 創建測試環境
        test_env = gym.make("Pendulum-v1")
        
        # 創建測試代理
        test_agent = A2CAgent(test_env, args, is_test=True)
        
        # 執行 20 次測試
        NUM_TEST_EPISODES = 20
        print(f"\n===== 開始測試 - 使用原始獎勵（不縮放）評估超參數性能 =====")
        test_scores = test_agent.test(
            checkpoint_path=best_model_path, 
            num_episodes=NUM_TEST_EPISODES,
            render=False  # 不生成視頻，加速測試
        )
        
        # 計算平均測試獎勵
        avg_test_reward = np.mean(test_scores)
        print(f"測試完成! 平均測試獎勵: {avg_test_reward:.2f}")
        
        # 記錄平均測試獎勵作為優化目標
        wandb.log({"avg_test_reward": avg_test_reward})
        
        # 關閉測試環境
        test_env.close()
    else:
        print("未找到任何模型檢查點進行測試")
    
    # 關閉訓練環境
    env.close()

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
        'discount_factor': {
            'distribution': 'uniform',
            'min': 0.8,
            'max': 0.99
        },
        'entropy_weight': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-1
        },
        'reward_scaling': {
            'distribution': 'uniform',
            'min': 0.05,
            'max': 0.5
        },
        'reward_clamp_min': {
            'values': [-2.0, -1.0, -0.5]
        },
        'reward_clamp_max': {
            'values': [0.5, 1.0, 2.0]
        },
        'actor_hidden_dim1': {
            'values': [64, 128, 256]
        },
        'actor_hidden_dim2': {
            'values': [32, 64, 128]
        },
        'critic_hidden_dim1': {
            'values': [64, 128, 256]
        },
        'critic_hidden_dim2': {
            'values': [32, 64, 128]
        },
        'num_episodes': {
            'value': 1000 # Don't change this value
        },
        'save_per_epoch': {
            'value': 100  # 每 100 個回合保存一次檢查點
        },
        'seed': {
            'values': [42, 77, 123, 456, 789]  # 多個隨機種子以提高穩定性
        }
    }
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50, help="Number of sweep runs to perform")
    args = parser.parse_args()
    
    # 初始化 sweep
    sweep_id = wandb.sweep(sweep_config, project="DLP-Lab7-A2C-Pendulum-Sweep-Reward-Scaling-Clamp")
    
    # 執行 sweep
    wandb.agent(sweep_id, train_agent_with_config, count=args.count) 
