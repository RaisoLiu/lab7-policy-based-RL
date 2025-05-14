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
from a2c_pendulum import A2CAgent, seed_torch

def train_agent_with_config():
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
            # 保存相關參數
            self.save_per_epoch = getattr(config, 'save_per_epoch', 100)
            self.result_dir = "result-a2c_pendulum_sweep"
    
    args = Args(config)
    
    # 創建並訓練代理
    agent = A2CAgent(env, args)
    agent.train()
    
    # 關閉環境
    env.close()

# 超參數搜索空間定義
sweep_config = {
    'method': 'bayes',  # 使用貝葉斯優化方法
    'metric': {
        'name': 'return',  # 優化回報值
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
            'value': 500  # 固定評估回合數以節省時間
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
    parser.add_argument("--count", type=int, default=30, help="Number of sweep runs to perform")
    args = parser.parse_args()
    
    # 初始化 sweep
    sweep_id = wandb.sweep(sweep_config, project="DLP-Lab7-A2C-Pendulum-Sweep")
    
    # 執行 sweep
    wandb.agent(sweep_id, train_agent_with_config, count=args.count) 