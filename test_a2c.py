#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C - Testing script

import argparse
import gymnasium as gym
import numpy as np
import torch
import os
import yaml
from a2c_pendulum import A2CAgent, seed_torch
import random

def test_model(checkpoint_path, num_episodes=20, seed=77):
    """測試保存的模型檢查點"""
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"找不到檢查點文件: {checkpoint_path}")
    
    # 獲取實驗目錄
    exp_dir = os.path.dirname(checkpoint_path)
    checkpoint_name = os.path.basename(checkpoint_path).split('.')[0]
    
    # 讀取配置文件
    config_path = os.path.join(exp_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise ValueError(f"找不到配置文件: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 創建測試輸出目錄
    test_dir = os.path.join(exp_dir, f"test_{checkpoint_name}")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)
    
    # 創建環境
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    
    # 設置隨機種子
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    
    # 創建參數物件
    class Args:
        def __init__(self, config):
            # 從配置檔案讀取所有必要參數
            self.discount_factor = config.get('discount_factor', 0.9)
            self.actor_lr = config.get('actor_lr', 1e-4)
            self.critic_lr = config.get('critic_lr', 1e-3)
            self.entropy_weight = config.get('entropy_weight', 1e-2)
            self.actor_hidden_dim1 = config.get('actor_hidden_dim1', 128)
            self.actor_hidden_dim2 = config.get('actor_hidden_dim2', 64)
            self.critic_hidden_dim1 = config.get('critic_hidden_dim1', 128)
            self.critic_hidden_dim2 = config.get('critic_hidden_dim2', 64)
            # 獎勵縮放和限制參數 (僅用於加載配置，測試時不會應用)
            self.reward_scaling = config.get('reward_scaling', 0.1)
            self.reward_clamp_min = config.get('reward_clamp_min', -1.0)
            self.reward_clamp_max = config.get('reward_clamp_max', 1.0)
            # 覆蓋種子以使用命令行提供的值
            self.seed = seed
            self.num_episodes = num_episodes
    
    args = Args(config)
    
    # 創建代理並加載檢查點
    agent = A2CAgent(env, args)
    agent.test(video_folder=test_dir, checkpoint_path=checkpoint_path, num_episodes=num_episodes)
    
    print(f"測試完成! 結果保存在 {test_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="要測試的模型檢查點路徑")
    parser.add_argument("--num-episodes", type=int, default=20, help="測試回合數")
    parser.add_argument("--seed", type=int, default=77, help="隨機種子")
    args = parser.parse_args()
    
    test_model(args.checkpoint, args.num_episodes, args.seed) 