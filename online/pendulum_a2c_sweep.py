#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Pendulum A2C Sweep: 超參數優化搜索

import random
import gymnasium as gym
import numpy as np
import torch
import wandb
import argparse
import os
import yaml
from pendulum_a2c_online import Actor, Critic, A2CLearner, Runner, Mish, seed_torch

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
    
    # 定義模型參數
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    
    # 創建 Actor 和 Critic 網絡
    actor = Actor(state_dim, n_actions, activation=Mish).to(device)
    critic = Critic(state_dim, activation=Mish).to(device)
    
    # 創建學習器
    learner = A2CLearner(
        actor,
        critic,
        gamma=config.gamma,
        entropy_beta=config.entropy_beta,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        save_per_epoch=config.save_per_epoch,
        result_dir=config.result_dir,
        mode="train"
    )
    
    # 創建 Runner
    runner = Runner(env, learner, actor, device)
    
    # 訓練代理
    episode_rewards = runner.train(
        episodes=config.num_episodes,
        episode_length=config.episode_length,
        steps_on_memory=config.steps_on_memory
    )
    
    # 訓練結束後，執行測試
    # 找到最好的模型檢查點
    best_model_path = os.path.join(learner.exp_dir, 'best_model.pt')
    
    # 如果最佳模型不存在，使用最後一個檢查點
    if not os.path.exists(best_model_path):
        checkpoint_files = [f for f in os.listdir(learner.exp_dir) if f.startswith('checkpoint_ep') and f.endswith('.pt')]
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: int(x.split('_ep')[1].split('.')[0]))
            best_model_path = os.path.join(learner.exp_dir, checkpoint_files[-1])
    
    if os.path.exists(best_model_path):
        # 創建測試環境
        test_env = gym.make("Pendulum-v1")
        
        # 創建測試學習器和運行器
        test_actor = Actor(state_dim, n_actions, activation=Mish).to(device)
        test_critic = Critic(state_dim, activation=Mish).to(device)
        test_learner = A2CLearner(
            test_actor,
            test_critic,
            gamma=config.gamma,
            entropy_beta=config.entropy_beta,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            mode="test"
        )
        test_runner = Runner(test_env, test_learner, test_actor, device)
        
        # 執行測試
        NUM_TEST_EPISODES = 20
        print(f"\n===== 開始測試 - 評估超參數性能 =====")
        test_scores = test_runner.test(
            checkpoint_path=best_model_path, 
            num_episodes=NUM_TEST_EPISODES,
            seed=seed
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
        'gamma': {
            'distribution': 'uniform',
            'min': 0.8,
            'max': 0.99
        },
        'entropy_beta': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-1
        },
        'num_episodes': {
            'value': 1000  # 固定值，不要修改
        },
        'episode_length': {
            'value': 200  # 固定值，每個回合步數
        },
        'steps_on_memory': {
            'values': [8, 16, 32]  # 記憶的步數
        },
        'save_per_epoch': {
            'value': 100  # 每 100 個回合保存一次檢查點
        },
        'result_dir': {
            'value': 'result-a2c_pendulum_sweep'  # 結果保存目錄
        },
        'seed': {
            'values': [42, 77, 123, 456, 789]  # 多個隨機種子以提高穩定性
        }
    }
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50, help="要執行的 sweep 運行次數")
    args = parser.parse_args()
    
    # 初始化 sweep
    sweep_id = wandb.sweep(sweep_config, project="DLP-Lab7-Pendulum-A2C-Online-Sweep")
    
    # 執行 sweep
    wandb.agent(sweep_id, train_agent_with_config, count=args.count) 