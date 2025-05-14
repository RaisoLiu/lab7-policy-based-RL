# 強化學習實驗 TODO 列表

## 程式碼完成
- [ ] 在 `a2c_pendulum.py` 中完成 Actor 類別的實現
  - [ ] 完成 `__init__` 方法中的層權重初始化
  - [ ] 完成 `forward` 方法實現
- [ ] 在 `a2c_pendulum.py` 中完成 Critic 類別的實現
  - [ ] 完成 `__init__` 方法中的層權重初始化
  - [ ] 完成 `forward` 方法實現
- [ ] 在 `a2c_pendulum.py` 中完成 `update_model` 方法
  - [ ] 實現 value_loss 計算
  - [ ] 實現 policy_loss 計算

- [ ] 在 `ppo_pendulum.py` 和 `ppo_walker.py` 中完成 Actor 類別的實現
  - [ ] 完成 `__init__` 方法中的層權重初始化
  - [ ] 完成 `forward` 方法實現
- [ ] 在 `ppo_pendulum.py` 和 `ppo_walker.py` 中完成 Critic 類別的實現
  - [ ] 完成 `__init__` 方法中的層權重初始化
  - [ ] 完成 `forward` 方法實現
- [ ] 在 `ppo_pendulum.py` 和 `ppo_walker.py` 中完成 `compute_gae` 函數
- [ ] 在 `ppo_pendulum.py` 和 `ppo_walker.py` 中完成 `update_model` 中的 loss 計算
  - [ ] 實現 actor_loss 計算
  - [ ] 實現 critic_loss 計算

## 實驗與分析
- [ ] 執行 A2C 算法在 Pendulum 環境中的訓練
- [ ] 執行 PPO 算法在 Pendulum 環境中的訓練
- [ ] 執行 PPO 算法在 Walker 環境中的訓練
- [ ] 比較 A2C 和 PPO 在 Pendulum 環境中的表現

## 超參數調優（使用 Wandb Sweep）
- [ ] 為 A2C 設置 Wandb Sweep 配置
  - [ ] 定義搜索空間 (learning rate, entropy weight, discount factor 等)
  - [ ] 設置優化方法（網格搜索、貝葉斯優化等）
  - [ ] 定義評估指標
- [ ] 為 PPO-Pendulum 設置 Wandb Sweep 配置
  - [ ] 定義搜索空間 (learning rate, clip epsilon, GAE lambda 等)
  - [ ] 設置優化方法
  - [ ] 定義評估指標
- [ ] 為 PPO-Walker 設置 Wandb Sweep 配置
  - [ ] 優化特定於 Walker 環境的超參數
  - [ ] 調整網絡架構參數
- [ ] 分析超參數調優結果
  - [ ] 識別每個任務的最佳超參數組合
  - [ ] 可視化不同超參數對性能的影響

## 報告撰寫
- [ ] 描述實現的算法原理
- [ ] 提供實驗結果與分析
- [ ] 比較不同算法的性能差異
- [ ] 總結實驗結果與心得
- [ ] 加入超參數調優結果與分析

## 其他
- [ ] 確保所有實驗都使用相同的隨機種子以保證可重現性
- [ ] 使用 wandb 記錄實驗結果和指標
- [ ] 優化代碼以提高訓練效率
- [ ] 參數調優以提高算法性能 