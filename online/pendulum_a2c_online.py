import numpy as np
import torch
import gymnasium as gym
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import wandb
import os
import datetime
import yaml
import json
import csv
import argparse
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device:", device)

def mish(input):
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, input): return mish(input)


    # helper function to convert numpy arrays to tensors
def t(x):
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    return torch.from_numpy(x).float()


class Actor(nn.Module):
    def __init__(self, state_dim, n_actions, activation=nn.Tanh):
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
        means = self.model(X)
        stds = torch.clamp(self.logstds.exp(), 1e-3, 50)
        
        return torch.distributions.Normal(means, stds)
    

## Critic module
class Critic(nn.Module):
    def __init__(self, state_dim, activation=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, 1),
        )
    
    def forward(self, X):
        return self.model(X)
    

def discounted_rewards(rewards, dones, gamma):
    ret = 0
    discounted = []
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + ret * gamma * (1-done)
        discounted.append(ret)
    
    return discounted[::-1]


def process_memory(memory, gamma=0.99, discount_rewards=True):
    actions = []
    states = []
    next_states = []
    rewards = []
    dones = []

    for action, reward, state, next_state, done in memory:
        actions.append(action)
        rewards.append(reward)
        states.append(state)
        next_states.append(next_state)
        dones.append(done)
    
    if discount_rewards:
        if False and dones[-1] == 0:
            rewards = discounted_rewards(rewards + [last_value], dones + [0], gamma)[:-1]
        else:
            rewards = discounted_rewards(rewards, dones, gamma)

    actions = t(actions).view(-1, 1).to(device)
    states = t(states).to(device)
    next_states = t(next_states).to(device)
    rewards = t(rewards).view(-1, 1).to(device)
    dones = t(dones).view(-1, 1).to(device)
    return actions, rewards, states, next_states, dones

def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)


class A2CLearner():
    def __init__(self, actor, critic, gamma=0.9, entropy_beta=0,
                 actor_lr=4e-4, critic_lr=4e-3, max_grad_norm=0.5, 
                 save_per_epoch=100, result_dir="result-a2c_pendulum", mode="train",
                 use_wandb=False):
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.actor = actor
        self.critic = critic
        self.entropy_beta = entropy_beta
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        
        # 保存相關參數
        self.save_per_epoch = save_per_epoch
        self.result_dir = result_dir
        self.use_wandb = use_wandb
        
        # 創建實驗資料夾
        timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.exp_dir = os.path.join(self.result_dir, f"exp_{timestr}")
        if not os.path.exists(self.exp_dir) and mode == "train":
            os.makedirs(self.exp_dir, exist_ok=True)
        
        # 保存配置
        if mode == "train":
            config = {
                'gamma': gamma,
                'entropy_beta': entropy_beta,
                'actor_lr': actor_lr,
                'critic_lr': critic_lr,
                'max_grad_norm': max_grad_norm,
                'save_per_epoch': save_per_epoch,
            }
            self.save_config(config)
            
            # wandb 初始化 - 只在指定使用 wandb 時啟動
            if self.use_wandb and wandb.run is None:  # 確保還沒有初始化
                wandb.init(
                    project="a2c-pendulum", 
                    config=config,
                    dir=self.exp_dir
                )
    
    def learn(self, memory, steps, discount_rewards=True):
        actions, rewards, states, next_states, dones = process_memory(memory, self.gamma, discount_rewards)

        if discount_rewards:
            td_target = rewards
        else:
            td_target = rewards + self.gamma*self.critic(next_states)*(1-dones)
        value = self.critic(states)
        advantage = td_target - value

        # actor
        norm_dists = self.actor(states)
        logs_probs = norm_dists.log_prob(actions)
        entropy = norm_dists.entropy().mean()
        
        actor_loss = (-logs_probs*advantage.detach()).mean() - entropy*self.entropy_beta
        self.actor_optim.zero_grad()
        actor_loss.backward()
        
        clip_grad_norm_(self.actor_optim, self.max_grad_norm)
        
        # 將TensorBoard轉為wandb - 梯度直方圖，只在使用wandb時記錄
        if self.use_wandb and wandb.run is not None:
            wandb.log({
                "gradients/actor": wandb.Histogram(
                    torch.cat([p.grad.view(-1) for p in self.actor.parameters()]).detach().cpu().numpy()
                )
            }, step=steps)
            
            # 將TensorBoard轉為wandb - 參數直方圖
            wandb.log({
                "parameters/actor": wandb.Histogram(
                    torch.cat([p.data.view(-1) for p in self.actor.parameters()]).detach().cpu().numpy()
                )
            }, step=steps)
        
        self.actor_optim.step()

        # critic
        critic_loss = F.mse_loss(td_target, value)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_optim, self.max_grad_norm)
        
        # 將TensorBoard轉為wandb - 梯度直方圖，只在使用wandb時記錄
        if self.use_wandb and wandb.run is not None:
            wandb.log({
                "gradients/critic": wandb.Histogram(
                    torch.cat([p.grad.view(-1) for p in self.critic.parameters()]).detach().cpu().numpy()
                )
            }, step=steps)
            
            # 將TensorBoard轉為wandb - 參數直方圖
            wandb.log({
                "parameters/critic": wandb.Histogram(
                    torch.cat([p.data.view(-1) for p in self.critic.parameters()]).detach().cpu().numpy()
                )
            }, step=steps)
            
            # 將TensorBoard轉為wandb - 報告各種指標
            wandb.log({
                "losses/log_probs": -logs_probs.mean().item(),
                "losses/entropy": entropy.item(),
                "losses/entropy_beta": self.entropy_beta,
                "losses/actor": actor_loss.item(),
                "losses/advantage": advantage.mean().item(),
                "losses/critic": critic_loss.item()
            }, step=steps)
        
        self.critic_optim.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def save_checkpoint(self, episode, score=None):
        """保存模型檢查點"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optim.state_dict(),
            'critic_optimizer': self.critic_optim.state_dict(),
            'episode': episode,
        }
        if score is not None:
            checkpoint['score'] = score
            
        checkpoint_path = os.path.join(self.exp_dir, f'checkpoint_ep{episode}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint at episode {episode} to {checkpoint_path}")
        
        # 使用wandb保存模型，只在使用wandb時記錄
        if self.use_wandb and wandb.run is not None:
            artifact = wandb.Artifact(f"model-checkpoint-{episode}", type="model")
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
            
        return checkpoint_path
    
    def save_best_model(self, episode, score):
        """保存最佳模型"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optim.state_dict(),
            'critic_optimizer': self.critic_optim.state_dict(),
            'episode': episode,
            'score': score
        }
        checkpoint_path = os.path.join(self.exp_dir, f'best_model_ep{episode}_score{score:.0f}.pt')
        torch.save(checkpoint, checkpoint_path)
        best_model_path = os.path.join(self.exp_dir, f'best_model.pt')
        torch.save(checkpoint, best_model_path)
        print(f"Saved best model with score {score} at episode {episode}")
        
        # 使用wandb保存最佳模型，只在使用wandb時記錄
        if self.use_wandb and wandb.run is not None:
            artifact = wandb.Artifact("best-model", type="model")
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
            
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """加載模型檢查點"""
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint file not found: {checkpoint_path}")
        
        try:
            # 嘗試使用 weights_only 參數（較新版本 PyTorch）
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            # 舊版 PyTorch 不支持 weights_only 參數
            checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 加載模型參數
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # 加載優化器參數（可選）
        if 'actor_optimizer' in checkpoint and 'critic_optimizer' in checkpoint:
            self.actor_optim.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer'])
            
        episode = checkpoint.get('episode', 0)
        score = checkpoint.get('score', None)
        
        print(f"Loaded checkpoint from episode {episode}" + 
              (f" with score {score}" if score is not None else ""))
        
        return episode, score
    
    def save_config(self, config):
        """保存配置到 YAML 文件"""
        config_path = os.path.join(self.exp_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)


class Runner():
    def __init__(self, env, learner, actor, device):
        self.env = env
        self.state = None
        self.done = True
        self.steps = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.learner = learner
        self.actor = actor
        self.device = device
        self.is_test = False
    
    def reset(self, seed=None):
        self.episode_reward = 0
        self.done = False
        if seed is not None:
            self.state, _ = self.env.reset(seed=seed)
        else:
            self.state, _ = self.env.reset()
    
    def run(self, max_steps, memory=None):
        if not memory: memory = []
        
        for i in range(max_steps):
            if self.done: 
                self.reset()
                
            # print("self.state:", self.state)
            
            dists = self.actor(t(self.state).to(self.device))
            # print("location of dists:", dists.device)

            # print("dists:", dists)
            
            actions = dists.sample().detach().cpu().numpy()
            actions_clipped = np.clip(actions, self.env.action_space.low.min(), self.env.action_space.high.max())
            output = self.env.step(actions_clipped)
            # print("output:", output)
            
            next_state, reward, terminated, truncated, _ = output
            self.done = terminated or truncated
            
            memory.append((actions, reward, self.state, next_state, self.done))

            self.state = next_state
            self.steps += 1
            self.episode_reward += reward
            
            if self.done:
                self.episode_rewards.append(self.episode_reward)
                # print(f"Episode {len(self.episode_rewards)} completed, reward: {self.episode_reward:.2f}")
                if len(self.episode_rewards) % 10 == 0:
                    print(f"Episode {len(self.episode_rewards)} completed, reward: {self.episode_reward:.2f}")
                
                # 將TensorBoard轉為wandb，只在使用wandb時記錄
                if self.learner.use_wandb and wandb.run is not None:
                    wandb.log({"episode_reward": self.episode_reward}, step=self.steps)
                    
        
        return memory
    
    def train(self, episodes, episode_length, steps_on_memory):
        """訓練智能體"""
        self.is_test = False
        best_score = -float('inf')
        
        # 創建結果記錄文件
        result_file = os.path.join(self.learner.exp_dir, 'training_results.csv')
        with open(result_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Score', 'Actor_Loss', 'Critic_Loss'])
        
        total_steps = (episode_length*episodes)//steps_on_memory
        
        # 使用tqdm顯示總體訓練進度
        for i in tqdm(range(total_steps), desc="Training Progress", ncols=100):
            # 使用tqdm顯示重置信息
            if self.done:
                tqdm.write(f"Reset environment, current steps: {self.steps}")
                
            memory = self.run(steps_on_memory)
            actor_loss, critic_loss = self.learner.learn(memory, self.steps, discount_rewards=False)
            
            # 如果有完成的episode
            if len(self.episode_rewards) > 0:
                score = self.episode_rewards[-1]
                episode = len(self.episode_rewards)
                
                # 使用tqdm.write顯示訓練信息
                if episode % 10 == 0:
                    tqdm.write(f"Episode {episode}, Score: {score:.2f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
                
                # 記錄結果
                with open(result_file, 'a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([episode, score, actor_loss, critic_loss])
                
                # 將結果記錄到wandb，只在使用wandb時記錄
                if self.learner.use_wandb and wandb.run is not None:
                    wandb.log({
                        "train/episode": episode,
                        "train/score": score,
                        "train/actor_loss": actor_loss,
                        "train/critic_loss": critic_loss
                    }, step=self.steps)
                
                # 保存檢查點
                if episode % self.learner.save_per_epoch == 0:
                    checkpoint_path = self.learner.save_checkpoint(episode, score)
                    tqdm.write(f"Saved checkpoint: {checkpoint_path}")
                
                # # 保存最佳模型
                # if score > best_score:
                #     best_score = score
                #     best_model_path = self.learner.save_best_model(episode, best_score)
                #     tqdm.write(f"New best model: {best_model_path} (Score: {best_score:.2f})")
                #     if self.learner.use_wandb and wandb.run is not None:
                #         wandb.log({"best_score": best_score, "best_episode": episode}, step=self.steps)
        
        # 訓練結束後保存最終模型
        final_path = self.learner.save_checkpoint(episodes)
        print(f"Training completed. Final model saved at: {final_path}")
        
        # 訓練結束時記錄摘要信息，只在使用wandb時記錄
        if self.learner.use_wandb and wandb.run is not None:
            wandb.run.summary["best_score"] = best_score
            
        return self.episode_rewards
    
    def test(self, checkpoint_path=None, num_episodes=5, seed=42, video_folder=None):
        """測試智能體性能"""
        print("===== 測試開始 =====")
        self.is_test = True
        
        if checkpoint_path:
            try:
                self.learner.load_checkpoint(checkpoint_path)
                # 確保actor和critic指向學習器的模型
                self.actor = self.learner.actor
                print(f"Actor模型已載入並設定為測試模式")
            except Exception as e:
                print(f"載入檢查點時發生錯誤: {e}")
                return []
        
        # 如果要記錄視頻
        tmp_env = self.env
        if video_folder:
            if not os.path.exists(video_folder):
                os.makedirs(video_folder, exist_ok=True)
            
            try:
                # 確保使用適合的參數
                self.env = gym.wrappers.RecordVideo(
                    self.env, 
                    video_folder=video_folder,
                    episode_trigger=lambda x: True  # 錄制所有episode
                )
                print(f"視頻將保存在: {video_folder}")
            except Exception as e:
                print(f"設置視頻記錄時出錯: {e}")
                print("繼續測試但不記錄視頻")
        
        scores = []
        # 使用tqdm顯示測試進度
        for ep in tqdm(range(num_episodes), desc="Testing Progress", ncols=100):
            try:
                self.reset(seed=seed+ep)
                done = False
                score = 0
                steps = 0
                
                while not done:
                    dists = self.actor(t(self.state).to(self.device))
                    actions = dists.sample().detach().cpu().numpy()
                    actions_clipped = np.clip(actions, self.env.action_space.low.min(), self.env.action_space.high.max())
                    next_state, reward, terminated, truncated, _ = self.env.step(actions_clipped)
                    done = terminated or truncated
                    
                    self.state = next_state
                    score += reward
                    steps += 1
                    
                    if done:
                        break
                
                scores.append(score)
                tqdm.write(f"Test Episode {ep+1}/{num_episodes}: Score = {score:.2f}, Steps = {steps}")
                
                # 記錄測試結果到wandb，只在使用wandb時記錄
                if self.learner.use_wandb and wandb.run is not None:
                    wandb.log({
                        "test/episode": ep+1,
                        "test/score": score,
                        "test/steps": steps
                    })
            except Exception as e:
                print(f"測試期間發生錯誤: {e}")
                continue
        
        # 保存結果到CSV
        if video_folder:
            try:
                result_file = os.path.join(video_folder, 'test_results.csv')
                with open(result_file, 'w', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(['Episode', 'Score'])
                    for i, score in enumerate(scores):
                        csv_writer.writerow([i+1, score])
                        
                # 保存統計摘要
                summary_file = os.path.join(video_folder, 'test_summary.json')
                summary = {
                    'mean_score': float(np.mean(scores)) if scores else 0,
                    'std_score': float(np.std(scores)) if len(scores) > 1 else 0,
                    'min_score': float(np.min(scores)) if scores else 0,
                    'max_score': float(np.max(scores)) if scores else 0,
                    'median_score': float(np.median(scores)) if scores else 0,
                    'num_episodes': num_episodes,
                    'completed_episodes': len(scores)
                }
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=4)
                
                # 將測試摘要記錄到wandb，只在使用wandb時記錄
                if self.learner.use_wandb and wandb.run is not None:
                    wandb.log(summary)
                    
                    # 上傳測試結果文件
                    wandb.save(result_file)
                    wandb.save(summary_file)
                
                print(f"測試結果已保存到: {result_file}")
                print(f"測試摘要已保存到: {summary_file}")
            except Exception as e:
                print(f"保存結果時發生錯誤: {e}")
            
            # 關閉記錄環境
            try:
                self.env.close()
                self.env = tmp_env
            except:
                print("關閉環境時出錯，已忽略")
        
        if scores:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"Test Summary: Mean Score = {mean_score:.2f} ± {std_score:.2f}")
            
            # 將測試摘要記錄到wandb，只在使用wandb時記錄
            if self.learner.use_wandb and wandb.run is not None:
                wandb.run.summary["test_mean_score"] = mean_score
                wandb.run.summary["test_std_score"] = std_score
        else:
            print("沒有完成任何測試回合")
        
        return scores


env = gym.make("Pendulum-v1")

# config
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

# 注意：這裡僅初始化基本模型，實際使用時會從命令列參數設置
# 默認使用 Tanh 激活函數
actor = Actor(state_dim, n_actions).to(device)
critic = Critic(state_dim).to(device)

# 參數設定
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--num-episodes", type=int, default=3300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entropy-beta", type=float, default=1e-2)
    parser.add_argument("--save-per-epoch", type=int, default=100)
    parser.add_argument("--result-dir", type=str, default="result-a2c_pendulum")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--checkpoint", type=str, help="checkpoint path for testing")
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--steps-on-memory", type=int, default=16)
    parser.add_argument("--wandb-project", type=str, default="a2c-pendulum")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--use-wandb", action="store_true", help="使用 wandb 記錄訓練過程")
    parser.add_argument("--use-mish", action="store_true", help="使用 Mish 激活函數代替 Tanh")
    
    args = parser.parse_args()
    
    # 設置隨機種子
    # 創建環境和Agent
    if args.mode == "test":
        # 測試模式下添加渲染
        seed = args.seed
        np.random.seed(seed)
        seed_torch(seed)
    
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
    else:
        env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    
    # 使用 Mish 激活函數，如果指定
    activation = Mish if args.use_mish else nn.Tanh
    
    actor = Actor(state_dim, n_actions, activation=activation).to(device)
    critic = Critic(state_dim, activation=activation).to(device)
    
    # 初始化wandb，只在訓練模式且指定使用wandb時啟動
    if args.mode == "train" and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "actor_lr": args.actor_lr,
                "critic_lr": args.critic_lr,
                "gamma": args.gamma,
                "entropy_beta": args.entropy_beta,
                "num_episodes": args.num_episodes,
                "episode_length": args.episode_length,
                "steps_on_memory": args.steps_on_memory,
                "seed": args.seed,
                "model": "A2C",
                "env": "Pendulum-v1",
                "activation": "Mish" if args.use_mish else "Tanh"
            }
        )
    
    learner = A2CLearner(
        actor, 
        critic, 
        gamma=args.gamma,
        entropy_beta=args.entropy_beta,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        save_per_epoch=args.save_per_epoch,
        result_dir=args.result_dir,
        mode=args.mode,
        use_wandb=args.use_wandb
    )
    
    runner = Runner(env, learner, actor, device)
    
    if args.mode == "train":
        # 訓練模式
        try:
            runner.train(
                episodes=args.num_episodes,
                episode_length=args.episode_length,
                steps_on_memory=args.steps_on_memory
            )
        finally:
            # 確保結束時關閉wandb
            if args.use_wandb and wandb.run is not None:
                wandb.finish()
    else:
        # 測試模式
        if not args.checkpoint:
            raise ValueError("In test mode, --checkpoint argument is required")
        
        # 創建測試輸出目錄
        checkpoint_name = os.path.basename(args.checkpoint).split('.')[0]
        test_dir = os.path.join(os.path.dirname(args.checkpoint), f"test_{checkpoint_name}")
        
        # 測試模式下不啟動wandb
        runner.test(
            checkpoint_path=args.checkpoint,
            num_episodes=args.test_episodes,
            seed=args.seed,
            video_folder=test_dir
        )