import os
import time
import json
import torch
import argparse
import numpy as np
import gymnasium as gym
from tqdm import trange
from datetime import datetime
from collections import deque
import utils.logger as logger

from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from utils.utils import count_vars, safemean, set_flat_params_to

# 定义一个包含演员（Actor）和评判家（Critic）的多层感知机（MLP）网络
class MlpActorCritic(torch.nn.Module):
    def __init__(self, obs_space, act_space):
        super().__init__()
        self.obs_dim = obs_space.shape[0]

        # 演员网络，用于预测动作的均值
        self.actor_mean = torch.nn.Sequential(
            torch.nn.Linear(self.obs_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, act_space.shape[0])
        )
        
        # 我们将标准差的对数作为一个可学习的独立参数
        # 这样做可以确保标准差始终为正，并且在优化过程中更稳定
        self.actor_log_std = torch.nn.Parameter(torch.zeros(act_space.shape[0]))

        # 评判家网络，用于估计状态的价值
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(self.obs_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, obs):
        """
        定义网络的前向传播。
        输入一个观测(obs),输出一个动作分布(dist)和状态价值(value)
        """

        # 确保输入形状正确
        obs = obs.view(-1, self.obs_dim)
        
        # 从演员网络获取均值
        mean = self.actor_mean(obs)
        # 对 log_std 求指数得到标准差
        std = torch.exp(self.actor_log_std)
        # 创建一个正态（高斯）分布对象，用于后续的动作采样和概率计算
        dist = torch.distributions.Normal(mean, std)

        # 从评判家网络获取价值
        value = self.critic(obs).squeeze(-1) # squeeze(-1)是为了移除多余的维度

        return dist, value

# Runner类的作用是在环境中执行当前的策略，并收集经验数据
class Runner:
    def __init__(self, env, model, nsteps, gamma, lam, device):
        self.env = env           # Gym 环境
        self.model = model       # MlpActorCritic 模型
        self.nsteps = nsteps     # 每次运行要收集的步数
        self.gamma = gamma       # 折扣因子
        self.lam = lam           # GAE(广义优势估计)的lambda系数
        self.device = device     # 运行设备 (CPU或GPU)
        
        # 初始化用于存储经验数据的张量
        self.obs = torch.zeros((self.nsteps + 1,) + self.env.observation_space.shape).to(device)
        self.actions = torch.zeros((self.nsteps,) + self.env.action_space.shape).to(device)
        self.rewards = torch.zeros(self.nsteps).to(device)
        self.dones = torch.zeros(self.nsteps).to(device)
        self.values = torch.zeros(self.nsteps + 1).to(device)
        
        # 获取初始观测
        self.current_obs, _ = self.env.reset()

    def run(self):
        """
        执行nsteps的模拟,收集数据并计算优势和回报
        """

        epinfos = [] # 存储每个回合结束时的信息（如奖励、长度）

        # --- 1. 与环境交互，收集 nsteps 的数据 ---
        for step in range(self.nsteps):
            self.obs[step] = torch.tensor(self.current_obs).to(self.device)
            
            with torch.no_grad():
                # 为模型添加批次维度
                obs_tensor = torch.tensor(self.current_obs).unsqueeze(0).to(self.device)
                dist, value = self.model(obs_tensor)
                action = dist.sample() # 从动作分布中采样一个动作
            
            # 存储不带批次维度的值和动作
            self.values[step] = value.squeeze(0)
            self.actions[step] = action.squeeze(0)

            # 与环境交互,让环境执行动作，并获得反馈
            action_np = action.squeeze(0).cpu().numpy()
            obs_next, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            # 存储结果
            self.rewards[step] = torch.tensor(reward).to(self.device)
            self.dones[step] = torch.tensor(done).to(self.device)
            self.current_obs = obs_next

            if done:
                # 从 info 字典中提取回合统计信息
                epinfos.append({
                    "r": info.get("episode", {}).get("r", 0), 
                    "l": info.get("episode", {}).get("l", 0)
                })
                self.current_obs, _ = self.env.reset()

        # 存储最后一步的观测，用于计算GAE
        self.obs[self.nsteps] = torch.tensor(self.current_obs).to(self.device)
        
        # --- 2. 计算优势（Advantages）和回报（Returns） ---
        # 计算最后一步的状态价值，用于GAE的起始计算
        with torch.no_grad():
            obs_tensor = torch.tensor(self.current_obs).unsqueeze(0).to(self.device)
            _, last_value = self.model(obs_tensor)
        self.values[self.nsteps] = last_value.squeeze(0)
        
        # 使用广义优势估计（GAE）计算优势函数
        advs = torch.zeros_like(self.rewards)
        last_gae_lam = 0

        # 从后向前遍历所有时间步
        for t in reversed(range(self.nsteps)):
            delta = self.rewards[t] + self.gamma * self.values[t + 1] * (1.0 - self.dones[t]) - self.values[t]
            advs[t] = last_gae_lam = delta + self.gamma * self.lam * (1.0 - self.dones[t]) * last_gae_lam
        
        returns = advs + self.values[:-1] # 计算回报（Returns），即优势加上状态价值

        # 数据已经是展平的批次
        return self.obs[:-1], self.actions, advs, returns, epinfos

# TRPO 学习函数
def learn(algo, actor_critic, writer, env, device, log_dir, 
          total_timesteps, nsteps, nminibatches, lam, gamma, noptepochs, lr, ent_coef, max_kl, log_interval):

    # --- 初始化 ---
    num_envs = 1 # 我们现在只有一个环境
    timesteps_per_epoch = nsteps * num_envs
    epochs = total_timesteps // timesteps_per_epoch

    minibatch_size = timesteps_per_epoch // nminibatches

    # 实例化 Runner
    runner = Runner(env=env, model=actor_critic, nsteps=nsteps, gamma=gamma, lam=lam, device=device)
    epinfobuf = deque(maxlen=100)

    # --- 共轭梯度法 (Conjugate Gradient) ---
    # 共轭梯度法
    # 这是TRPO的核心，用于高效求解 Ax=g，其中A是Fisher信息矩阵，g是梯度
    # 它避免了直接计算和求逆巨大的Hessian矩阵A
    def conjugate_gradient(fvp_fn, g, nsteps_cg=10, residual_tol=1e-10):
        # fvp_fn 就是计算 H*p 的函数
        # g 是梯度
    
        # 1. 初始化
        x = torch.zeros_like(g) # 初始化解 x 。1.改成用上一次迭代的结果，2.结果plot为reward，展示修改之后的和修正版, 3.应用在其他环境中。
        r = g.clone()           # 残差 r 初始化为梯度 g
        p = r.clone()           # 搜索方向 p 初始化为梯度 g
        rdotr = torch.dot(r, r) # 残差的模平方

        # 2. 迭代求解 (最多迭代10次)
        for i in range(nsteps_cg):
            # --- 关键计算步骤 ---
            z = fvp_fn(p) # 计算 H * p，这是整个算法中最核心的计算，也是与矩阵H唯一的交互
        
            # 计算本次迭代的最佳步长 alpha
            # alpha 的含义是：沿着当前搜索方向 p 走多远，能最大程度地减小残差 r
            alpha = rdotr / (torch.dot(p, z) + 1e-8)
        
            # 更新解 x 和残差 r
            x += alpha * p # 沿着搜索方向 p，前进 alpha 步，更新我们的解
            r -= alpha * z # 更新残差。可以证明，这样更新后，新的残差与旧的残差正交
        
            # --- 准备下一次迭代 ---
            new_rdotr = torch.dot(r, r)
        
            # 如果残差已经足够小，说明 x 已经很接近真实解了，可以提前结束
            if new_rdotr < residual_tol:
                break
            
            # 计算 beta，用来确定下一个搜索方向
            # beta 的作用是修正梯度方向，使得新的搜索方向 p 与之前的方向“共轭”，
            # 保证每次都在一个全新的、不重复的维度上进行搜索，从而提高效率。
            beta = new_rdotr / (rdotr + 1e-8)
        
            # 更新搜索方向 p
            p = r + beta * p # 新的搜索方向是新的残差方向，加上一个被beta缩放的旧搜索方向
        
            # 更新残差的模
            rdotr = new_rdotr
            
        return x # 近似的 A^-1 g，这就是策略更新的方向

    # --- 训练准备 ---
    tfirststart = time.perf_counter()
    tepochs = trange(epochs + 1, desc='Epoch', leave=True)

    # 分别获取策略网络（演员）和价值网络（评判家）的参数
    params_pi = list(actor_critic.actor_mean.parameters()) + [actor_critic.actor_log_std]
    params_v = list(actor_critic.critic.parameters())
    v_optimizer = Adam(params_v, lr=lr) # 价值网络的优化器，使用Adam

    # 辅助函数：将参数列表展平成一个一维向量
    make_flat = lambda x: torch.cat([p.contiguous().view(-1) for p in x if p is not None])
    # 辅助函数：获取参数列表的展平梯度
    get_flat_grad = lambda params: torch.cat([p.grad.contiguous().view(-1) for p in params if p.grad is not None])

    # --- 主训练循环 ---
    for epoch in tepochs:
        tstart = time.perf_counter()
        actor_critic.eval()  # 将模型设置为评估模式，用于数据收集
        
        # --- 1. 收集数据 ---
        obs, act, adv, ret, epinfos = runner.run()
        epinfobuf.extend(epinfos)
        
        # --- 2. 策略更新 (Policy Update) ---
        with torch.no_grad():
            old_dist, _ = actor_critic(obs) # 获取旧策略的动作分布
            
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) # 标准化优势函数，可以稳定训练

         # 定义一个函数，用于计算策略损失和KL散度
        def get_policy_loss_and_kl(current_dist):
            logp = current_dist.log_prob(act).sum(-1) # 新策略下动作的对数概率
            old_logp = old_dist.log_prob(act).sum(-1) # 旧策略下动作的对数概率
            ratio = torch.exp(logp - old_logp) # 重要性采样比率
            pi_loss = -(ratio * adv).mean() # 策略损失（目标函数）
            kl = torch.distributions.kl.kl_divergence(old_dist, current_dist).mean() # 新旧策略的KL散度
            return pi_loss, kl

        # 定义 Fisher-向量积 (FVP) 函数 F*x   
        def fisher_vector_product(x):
            damping = 0.1 # 阻尼项，增加数值稳定性
            new_dist, _ = actor_critic(obs)
            _, kl = get_policy_loss_and_kl(new_dist)
            
            # 计算KL散度关于策略参数的梯度
            kl_grads = torch.autograd.grad(kl, params_pi, create_graph=True)
            flat_kl_grads = make_flat(kl_grads)
            
            # 计算 kl_grads 和输入向量 x 的点积
            dot_prod = torch.dot(flat_kl_grads, x)
            fvp = torch.autograd.grad(dot_prod, params_pi) # 再次求导，得到 FVP
            flat_fvp = make_flat(fvp)
            
            return flat_fvp + damping * x
        
        # --- a. 计算策略梯度 g ---
        new_dist_for_grad, _ = actor_critic(obs)
        pi_loss_for_grad, _ = get_policy_loss_and_kl(new_dist_for_grad)
        entropy = new_dist_for_grad.entropy().mean()
        
        actor_critic.zero_grad()
        (pi_loss_for_grad - ent_coef * entropy).backward()
        g = get_flat_grad(params_pi).detach()
        
        # --- b. 使用共轭梯度法计算搜索方向 step_dir (Hx = -g) ---
        step_dir = conjugate_gradient(fisher_vector_product, -g).detach()
        
        # --- c. 回溯线性搜索 (Backtracking Line Search) 寻找最佳步长 ---
        # 计算理论上的最大步长，使得KL散度恰好等于 max_kl
        shs = 0.5 * torch.dot(step_dir, fisher_vector_product(step_dir)) # 1/2 Δθ^T H Δθ
        lm = torch.sqrt(shs / max_kl) # 计算缩放因子，确保如果走完整个step_dir，KL散度恰好等于max_kl
        full_step = step_dir / (lm + 1e-8) # 在理论上能走的最大一步
        
        step_size = 1.0 # 从最大步长 step_size = 1.0 (对应 full_step) 开始尝试
        flat_params = make_flat(params_pi).detach()
        
        with torch.no_grad():
            pi_loss_old, _ = get_policy_loss_and_kl(old_dist)

        # 循环最多10次，不断缩小步长，直到找到满足条件的更新
        for _ in range(10):
            new_params = flat_params + full_step * step_size
            set_flat_params_to(params_pi, new_params)
            
            with torch.no_grad():
                new_dist, _ = actor_critic(obs)
                pi_loss_new, kl_new = get_policy_loss_and_kl(new_dist)

            # 检查两个条件：1. KL散度在约束内 2. 策略损失有改善
            if kl_new <= max_kl and pi_loss_new <= pi_loss_old:
                break
            step_size *= 0.5
        else:
            set_flat_params_to(params_pi, flat_params)

        # --- 3. 价值函数更新 (Value Function Update) ---
        actor_critic.train() # 将模型设置为训练模式
        inds = np.arange(timesteps_per_epoch)
        for _ in range(noptepochs):
            np.random.shuffle(inds)
            for start in range(0, timesteps_per_epoch, minibatch_size):
                end = start + minibatch_size
                mbinds = inds[start:end]
                
                _, values = actor_critic(obs[mbinds])
                v_loss = F.mse_loss(values, ret[mbinds])
                
                v_optimizer.zero_grad()
                v_loss.backward()
                v_optimizer.step()

        # --- 4. 日志记录 ---
        tnow = time.perf_counter()
        fps = int(timesteps_per_epoch / (tnow - tstart))

        if logger.get_dir() is not None and (epoch + 1) % log_interval == 0:
            logger.logkv("misc/epoch", epoch)
            logger.logkv("misc/total_timesteps", (epoch + 1) * timesteps_per_epoch)
            logger.logkv("fps", fps)
            logger.logkv("entropy", entropy.item())
            logger.logkv("kl", kl_new.item())
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('misc/time_elapsed', tnow - tfirststart)
            logger.dumpkvs()
    
    if log_dir is not None:
        torch.save({'model_state_dict': actor_critic.state_dict()}, f'{log_dir}/model.ckpt')

def train_fn(device_id):
    algo = 'trpo'
    env_name = 'MountainCarContinuous-v0'
    total_timesteps = 100_000
    nsteps = 2048
    nminibatches = 4
    gamma = 0.99
    lam = 0.95
    noptepochs = 10
    lr = 3e-4
    ent_coef = 0.0
    max_kl = 0.01
    log_interval = 1

    time_now = datetime.now().strftime('%Y%m%d-%H%M%S')
    seed = np.random.randint(1e6)
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = f"logs/{algo}.mlp/{env_name}.{time_now}_{seed}"
    logger.configure(dir=log_dir, format_strs=['csv', 'stdout'])
    writer = SummaryWriter(log_dir=log_dir)
    
    logger.info("创建环境中...")
    # +++ 使用标准的、非向量化的环境 +++
    env = gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env) # 替代 VecMonitor

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    logger.info(f'运行设备: {device}')

    actor_critic = MlpActorCritic(env.observation_space, env.action_space).to(device)

    var_counts = count_vars(actor_critic)
    logger.log(f'网络参数数量: {var_counts}')

    kwargs = dict(locals())
    json.dump(kwargs, open(f"{log_dir}/kwargs.json", 'w'), default=lambda o: '<not serializable>')
    
    learn(algo, actor_critic, writer, env, device, log_dir,
          total_timesteps=total_timesteps, nsteps=nsteps, nminibatches=nminibatches, 
          lam=lam, gamma=gamma, noptepochs=noptepochs, lr=lr, 
          ent_coef=ent_coef, max_kl=max_kl, log_interval=log_interval)

def main():
    parser = argparse.ArgumentParser(description='Train TRPO for MountainCarContinuous.')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID to use.')
    args = parser.parse_args()
    train_fn(args.device)

if __name__ == '__main__':
    main()
