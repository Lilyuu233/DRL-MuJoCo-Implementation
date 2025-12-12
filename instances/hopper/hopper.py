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

import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.optim import Adam

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def set_flat_params_to(params, flat_params):
    prev_ind = 0
    for param in params:
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

# 1. 调整网络结构以匹配 Hopper 参数
class MlpActorCritic(torch.nn.Module):
    def __init__(self, obs_space, act_space):
        super().__init__()
        self.obs_dim = obs_space.shape[0]
        # 将隐藏层大小改为 50
        hidden_size = 50
        self.actor_mean = torch.nn.Sequential(
            torch.nn.Linear(self.obs_dim, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, act_space.shape[0])
        )
        self.actor_log_std = torch.nn.Parameter(torch.zeros(act_space.shape[0]))
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(self.obs_dim, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1)
        )

    def forward(self, obs):
        obs = obs.view(-1, self.obs_dim)
        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_log_std)
        dist = torch.distributions.Normal(mean, std)
        value = self.critic(obs).squeeze(-1)
        return dist, value

class Runner:
    def __init__(self, env, model, nsteps, gamma, lam, device):
        self.env = env
        self.model = model
        self.nsteps = nsteps
        self.gamma = gamma
        self.lam = lam
        self.device = device
        self.obs = torch.zeros((self.nsteps + 1,) + self.env.observation_space.shape).to(device)
        self.actions = torch.zeros((self.nsteps,) + self.env.action_space.shape).to(device)
        self.rewards = torch.zeros(self.nsteps).to(device)
        self.dones = torch.zeros(self.nsteps).to(device)
        self.values = torch.zeros(self.nsteps + 1).to(device)
        self.current_obs, _ = self.env.reset()

    def run(self):
        epinfos = []
        for step in trange(self.nsteps, desc="Sampling", leave=False):
            current_obs_tensor = torch.tensor(self.current_obs, dtype=torch.float32).to(self.device)
            self.obs[step] = current_obs_tensor
            with torch.no_grad():
                dist, value = self.model(current_obs_tensor.unsqueeze(0))
                action = dist.sample()
            self.values[step] = value.squeeze(0)
            self.actions[step] = action.squeeze(0)
            action_np = action.squeeze(0).cpu().numpy()
            obs_next, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            self.rewards[step] = torch.tensor(reward, dtype=torch.float32).to(self.device)
            self.dones[step] = torch.tensor(done).to(self.device)
            self.current_obs = obs_next
            if done:
                epinfos.append({"r": info.get("episode", {}).get("r", 0), "l": info.get("episode", {}).get("l", 0)})
                self.current_obs, _ = self.env.reset()
        
        final_obs_tensor = torch.tensor(self.current_obs, dtype=torch.float32).to(self.device)
        self.obs[self.nsteps] = final_obs_tensor
        with torch.no_grad():
            _, last_value = self.model(final_obs_tensor.unsqueeze(0))
        self.values[self.nsteps] = last_value.squeeze(0)

        advs = torch.zeros_like(self.rewards)
        last_gae_lam = 0
        for t in reversed(range(self.nsteps)):
            delta = self.rewards[t] + self.gamma * self.values[t + 1] * (1.0 - self.dones[t]) - self.values[t]
            advs[t] = last_gae_lam = delta + self.gamma * self.lam * (1.0 - self.dones[t]) * last_gae_lam
        returns = advs + self.values[:-1]
        return self.obs[:-1], self.actions, advs, returns, epinfos

def learn(algo, actor_critic, env, device, log_dir, env_name,
          total_timesteps, nsteps, nminibatches, lam, gamma, noptepochs, lr, ent_coef, max_kl, log_interval):
    num_envs = 1
    timesteps_per_epoch = nsteps * num_envs
    epochs = total_timesteps // timesteps_per_epoch
    minibatch_size = timesteps_per_epoch // nminibatches
    runner = Runner(env=env, model=actor_critic, nsteps=nsteps, gamma=gamma, lam=lam, device=device)
    epinfobuf = deque(maxlen=100)

    def conjugate_gradient(fvp_fn, g, x, nsteps_cg=10, residual_tol=1e-10):
        r = g - fvp_fn(x)
        p = r.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps_cg):
            z = fvp_fn(p)
            alpha = rdotr / (torch.dot(p, z) + 1e-8)
            x += alpha * p
            r -= alpha * z
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
        return x

    tepochs = trange(epochs, desc='Epoch', leave=True)
    params_pi = list(actor_critic.actor_mean.parameters()) + [actor_critic.actor_log_std]
    params_v = list(actor_critic.critic.parameters())
    v_optimizer = Adam(params_v, lr=lr)
    make_flat = lambda x: torch.cat([p.contiguous().view(-1) for p in x if p is not None])
    get_flat_grad = lambda params: torch.cat([p.grad.contiguous().view(-1) for p in params if p.grad is not None])
    x_cg = None

    epoch_history = []
    rewards_history = []

    for epoch in tepochs:
        tstart = time.perf_counter()
        actor_critic.eval()
        obs, act, adv, ret, epinfos = runner.run()
        epinfobuf.extend(epinfos)
        with torch.no_grad():
            old_dist, _ = actor_critic(obs)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        def get_policy_loss_and_kl(current_dist):
            logp = current_dist.log_prob(act).sum(-1)
            old_logp = old_dist.log_prob(act).sum(-1)
            ratio = torch.exp(logp - old_logp)
            pi_loss = -(ratio * adv).mean()
            kl = torch.distributions.kl.kl_divergence(old_dist, current_dist).mean()
            return pi_loss, kl

        def fisher_vector_product(x):
            damping = 0.1
            new_dist, _ = actor_critic(obs)
            _, kl = get_policy_loss_and_kl(new_dist)
            kl_grads = torch.autograd.grad(kl, params_pi, create_graph=True)
            flat_kl_grads = make_flat(kl_grads)
            dot_prod = torch.dot(flat_kl_grads, x)
            fvp = torch.autograd.grad(dot_prod, params_pi)
            flat_fvp = make_flat(fvp)
            return flat_fvp + damping * x
        
        new_dist_for_grad, _ = actor_critic(obs)
        pi_loss_for_grad, _ = get_policy_loss_and_kl(new_dist_for_grad)
        entropy = new_dist_for_grad.entropy().mean()
        actor_critic.zero_grad()
        (pi_loss_for_grad - ent_coef * entropy).backward()
        g = get_flat_grad(params_pi).detach()
        if x_cg is None:
            x_cg = torch.zeros_like(g)
        step_dir = conjugate_gradient(fisher_vector_product, -g, x=x_cg).detach()
        x_cg = step_dir.clone()
        shs = 0.5 * torch.dot(step_dir, fisher_vector_product(step_dir))
        lm = torch.sqrt(shs / max_kl)
        full_step = step_dir / (lm + 1e-8)
        step_size = 1.0
        flat_params = make_flat(params_pi).detach()
        with torch.no_grad():
            pi_loss_old, _ = get_policy_loss_and_kl(old_dist)
        for _ in range(10):
            new_params = flat_params + full_step * step_size
            set_flat_params_to(params_pi, new_params)
            with torch.no_grad():
                new_dist, _ = actor_critic(obs)
                pi_loss_new, kl_new = get_policy_loss_and_kl(new_dist)
            if kl_new <= max_kl and pi_loss_new <= pi_loss_old:
                break
            step_size *= 0.5
        else:
            set_flat_params_to(params_pi, flat_params)

        actor_critic.train()
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

        tnow = time.perf_counter()
        fps = int(timesteps_per_epoch / (tnow - tstart))
        if (epoch + 1) % log_interval == 0:
            mean_reward = safemean([epinfo['r'] for epinfo in epinfobuf])
            
            print(f"--- Epoch: {epoch} ---")
            print(f"Total Timesteps: {(epoch + 1) * timesteps_per_epoch}")
            print(f"Mean Reward (last 100 episodes): {mean_reward:.2f}")
            print(f"FPS: {fps}")
            print("-" * (15 + len(str(epoch))))

            epoch_history.append(epoch)
            rewards_history.append(mean_reward)
    
    if log_dir is not None:
        torch.save({'model_state_dict': actor_critic.state_dict()}, f'{log_dir}/model.ckpt')

    if epoch_history and rewards_history:
        print("\nTraining finished. Generating plot...")
        try:
            plt.style.use('ggplot')
            plt.figure(figsize=(10, 6.5))
            plt.plot(epoch_history, rewards_history, label='TRPO')
            plt.title(f'Training on {env_name}')
            plt.xlabel('Number of Policy Iterations')
            plt.ylabel('Reward')
            plt.legend(loc='lower right')
            plt.show()
        except Exception as e:
            print(f"Could not generate plot. Error: {e}")
            plt.style.use('default')
            plt.figure(figsize=(10, 6.5))
            plt.plot(epoch_history, rewards_history, label='TRPO')
            plt.title(f'Training on {env_name}')
            plt.xlabel('Number of Policy Iterations')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.legend(loc='lower right')
            plt.show()

def train_fn(device_id):
    algo = 'trpo'
    # 2.更新环境名称
    env_name = 'Hopper-v5' 
    
    # 3.调整训练步数以匹配 Hopper 参数
    # 论文参数: 200 policy iterations, 1M sim steps per iter.
    policy_iterations = 200
    nsteps = 1000000        # Sim. steps per iter.
    total_timesteps = policy_iterations * nsteps # = 200,000,000
    
    # 保持不变的参数
    nminibatches = 32
    gamma = 0.99    
    lam = 0.95
    noptepochs = 10
    lr = 3e-4
    ent_coef = 0.0
    max_kl = 0.01    
    log_interval = 1
    
    time_now = datetime.now().strftime('%Y%m%d-%H%M%S')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    log_dir = f"logs/{algo}_{env_name}_{time_now}_{seed}"
    os.makedirs(log_dir, exist_ok=True)
    
    print("创建环境中...")
    env = gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    print(f'运行设备: {device}')

    actor_critic = MlpActorCritic(env.observation_space, env.action_space).to(device)
    
    kwargs = {k: v for k, v in locals().items() if k not in [
        'env', 'actor_critic', 'kwargs', 'device_id', 'log_dir', 'time_now', 'device'
    ]}
    with open(f"{log_dir}/hyperparameters.json", 'w') as f:
        json.dump(kwargs, f, indent=4)
    
    learn(algo=algo, actor_critic=actor_critic, env=env, device=device, log_dir=log_dir, env_name=env_name,
          total_timesteps=total_timesteps, nsteps=nsteps, nminibatches=nminibatches, 
          lam=lam, gamma=gamma, noptepochs=noptepochs, lr=lr, 
          ent_coef=ent_coef, max_kl=max_kl, log_interval=log_interval)

def main():
    # 4. 更新命令行描述
    parser = argparse.ArgumentParser(description='Train TRPO for Hopper-v5 based on paper parameters.')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID to use.')
    args = parser.parse_args()
    train_fn(args.device)

if __name__ == '__main__':
    main()