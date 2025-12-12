from procgen import ProcgenEnv

import os
import time
import json
import torch
import argparse
import numpy as np
from tqdm import trange
from datetime import datetime
from collections import deque
import utils.logger as logger

from torch.nn import functional as F

# pytorch distributed training
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    TPU_MODE = True

except ImportError:
    TPU_MODE = False

from torch.optim import Adam
from utils.runners import Runner
from torch.utils.tensorboard import SummaryWriter

from utils.utils import build_cnn, build_resnet, build_vit, build_pretrained
from utils.utils import SeparateActorCritic, count_vars, safemean
from utils.utils import set_flat_params_to

from vec_env import ( VecExtractDictObs, VecMonitor, VecNormalize)

def learn(world_size, algo, actor_critic, writer, venv, eval_venv, device, log_dir, 
          total_timesteps, nsteps, nminibatches, lam, gamma, noptepochs, lr, ent_coef, log_interval):

    per_epoch_timesteps = nsteps * venv.num_envs
    epochs = total_timesteps // per_epoch_timesteps + 1

    minibatch_size = per_epoch_timesteps // nminibatches

    # Instantiate the runner object
    runner = Runner(env=venv, model=actor_critic, nsteps=nsteps, gamma=gamma, lam=lam, device=device)
    if eval_venv is not None:
        eval_runner = Runner(env=eval_venv, model=actor_critic, nsteps=nsteps, gamma=gamma, lam=lam, device=device, test_mode=True)

    epinfobuf = deque(maxlen=100)
    if eval_venv is not None:
        eval_epinfobuf = deque(maxlen=100)

    def conjugate_gradient(fn_fvp, g, nsteps=10, residual_tol=1e-10):
        x = torch.zeros_like(g) 
        r = g.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)

        sFs = 0.0
        for i in range(nsteps):
            z = fn_fvp(p)
            alpha = rdotr / torch.dot(p, z)
            sFs += 0.5 * alpha * rdotr
            x += alpha * p
            r -= alpha * z
            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x, sFs

    # Start total timer
    tfirststart = time.perf_counter()
    tepochs = trange(epochs+1, desc='Epoch starts', leave=True)

    def compute_loss(_obs, _act, _adv, _logp_old):
        # pi loss
        _logits = actor_critic.logits_net(_obs)
        _logp_all = F.log_softmax(_logits, dim=-1)
        _kl = (-torch.exp(_logp_all.detach()) * _logp_all ).sum(dim=-1).mean() # no need to use full kl

        _logp = torch.gather(_logp_all, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
        _ratio = torch.exp(_logp - _logp_old)
        _entropy = torch.mean((-torch.exp(_logp_all) * _logp_all).sum(-1))
        _loss_pi = (- _ratio * _adv).mean()

        return _kl, _loss_pi, _entropy, _logp_all

    make_flat = lambda x:  torch.cat([grad.contiguous().view(-1) for grad in x if grad is not None])
    get_flat_grad = lambda params:  torch.cat([p.grad.contiguous().view(-1) for p in params if p.grad is not None])

    # Main loop: collect experience in env and update/log each epoch
    inds = np.arange(per_epoch_timesteps)
    params_pi = list(actor_critic.logits_net.parameters())
    params_v = list(actor_critic.v_net.parameters())

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(params_pi, lr=lr)
    v_optimizer = Adam(params_v, lr=lr)

    for epoch in tepochs:
        tstart = time.perf_counter()

        tepochs.set_description('Stepping environment...')

        actor_critic.eval() # set to eval mode for PPO
        obs, ret, act, adv, logp_old, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)

        tepochs.set_description('Minibatch training...')
        max_kl = 0.01

        # actor update
        slices = (obs, act, adv, logp_old)
        actor_critic.eval() # need to set to eval mode otherwise trust region is violated
        kl, mb_pi_loss, entropy, logp_all_old = compute_loss(*slices)

        def fisher_vector_product(x):
            damping = 0.01
            kl_grad = torch.autograd.grad(kl, params_pi, create_graph=True)
            kl_grad_flat = make_flat(kl_grad)
            dot_prod = torch.dot(kl_grad_flat, x)
            fvp = torch.autograd.grad(dot_prod, params_pi, retain_graph=True)
            fvp_flat = make_flat(fvp)
            return fvp_flat + damping * x

        # udpate actor
        pi_optimizer.zero_grad()
        mb_loss = mb_pi_loss - ent_coef * entropy
        mb_loss.backward(retain_graph=True)
        loss_grad_pi_flat = get_flat_grad(params_pi).detach()

        step_dir, sFs = conjugate_gradient(fisher_vector_product, loss_grad_pi_flat, nsteps=10)
        assert not torch.isnan(step_dir).all()
        lm = 1 / torch.sqrt(sFs / max_kl)
        assert not torch.isnan(lm).all()
        full_step = lm * step_dir

        # trust region line search:
        step_size = 1.0 
        params = make_flat(params_pi)
        actor_critic.eval()
        for _ in range(10):
            new_params = params - full_step * step_size
            set_flat_params_to(params_pi, new_params)

            d_logits = actor_critic.logits_net(obs)
            d_logp_all = F.log_softmax(d_logits, dim=-1)
            kl = (torch.exp(logp_all_old) * (logp_all_old - d_logp_all)).sum(dim=-1).mean()

            if kl > max_kl * 1.5:
                logger.log("violated KL constraint. shrinking step.")
            else:
                break
            step_size *= .5
        else:
            logger.log("couldn't compute a good step")
            set_flat_params_to(params_pi, params)

        del params, step_dir, full_step, loss_grad_pi_flat

        # critic update
        actor_critic.train()
        for _ in range(noptepochs):
            # Randomize the indexes
            np.random.shuffle(inds)

            # 0 to batch_size with batch_train_size step
            for start in range(0, per_epoch_timesteps, minibatch_size):
                end = start + minibatch_size
                mbinds = inds[start:end]
                _vals = actor_critic.v_net(obs[mbinds])
                # value loss
                mb_v_loss = F.mse_loss(_vals, ret[mbinds])
                v_optimizer.zero_grad()
                mb_v_loss.backward()
                v_optimizer.step()
                tepochs.set_postfix(loss_pi=mb_pi_loss.item(), loss_v=mb_v_loss.item(), entropy=entropy.item(), kl=kl.item())

        # clean GPU cache
        del obs, ret, act, adv
        torch.cuda.empty_cache()

        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(per_epoch_timesteps / (tnow - tstart))

        if eval_venv is not None and (epoch+1) % log_interval == 0:
            logger.info('Testing...')
            actor_critic.eval() # set to eval mode for PPO
            eval_epinfos = eval_runner.run() 
            eval_epinfobuf.extend(eval_epinfos)

        if logger.get_dir() is not None and (epoch+1) % log_interval == 0:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            logger.logkv("misc/serial_timesteps", (epoch+1)*per_epoch_timesteps)
            logger.logkv("misc/nupdates", epoch)
            logger.logkv("misc/total_timesteps", (epoch+1)*per_epoch_timesteps*world_size)
            logger.logkv("fps", fps)
            logger.logkv("entropy", entropy.item())
            logger.logkv("kl", kl.item())
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            if eval_venv is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]))
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]))
            logger.logkv('misc/time_elapsed', tnow - tfirststart)
            logger.dumpkvs()

        # Log changes from update
        # writer.add_scalar('train/rewards', rew.sum(), epoch)
        if writer is not None:
            writer.add_scalar('train/kl', kl.item(), epoch)
            writer.add_scalar('train/entropy', entropy.item(), epoch)
            writer.add_scalar("misc/serial_timesteps", (epoch+1)*per_epoch_timesteps, epoch)
            writer.add_scalar("misc/nupdates", epoch)
            writer.add_scalar("misc/total_timesteps", (epoch+1)*per_epoch_timesteps*world_size, epoch)
            writer.add_scalar('train/eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]), epoch)
            writer.add_scalar('train/eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]), epoch)
            if eval_venv is not None and (epoch+1) % log_interval == 0:
                writer.add_scalar('eval/eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]), epoch)
                writer.add_scalar('eval/eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]), epoch)
            writer.add_scalar('misc/time_elapsed', tnow - tfirststart, epoch)

    if log_dir is not None:
        # save checkpoints
        torch.save({'model_state_dict': actor_critic.state_dict(),}, f'{log_dir}/model.ckpt')

def train_fn(rank, world_size, algo, nets_type, env_name, num_envs, eval_mode, distribution_mode, nsteps, device=-1):
    # create default process group
    if not TPU_MODE and world_size > 1:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # hp take from https://github.com/openai/train-procgen/blob/master/train_procgen/train.py
    learning_rate = 5e-4
    gamma = .999
    lam = .95
    nminibatches = 8
    epochs = 3
    n_eval_trials=20
    log_interval = 10

    ent_coef = 0.01

    if eval_mode == 'test':
        timesteps_per_proc = 5_000_000
        distribution_mode = 'easy'
    else:
        timesteps_per_proc = 25_000_000 if distribution_mode == 'easy' else 100_000_000 # 25M for easy level, 100M for difficult level

    # Serialize data into file:
    time_now = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Random seed
    seed = np.random.randint(1e6) + 10000 * rank # different seeds for each process
    torch.manual_seed(seed)
    np.random.seed(seed)

    if rank==0:
        log_dir = f"logs/{algo}.{nets_type}.workers_{world_size}/{env_name}.mode_{eval_mode}.distribution_{distribution_mode}.{time_now}_{seed}"
        format_strs = ['csv', 'stdout'] 
        logger.configure(dir=log_dir, format_strs=format_strs)
        writer = SummaryWriter(log_dir=log_dir)
    else:
        log_dir = None
        writer = None
    
    if rank==0:
        logger.info("creating environment")

    if eval_mode == 'eff':
        num_levels = 0  # The number of unique levels that can be generated. Set to 0 to use unlimited levels.
        start_level = 0 # The lowest seed that will be used to generated levels. 'start_level' and 'num_levels' fully specify the set of possible levels.
        eval_num_levels = 0
        eval_start_level = 0
    elif eval_mode == 'gen':
        num_levels = 500 if distribution_mode == 'hard' else 200
        start_level = 0
        eval_num_levels = 0
        eval_start_level = 0
    elif eval_mode == 'test':
        num_levels = 1
        start_level = 0
        eval_num_levels = 1
        eval_start_level = 0
    else:
        raise NotImplementedError
    
    if 'atari' in env_name:
        from stable_baselines3.common.env_util import make_atari_env
        from stable_baselines3.common.vec_env import VecFrameStack
        env_name = env_name.split('.')[1]
        # use atari env with terminal on life loss for better value bootstrap
        # cannot use VecMonitor then: episodic return and length will be incorrect
        # venv = make_atari_env(env_name, n_envs=num_envs, monitor_dir=log_dir, wrapper_kwargs={'terminal_on_life_loss': True})
        venv = make_atari_env(env_name, n_envs=num_envs)
        venv = VecFrameStack(venv, n_stack=3) # set stack number to 3 (compatible with Procgen number of channels)
    else:
        venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels, start_level=start_level, distribution_mode=distribution_mode)
        venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(venv=venv, filename=log_dir)

    if rank==0 and eval_mode == 'gen':
        eval_venv = ProcgenEnv(num_envs=n_eval_trials, env_name=env_name, num_levels=eval_num_levels, start_level=eval_start_level, distribution_mode=distribution_mode)
        eval_venv = VecExtractDictObs(eval_venv, "rgb")
        eval_venv = VecMonitor(venv=eval_venv)
    else:
        eval_venv = None

    if TPU_MODE:
        device = xm.xla_device()
    else:
        if device == -1:
            if torch.cuda.is_available(): # i.e. for NVIDIA GPUs
                device_type = "cuda"
            else:
                device_type = "cpu"
            
            device = torch.device(device_type) # Select best available device
        else:
            assert device >= 0
            device = f"cuda:{device}"

    obs_space = venv.observation_space

    # Create actor-critic module
    feature_input = False
    if nets_type == 'vit':
        embed_dim = 192
        kwargs = {'patch_size': 4, 'depth': 3, 'num_heads': 3, 'drop_path_rate': 0.1, 'device': device}
        neural_nets, preprocess = build_vit(obs_space, embed_dim, **kwargs)
    elif nets_type == 'resnet':
        embed_dim=256
        kwargs = {'with_norm_layer': True, 'depths': [16, 32, 32], 'device': device}
        neural_nets, preprocess = build_resnet(obs_space, embed_dim, **kwargs)
    elif nets_type == 'cnn':
        embed_dim=512
        kwargs = {'with_norm_layer': True, 'p_dropblock': 0.0, 'device': device}
        neural_nets, preprocess = build_cnn(obs_space, embed_dim, **kwargs)
    elif 'pretrained' in nets_type:
        embed_dim = 512 * 7 * 7
        n_type = nets_type.split('_')[1]
        kwargs = {'n_type': n_type}
        neural_nets, preprocess = build_pretrained(obs_space, **kwargs) 
        feature_input = True
    else: 
        raise NotImplementedError

    act_num = venv.action_space.n
    actor_critic = SeparateActorCritic(neural_nets, embed_dim, act_num, feature_input).to(device)
    if not TPU_MODE and world_size > 1:
        actor_critic = DDP(actor_critic, device_ids=[device])
    else:
        actor_critic = actor_critic

    venv = VecNormalize(venv=venv, norm_ret='atari' not in env_name, obs_preprocess=preprocess) # img transform and reward normalization
    if eval_venv is not None:
        eval_venv = VecNormalize(venv=eval_venv, obs_preprocess=preprocess)

    if rank==0:
        logger.info(f'Running on device: {device}')
        logger.info(f"training...")

        # Count variables
        var_counts = count_vars(neural_nets)
        logger.log(f'\nNumber of parameters: {var_counts}\n')

        kwargs = dict(locals())
        json.dump(kwargs, open( f"{log_dir}/kwargs.json", 'w' ) , default=repr)

    learn(world_size, algo, actor_critic, writer, venv, eval_venv, device, log_dir, total_timesteps=timesteps_per_proc,
          nsteps=nsteps, nminibatches=nminibatches, lam=lam, gamma=gamma, noptepochs=epochs, lr=learning_rate, 
          ent_coef=ent_coef, log_interval=log_interval)

def main():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--algo', type=str, default='trpo', choices=['trpo'])
    parser.add_argument('--nets_type', type=str, default='cnn', choices=['resnet', 'vit', 'cnn', 'pretrained_clip', 'pretrained_resnet'])
    parser.add_argument('--env_name', type=str, default='dodgeball')
    parser.add_argument('--num_envs', type=int, default=16)
    parser.add_argument('--distribution_mode', type=str, default='hard', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--eval_mode', type=str, default='test', choices=['eff', 'gen', 'test'])
    parser.add_argument('--nsteps', type=int, default=256)
    parser.add_argument('--device', type=int, default=-1) # -1: use any available device
    parser.add_argument('--n_proc', type=int, default=1) # distributed training: number of processes
    parser.add_argument('--port_num', type=int, default=29500) # distributed training: number of processes

    args = parser.parse_args()

    if args.n_proc > 1:
        # multiple nodes
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(args.port_num)

        main_mp = xmp if TPU_MODE else mp
        main_mp.spawn(train_fn, args=(args.n_proc, args.algo, args.nets_type, args.env_name, args.num_envs, 
                                            args.eval_mode, args.distribution_mode, args.nsteps, args.device, ),
                        nprocs=args.n_proc, # INFO: for TPU, either 1 or the maximum number of TPU chips
                        join=True)

    else:
        train_fn(0, args.n_proc, args.algo, args.nets_type, args.env_name, args.num_envs, 
                args.eval_mode, args.distribution_mode, args.nsteps, args.device, )

if __name__ == '__main__':
    main()
