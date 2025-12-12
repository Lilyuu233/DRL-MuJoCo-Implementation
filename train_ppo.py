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

from utils.runners import Runner
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from utils.utils import build_cnn, build_resnet, build_vit, build_pretrained
from utils.utils import CategoricalActorCritic, count_vars, safemean
from vec_env import ( VecExtractDictObs, VecMonitor, VecNormalize)

def learn(world_size, actor_critic, writer, venv, eval_venv, device, log_dir, 
          total_timesteps, nsteps, nminibatches, lam, gamma, noptepochs, lr, cliprange, ent_coef, max_grad_norm=0.5):

    per_epoch_timesteps = nsteps * venv.num_envs
    # epochs = total_timesteps // (per_epoch_timesteps * world_size) + 1
    epochs = total_timesteps // per_epoch_timesteps + 1

    log_interval = 10

    minibatch_size = per_epoch_timesteps // nminibatches

    # Instantiate the runner object
    runner = Runner(env=venv, model=actor_critic, nsteps=nsteps, gamma=gamma, lam=lam, device=device)
    if eval_venv is not None:
        eval_runner = Runner(env=eval_venv, model=actor_critic, nsteps=nsteps, gamma=gamma, lam=lam, device=device, test_mode=True)

    epinfobuf = deque(maxlen=100)
    if eval_venv is not None:
        eval_epinfobuf = deque(maxlen=100)

    # Set up optimizers for policy and value function
    ac_optimizer = Adam(actor_critic.parameters(), lr=lr)

    # Start total timer
    tfirststart = time.perf_counter()

    def compute_loss(_obs, _ret, _act, _adv, _logp_old):
        # advantage normalization
        _adv_mean, _adv_std = _adv.mean(), _adv.std()
        _adv = (_adv - _adv_mean) / (_adv_std + 1e-8)

        _vals, _logits = actor_critic(_obs)
        _logp_act = F.log_softmax(_logits, dim=-1)
        _logp = torch.gather(_logp_act, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
        _ratio = torch.exp(_logp - _logp_old)
        _p_log_p = torch.exp(_logp_act) * _logp_act
        _entropy = - _p_log_p.sum(-1).mean()

        _clip_adv = torch.clamp(_ratio, 1-cliprange, 1+cliprange) * _adv
        _losses_pi = torch.max(- _ratio * _adv, - _clip_adv)
        _loss_pi = _losses_pi.mean()

        # value loss
        _loss_v = F.mse_loss(_vals, _ret)
        _loss = _loss_pi + 0.5 * _loss_v - ent_coef * _entropy

        # Useful extra info
        with torch.no_grad():
            approx_kl = (_logp_old - _logp).mean().item()
            ent = _entropy.item()
            clipped = _ratio.gt(1+cliprange) | _ratio.lt(1-cliprange)
            clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
            pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return _loss, _loss_pi, _loss_v, pi_info

    tepochs = trange(epochs+1, desc='Epoch starts', leave=True)

    # Main loop: collect experience in env and update/log each epoch
    inds = np.arange(per_epoch_timesteps)
    for epoch in tepochs:
        tstart = time.perf_counter()

        tepochs.set_description('Stepping environment...')

        actor_critic.eval() # set to eval mode for PPO
        obs, ret, act, adv, logp_old, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)

        tepochs.set_description('Minibatch training...')

        for _ in range(noptepochs):
            # Randomize the indexes
            np.random.shuffle(inds)

            # 0 to batch_size with batch_train_size step
            for start in range(0, per_epoch_timesteps, minibatch_size):
                end = start + minibatch_size
                mbinds = inds[start:end]
                slices = (arr[mbinds] for arr in (obs, ret, act, adv, logp_old))

                ac_optimizer.zero_grad()

                actor_critic.eval() # need to set to eval otherwise trust region can be violated by batch norm and dropout
                mb_loss, mb_loss_pi, mb_loss_v, pi_info = compute_loss(*slices)
                mb_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)

                if TPU_MODE and (world_size > 1):
                    xm.optimizer_step(ac_optimizer)
                else:
                    ac_optimizer.step()
                
                if TPU_MODE: # TPU backend
                    xm.mark_step()

            tepochs.set_postfix(loss_pi=mb_loss_pi.item(), loss_v=mb_loss_v.item(), entropy=pi_info['ent'], kl=pi_info['kl'], cf=pi_info['cf'])

        # clean GPU cache
        del obs, ret, act, adv, logp_old
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
            logger.logkv("entropy", pi_info['ent'])
            logger.logkv("kl", pi_info['kl'])
            logger.logkv("clipfrac", pi_info['cf'])
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
            writer.add_scalar('train/kl', pi_info['kl'], epoch)
            writer.add_scalar('train/clipfrac', pi_info['cf'], epoch)
            writer.add_scalar('train/entropy', pi_info['ent'], epoch)
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
        torch.save({'model_state_dict': actor_critic.state_dict(),
                    'optimizer_state_dict': ac_optimizer.state_dict(), }, f'{log_dir}/model.ckpt')

def train_fn(rank, world_size, nets_type, env_name, num_envs, eval_mode, distribution_mode, nsteps, device=-1):
    # create default process group
    if not TPU_MODE and world_size > 1:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # hp take from https://github.com/openai/train-procgen/blob/master/train_procgen/train.py
    learning_rate = 5e-4
    gamma = .999
    lam = .95
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    ent_coef = 0.01
    n_eval_trials=20

    if eval_mode == 'test':
        timesteps_per_proc = 5_000_000
        distribution_mode = 'easy'
    else:
        timesteps_per_proc = 25_000_000 if distribution_mode == 'easy' else 200_000_000 # 25M for easy level, 200M for difficult level

    # Serialize data into file:
    time_now = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Random seed
    seed = np.random.randint(1e6) + 10000 * rank # different seeds for each process
    torch.manual_seed(seed)
    np.random.seed(seed)

    if rank==0:
        log_dir = f"logs/{nets_type}.workers_{world_size}/{env_name}.mode_{eval_mode}.distribution_{distribution_mode}.{time_now}_{seed}"
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
        venv = make_atari_env(env_name, n_envs=num_envs, monitor_dir=log_dir, wrapper_kwargs={'terminal_on_life_loss': True})
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
    actor_critic = CategoricalActorCritic(neural_nets, embed_dim, act_num, feature_input).to(device)
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

    learn(world_size, actor_critic, writer, venv, eval_venv, device, log_dir, total_timesteps=timesteps_per_proc,
          nsteps=nsteps, nminibatches=nminibatches, lam=lam, gamma=gamma, noptepochs=ppo_epochs, 
          lr=learning_rate, cliprange=clip_range, ent_coef=ent_coef)

def main():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--nets_type', type=str, default='cnn', choices=['resnet', 'vit', 'cnn', 'pretrained_clip', 'pretrained_resnet'])
    parser.add_argument('--env_name', type=str, default='coinrun')
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--distribution_mode', type=str, default='hard', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--eval_mode', type=str, default='gen', choices=['eff', 'gen', 'test'])
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
        main_mp.spawn(train_fn, args=(args.n_proc, args.nets_type, args.env_name, args.num_envs, 
                                            args.eval_mode, args.distribution_mode, args.nsteps, args.device, ),
                        nprocs=args.n_proc, # INFO: for TPU, either 1 or the maximum number of TPU chips
                        join=True)

    else:
        train_fn(0, args.n_proc, args.nets_type, args.env_name, args.num_envs, 
                args.eval_mode, args.distribution_mode, args.nsteps, args.device, )

if __name__ == '__main__':
    main()
