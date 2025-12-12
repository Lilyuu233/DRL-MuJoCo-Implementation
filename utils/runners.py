import torch
import numpy as np
from abc import ABC, abstractmethod
from utils.utils import model_step

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.obs = env.reset()
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, nsteps, gamma, lam, device, test_mode=False):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.device = device
        self.test_mode = test_mode

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_logpacs = [],[],[],[],[],[]
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            # actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            actions, values, logpacs = model_step(self.model, self.obs, deterministic=self.test_mode)
            mb_obs.append(self.obs.clone())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_logpacs.append(logpacs)
            mb_dones.append(self.dones) 

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs, rewards, self.dones, infos = self.env.step(actions.cpu().numpy()) # done==true: a new episode just started

            for idx, info in enumerate(infos):
                maybeepinfo = info.get('episode')
                if maybeepinfo: 
                    epinfos.append(maybeepinfo)

            mb_rewards.append(rewards)

        if self.test_mode:
            return epinfos

        #batch of steps to batch of rollouts
        mb_obs = torch.stack(mb_obs, dim=0)
        mb_rewards = torch.from_numpy(np.asarray(mb_rewards)).to(self.device)
        mb_actions = torch.stack(mb_actions, dim=0)
        mb_values = torch.stack(mb_values, dim=0)
        mb_logpacs = torch.stack(mb_logpacs, dim=0)
        mb_dones = torch.from_numpy(np.asarray(mb_dones).astype(np.float32)).to(self.device)

        last_values = model_step(self.model, self.obs)[1]

        # discount/bootstrap off value fn
        mb_returns = torch.zeros_like(mb_rewards)
        mb_advs = torch.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - torch.from_numpy(self.dones.astype(np.float32)).to(self.device)
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_actions, mb_advs, mb_logpacs)), epinfos)

# obs, returns, actions, advantages, logpacs = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


