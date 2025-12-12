import copy
import torch
import scipy
import numpy as np

from torch import nn
from torch.utils.data import Dataset
from torch.distributions.categorical import Categorical

from utils.vision_transformers import vit_tiny
from utils.vit import ViT
from utils.resnet import ResNetImpala
from utils.convnet import NatureCNN

def build_cnn(obs_space, emb_size=256, device='cpu', **kwargs):
    def preprocess(obs_batch):
        obs_batch = np.asarray(obs_batch).astype(np.float32)
        obs_tensor = torch.from_numpy(obs_batch).to(device)
        obs = obs_tensor / 255.0
        return torch.permute(obs, (0, 3, 1, 2)) if len(obs.size()) == 4 else torch.permute(obs, (2, 0, 1)) 

    img_size = obs_space.shape[1]
    return NatureCNN(img_size, emb_size, **kwargs), preprocess

def build_pretrained(obs_space, n_type, **kwargs):
    import timm
    from torchvision import transforms
    if n_type == 'resnet':
        embed_nets = timm.create_model('resnet18', pretrained=True)

    elif n_type == 'clip':
        raise NotImplementedError
    else:
        raise NotImplementedError

    def preprocess(obs_batch):
        obs = torch.from_numpy(obs_batch).float() / 255.0
        obs = torch.permute(obs, (0, 3, 1, 2))
        transform = transforms.Compose(
            [
                transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None), 
                transforms.CenterCrop(size=(224, 224)), 
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250])), 
            ])
        return transform(obs)

    return embed_nets, preprocess

def build_resnet(obs_space, emb_size=256, device='cpu', **kwargs):
    def preprocess(obs_batch):
        obs_batch = obs_batch.astype(np.float32)
        obs_tensor = torch.from_numpy(obs_batch).to(device)
        obs = obs_tensor / 255.0
        return torch.permute(obs, (0, 3, 1, 2)) if len(obs.size()) == 4 else torch.permute(obs, (2, 0, 1)) 

    img_size = obs_space.shape[1]
    return ResNetImpala(img_size, emb_size, **kwargs), preprocess

def build_vit(obs_space, embed_dim, device, **kwargs):
    def preprocess(obs_batch):
        obs_batch = np.asarray(obs_batch).astype(np.float32)
        obs_tensor = torch.from_numpy(obs_batch).to(device)
        obs = obs_tensor / 255.0
        return torch.permute(obs, (0, 3, 1, 2))

    return vit_tiny(**kwargs), preprocess

def build_clip(ob_space, **kwargs):
    import open_clip
    embed_nets, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    return embed_nets, preprocess

def build_mlp(obs_space, output_dim, net_arch=[256, 256], activation_fn=nn.ReLU, squash_output=False):
    input_dim = obs_space.shape[0]

    modules = []
    if len(net_arch) > 0:
        modules.append(nn.Linear(input_dim, net_arch[0], bias=True))
        modules.append(activation_fn())

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=True))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=True))

    if squash_output:
        modules.append(nn.Tanh())
    return modules

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

class BufferDataset(Dataset):
    def __init__(self, obs, ret, val, act, logp):
        self.obs = obs
        self.ret = ret
        self.val = val
        self.act = act
        self.logp = logp

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, index):
        return self.obs[index], self.ret[index], self.val[index], self.act[index], self.logp[index]

class CategoricalActorCritic(nn.Module):
    def __init__(self, embed_net, embed_dim, n_actions, feature_input=False):
        super().__init__()
        self.n_actions = n_actions # used for one-hot encoding
        if feature_input:
            self.embed_net = lambda x: torch.flatten(embed_net.forward_features(x), start_dim=1)
        else:
            self.embed_net = embed_net

        self.logits_net = nn.Sequential(
            # nn.Linear(embed_dim, embed_dim),
            # nn.ReLU(),
            nn.Linear(embed_dim, n_actions),
        )
        self.v_net = nn.Sequential(
            # nn.Linear(embed_dim, embed_dim),
            # nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, obs):
        latents = self.embed_net(obs)
        logits = self.logits_net(latents)
        vals = self.v_net(latents)

        return vals.squeeze(-1), logits

class CategoricalActor(nn.Module):
    def __init__(self, embed_net, embed_dim, n_actions, feature_input=False):
        super().__init__()
        self.n_actions = n_actions # used for one-hot encoding
        if feature_input:
            self.embed_net = lambda x: torch.flatten(embed_net.forward_features(x), start_dim=1)
        else:
            self.embed_net = embed_net

        self.logits_net = nn.Sequential(
            nn.Linear(embed_dim, n_actions),
        )

    def forward(self, obs):
        latents = self.embed_net(obs)
        logits = self.logits_net(latents)
        return logits

class ValueCritic(nn.Module):
    def __init__(self, embed_net, embed_dim, n_actions, feature_input=False):
        super().__init__()
        self.n_actions = n_actions # used for one-hot encoding
        if feature_input:
            self.embed_net = lambda x: torch.flatten(embed_net.forward_features(x), start_dim=1)
        else:
            self.embed_net = embed_net

        self.v_net = nn.Sequential(
            # nn.Linear(embed_dim, embed_dim),
            # nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, obs):
        latents = self.embed_net(obs)
        vals = self.v_net(latents)
        return vals.squeeze(-1)

class SeparateActorCritic(nn.Module):
    def __init__(self, embed_net, embed_dim, n_actions, feature_input=False):
        super().__init__()
        self.logits_net = CategoricalActor(embed_net, embed_dim, n_actions, feature_input=False)
        self.v_net = ValueCritic(copy.deepcopy(embed_net), embed_dim, n_actions, feature_input=False)

    def forward(self, obs):
        logits = self.logits_net(obs)
        vals = self.v_net(obs)
        return vals, logits
    
    def forward_logits(self, obs):
        return self.logits_net(obs)
    
    def forward_vals(self, obs):
        return self.v_net(obs)

def model_step(model, obs, deterministic=False):
    with torch.no_grad():
        vals, logits = model(obs)

        if deterministic: 
            act = torch.argmax(logits, dim=-1)
        else:
            pi = Categorical(logits=logits)
            act = pi.sample()

        logp_a = torch.gather( nn.functional.log_softmax(logits, dim=-1), dim=-1, index=act.unsqueeze(-1)).squeeze(1)

    return act, vals, logp_a

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))

    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(params, flat_params):
    prev_ind = 0
    for param in params:
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

def set_grads_from_flat(params, flat_grads):
    prev_ind = 0
    for param in params:
        flat_size = int(np.prod(list(param.grad.size())))
        param.grad.data.copy_(
            flat_grads[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size