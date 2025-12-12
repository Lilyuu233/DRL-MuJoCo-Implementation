from .likelihood_base import *
from .bernoulli_likelihood import *
from .gaussian_likelihood import *
from .softmax_likelihood import *

FISH_LIKELIHOODS = {
    "gaussian": GaussianLikelihood,
    "bernoulli": BernoulliLikelihood,
    "softmax": SoftMaxLikelihood,
}