from flax.training import orbax_utils
import jax.numpy as jnp
from jax import random
from orbax.checkpoint import PyTreeCheckpointer
from torchvision.datasets import MNIST
from tqdm import tqdm

from dataseqs import shuffle_batch
from evaluate import acc_softmax
from models.pretrained_mnist.foundation import state_init
from train import make_loss_reg, make_loss_sce, make_step


mnist_train = MNIST('data', train=True, download=True)
xtrain = jnp.array(mnist_train.data)[:, :, :, None] / 255.0
ytrain = jnp.array(mnist_train.targets)

state = state_init
state = state.replace(hyperparams={'precision': 0.1})
loss_basic = make_loss_sce(state)
loss = make_loss_reg(state, loss_basic)
step = make_step(loss)
key = random.PRNGKey(1337)
for _ in tqdm(range(1)):
    key, key1 = random.split(key)
    for xbatch, ybatch in shuffle_batch(key1, 64, xtrain, ytrain):
        state = step(state, xbatch, ybatch)
PyTreeCheckpointer().save(
    '/home/waiyan/thesis/ckpt/pretrained_mnist/foundation',
    state.params,
    save_args=orbax_utils.save_args_from_target(state.params)
)
