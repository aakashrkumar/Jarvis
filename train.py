import jax
import flax
import optax
from t5x.partitioning import PjitPartitioner
from palm_main import PaLM
from config import PaLMConfig

from torch.utils.data import DataLoader, Dataset

import tqdm
import wandb

import jax.numpy as jnp

def train():
    config = PaLMConfig
    model = PaLM(config = config)
    pb = tqdm.tqdm()
    
    while True:
        seq = jnp.ones((16, 2049), dtype=jnp.uint)
        metrics = model.train_step(seq)
        pb.update(1)
        # wandb.log(metrics)