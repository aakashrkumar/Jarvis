import jax
import flax
import optax
from palm_main import PaLM
from config import PaLMConfig

from torch.utils.data import DataLoader, Dataset

import tqdm
import wandb

import jax.numpy as jnp
from datasets import load_dataset

def train():
    config = PaLMConfig
    model = PaLM(config = config)
    pb = tqdm.tqdm()
    
    while True:
        seq = jnp.ones((16, 2049), dtype=int)
        metrics = model.train_step(seq)
        pb.update(1)
        # wandb.log(metrics)
if __name__ == "__main__":
    train()