from typing import Any, Dict, Tuple
import time
import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax.experimental import pjit, PartitionSpec as P
from flax import struct


import optax
from t5x.optimizers import adamw
from t5x.partitioning import PjitPartitioner
from t5x.train_state import FlaxOptimTrainState

from einops import rearrange

from modeling_palm import PaLMModel
from config import PaLMConfig


DEFAULT_TPU_RULES = [
    ('batch', 'data'),
    ('mlp', 'model'),
    ('heads', 'model'),
    ('vocab', 'model'),
    ('embed', None),
    ('kv', None),
    ('joined_kv', None),
    ('relpos_buckets', None),
    ('abspos_buckets', None),
    ('length', None),
    ('layers', None),
    ('stack', None),
    ('mlp_activations', None),

    ("data", "data"),
    ("model", "model"),
    (None, None),
]


class PaLMState(struct.PyTreeNode):
    train_state: FlaxOptimTrainState
    apply_fn: Any = struct.field(pytree_node=False)
    lr: Any = struct.field(pytree_node=False)

    step: int
    config: Any = struct.field(pytree_node=False)


def cross_entropy(logprobs, targets):
    target_class = jnp.argmax(targets, axis=1)
    nll = jnp.take_along_axis(logprobs, jnp.expand_dims(target_class, axis=1), axis=1)
    ce = -jnp.mean(nll)
    return ce


def train_step(palm_state, seqs):
    def loss_fn(params):
        inp, labels = seqs[:, :-1], seqs[:, 1:]
        labels = jax.nn.one_hot(labels, palm_state.config.num_tokens)
        logits = palm_state.apply_fn(
            {"params": params},
            inp
        )
        loss = cross_entropy(logits, labels) # TODO: See if should be rearranged
        return loss
    gradient_fn = jax.value_and_grad(loss_fn)
    loss, grads = gradient_fn(palm_state.train_state.params)
    
    train_state = palm_state.train_state.apply_gradient(grads=grads, learning_rate=palm_state.lr)
    
    palm_state = palm_state.replace(train_state=train_state, step=palm_state.step + 1)
    
    return palm_state, {"loss": loss}


class PaLM:
    def __init__(self, config: PaLMConfig):
        start_time = time.time()
        self.config = config
        self.random_state = jax.random.PRNGKey(seed=config.seed)

        self.partitioner = PjitPartitioner(num_partitions=1, logical_axis_rules=DEFAULT_TPU_RULES)

        lr = 1e-2
        opt = adamw(learning_rate=lr, b1=0.9, b2=0.999,
                    eps=1e-8, weight_decay=1e-8)
        model = PaLMModel(config=config)
        def init_model(rng):
            rng, key = jax.random.split(rng)
            seq = jax.random.randint(key, (self.config.batch_size, 2048), 0, config.num_tokens)
            rng, key = jax.random.split(rng)
            params = model.init(key, seq)
            return FlaxOptimTrainState.create(opt, params)

        params_shape = jax.eval_shape(
            init_model, self.get_key()
        )
        params_spec = self.partitioner.get_mesh_axes(params_shape)

        p_init = self.partitioner.partition(init_model,
                                            in_axis_resources=(None,),
                                            out_axis_resources=(params_spec)
                                            )
        params = p_init(self.get_key())


        model_state_spec = PaLMState(
            train_state=params_spec,
            apply_fn=model.apply,
            lr=lr,
            step=None,
            config=self.config
        )

        self.model_state = PaLMState(
            train_state=params,
            apply_fn=model.apply,
            lr=lr,
            step=0,
            config=self.config
        )
        
        self.p_train_step = self.partitioner.partition(
            train_step,
            in_axis_resources=(
                model_state_spec, 
                P("data",)
            ),
            out_axis_resources=(model_state_spec, None)
        )
        
        n_params_flax = sum(
            jax.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape), params))
        )
        print(f"Setup complete, it took {time.time() - start_time:0.2f} seconds, for a total of {n_params_flax:,} parameters"
              )
    def get_key(self):
        self.random_state, key = jax.random.split(self.random_state)
        return key

    def train_step(self, seqs):
        self.model_state, metrics =  self.p_train_step(self.model_state, seqs)
        return metrics

def test():
    config = PaLMConfig()
    model = PaLM(config=config)
    pb = tqdm(range(100000))
    while True:
        model.train_step(jnp.ones((config.batch_size, 2049), dtype=int))
        pb.update(1)
if __name__ == "__main__":
    test()