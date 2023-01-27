from functools import partial
from typing import Any, Callable, Dict, Tuple
import time
import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax.experimental import PartitionSpec as P
from jax.experimental.pjit import pjit
from flax import struct
from jax.experimental import maps


import optax

from einops import rearrange

from modeling_palm import PaLMModel
from config import PaLMConfig
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax import core, struct, traverse_util
import partitioning as nnp


class TrainState(struct.PyTreeNode):
    step: int
    params: FrozenDict[str, Any]
    opt_state: optax.OptState
    apply_fn: Callable = struct.field(pytree_node=False)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    dropout_rng: jnp.ndarray = None
    epoch: int = 0
    train_time: float = 0.0  # total time the model trained
    train_samples: int = 0  # number of samples seen

    def apply(self, *args, **kwargs):
        return self.apply_fn(self, self.params, *args, **kwargs)

    def apply_gradients(self, *, grads, **kwargs):
        grads = split_params(grads)
        params = split_params(
            self.params
        )
        opt_state = {}
        # we loop over keys: "standard", "scanned_encoder", "scanned_decoder"
        for k, param in params.items():
            update_fn = self.tx[k].update
            updates, new_opt_state = update_fn(grads[k], self.opt_state[k], param)
            params[k] = optax.apply_updates(param, updates)
            opt_state[k] = new_opt_state
        params = unsplit_params(params)
        # merge with non-trainable params
        params, new_params = traverse_util.flatten_dict(
            unfreeze(self.params)
        ), traverse_util.flatten_dict(unfreeze(params))
        params.update(new_params)
        params = freeze(traverse_util.unflatten_dict(params))

        return self.replace(
            step=self.step + 1,
            params=params,
            opt_state=freeze(opt_state),
            **kwargs,
        )


    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        opt_state = {}
        for k, p in split_params(params).items():
            init_fn = tx[k].init
            opt_state[k] = init_fn(p)
        return cls(
                step=0,
                apply_fn=apply_fn,
                params=params,
                tx=tx,
                opt_state=freeze(opt_state),
                **kwargs,
            )
    


class PaLMState(struct.PyTreeNode):
    train_state: TrainState
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
        logits = palm_state.train_state.apply_fn(
            params,
            inp
        )
        loss = cross_entropy(logits, labels)  # TODO: See if should be rearranged
        return loss
    gradient_fn = jax.value_and_grad(loss_fn)
    loss, grads = gradient_fn(palm_state.train_state.params)

    train_state = palm_state.train_state.apply_gradients(grads=grads,)
    palm_state = palm_state.replace(train_state=train_state)
    return palm_state, {"loss": loss}


def split_params(data):
    """Split params between scanned and non-scanned"""
    flat = traverse_util.flatten_dict(unfreeze(data))
    split = {"standard": {}}
    for k, v in flat.items():
        split["standard"][k] = v
    # remove empty keys
    split = {k: v for k, v in split.items() if v}
    for k, v in split.items():
        split[k] = freeze(traverse_util.unflatten_dict(v))
    return split


def unsplit_params(data):
    flat = {}
    for k in ["standard"]:
        if k in data:
            flat.update(traverse_util.flatten_dict(unfreeze(data[k])))
    return freeze(traverse_util.unflatten_dict(flat))


def _opt_state_spec_per_leaf(x, spec):
    if isinstance(x, FrozenDict):
        # variables with same structure as params
        return spec
    else:
        # other variables such as count
        return None


class PaLM:
    def __init__(self, config: PaLMConfig):
        start_time = time.time()
        self.config = config
        self.random_state = jax.random.PRNGKey(seed=config.seed)

        mesh_shape = (1, 8)
        self.devices = np.asarray(jax.devices()).reshape(*mesh_shape)
        self.mesh = maps.Mesh(self.devices, ("dp", "mp"))


        model = PaLMModel(config=config)

        def init_params(rng):
            rng, key = jax.random.split(rng)
            seq = jax.random.randint(key, (self.config.batch_size, 2048), 0, config.num_tokens)
            rng, key = jax.random.split(rng)
            params = model.init(key, seq)
            return params

        lr = 1e-4
        optimizer = optax.adamw(learning_rate=lr, b1=0.9, b2=0.999,
                          eps=1e-8, weight_decay=1e-8)
        params_shape = jax.eval_shape(
            init_params, self.get_key()
        )

        param_spec = nnp.set_partitions(params_shape)
        params_shape = freeze(params_shape)

        optimizer = {k: optimizer for k in split_params(params_shape)}

        opt_state_shape = {}
        for k, p in split_params(params_shape).items():
            opt_state_shape[k] = jax.eval_shape(optimizer[k].init, p)
        split_spec = split_params(nnp.set_partitions(params_shape))
        opt_state_spec = {}
        for k, p in split_params(params_shape).items():
            opt_state_spec[k] = jax.tree_util.tree_map(
                partial(_opt_state_spec_per_leaf, spec=split_spec[k]),
                opt_state_shape[k],
                # return None spec for empty elements
                is_leaf=lambda x: isinstance(x, (FrozenDict, optax.EmptyState)),
            )
        opt_state_shape = freeze(opt_state_shape)
        opt_state_spec = freeze(opt_state_spec)


        train_state_spec = TrainState(
            params=param_spec,
            opt_state=opt_state_spec,
            dropout_rng=None,
            step=None,
            epoch=None,
            train_time=None,
            train_samples=None,
            apply_fn=model.apply,
            tx=optimizer,
        )

        with self.mesh:
            params = pjit(init_params, in_axis_resources=(None,), out_axis_resources=(param_spec))(self.get_key())
            def init_state(params):
                return TrainState.create(
                    apply_fn=model.apply,
                    tx=optimizer,
                    params=params,
                )

            train_state = pjit(
                init_state,
                in_axis_resources=(param_spec,),
                out_axis_resources=train_state_spec,
                donate_argnums=(0,),
            )(params)
        
        
        model_state_spec = PaLMState(
            train_state=train_state_spec,
            config=self.config
        )
        
        self.model_state = PaLMState(
            train_state=train_state,
            config=self.config,
        )

        self.p_train_step = pjit(
            train_step,
            in_axis_resources=(
                model_state_spec,
                P("dp",)
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
        with maps.Mesh(self.devices, ("dp", "mp")):
            self.model_state, metrics = self.p_train_step(self.model_state, seqs)
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
