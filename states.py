from typing import Any, Callable
import flax
import jax
import jax.numpy as jnp

import optax
from flax import struct, traverse_util
from flax.core.frozen_dict import FrozenDict
from flax.linen import partitioning as flax_partitioning
from flax import serialization
from flax.core import frozen_dict
import optimizers

def _validate_params_axes(params_axes, params):
    axis_names = flax_partitioning.get_axis_names(params_axes)
    missing_params_axes = (
        set(traverse_util.flatten_dict(params, sep='/')) -
        set(traverse_util.flatten_dict(axis_names, sep='/')))
    if missing_params_axes:
        raise ValueError(
            f'Missing axis names for parameters: {missing_params_axes}')


class TrainState(struct.PyTreeNode):
    optimizer: optimizers.Optimizer # state for optax optimizer state, target for model params 
    params_axes: FrozenDict[str, Any] = None

    def apply_gradients(self, *, grads, **kwargs):
        new_optimizer = self.optimizer.apply_gradient(grads,)
        return self.replace(optimizer=new_optimizer)

    @classmethod
    def create(cls, *, params, optimizer, **kwargs):
        other_variables, params = params.pop("params")
        if 'params_axes' in other_variables:
            other_variables, params_axes = other_variables.pop('params_axes')
            _validate_params_axes(params_axes, params)
        else:
            raise ValueError("params_axes not found")
        if hasattr(optimizer, 'set_param_axes'):
            if params_axes is None:
                raise ValueError('The optimizer supports params_axes for model-based '
                                 'partitioning, but the model is not emitting them.')
            # `get_axis_names` removes "_axes" suffix in the leaf name and replaces
            # `AxisMetadata` with `PartitionSpec`.
            axis_names = flax_partitioning.get_axis_names(params_axes)
            optimizer.set_param_axes(axis_names)

        return cls(params_axes=params_axes, optimizer=optimizer.create(params))
    
    def state_dict(self):
        state_dict = self.optimizer.state_dict()
        return state_dict
        
    def restore_state(self, state_dict):
        new_optimizer = self.optimizer.restore_state(state_dict)
        return self.replace(
            optimizer=new_optimizer,)

    
    @property
    def param_states(self):
        return self.optimizer.state.param_states
    
    
    def as_logical_axes(self):
        # TODO: Get optimizer logical axes based on params_axes
        return TrainState(
            optimizer=self.optimizer.optimizer_def.derive_logical_axes(
                self.optimizer,
                flax_partitioning.get_axis_names(self.params_axes))  # derive logical axes
        )
