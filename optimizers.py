# Copyright 2022 The T5X Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""T5X Optimizer Support.

Tools for wrapping Optax optimizers and handling SPMD annotations for use with
pjit.

Additional support for the legacy Adafactor implementation.
"""

import functools
from typing import Any, Optional, Union, Sequence, Tuple, Mapping

import flax

# just used for transitional type definitions

from flax import serialization
from flax import struct
from flax import traverse_util
from flax.core import frozen_dict
from flax.serialization import from_state_dict
from flax.serialization import to_state_dict
import jax
import jax.numpy as jnp
from jestimator import amos
from jestimator import amos_helper
import optax

freeze = flax.core.frozen_dict.freeze
unfreeze = flax.core.frozen_dict.unfreeze

Dtype = Any


@struct.dataclass
class OptimizerState:
    step: jnp.ndarray
    param_states: Any

# Optax Elementwise Wrapper


class OptaxStatePartitionRules:
    """Collection of rules to partition optax states.

    These rules work for optimizers whose states are simply replications of
    params, e.g., Adam. Optimizers that aim to save memory by factoring states,
    e.g., Adafactor, SM3, are not supported currently.
    """

    # Rules mapping a particular optax state to a callable returning the state
    # with arrays replaced by t5x PartitionSpec or None.
    #
    # NOTE(levskaya): This is not an entirely exhaustive list, add to this list
    # to support additional optimizers / transformations.
    #
    # pylint: disable=g-long-lambda

    _RULES = {

        # Leaf Optax States:
        amos.ScaleByAmosState:
            amos_helper.state_partition_rule,
        optax.AddNoiseState:
            lambda state, params_axes: optax.AddNoiseState(
                count=None, rng_key=None),
        optax.DifferentiallyPrivateAggregateState:
            lambda state, params_axes: optax.DifferentiallyPrivateAggregateState(
                rng_key=None),
        optax.EmaState:
            lambda state, params_axes: optax.EmaState(
                count=None, ema=params_axes),
        optax.EmptyState:
            lambda state, params_axes: optax.EmptyState(),
        optax.TraceState:
            lambda state, params_axes: optax.TraceState(trace=params_axes),
        optax.ScaleByAdamState:
            lambda state, params_axes: optax.ScaleByAdamState(
                count=None, mu=params_axes, nu=params_axes),
        optax.ScaleByBeliefState:
            lambda state, params_axes: optax.ScaleByBeliefState(
                count=None, mu=params_axes, nu=params_axes),
        optax.ScaleByRssState:
            lambda state, params_axes: optax.ScaleByRssState(
                sum_of_squares=params_axes),
        optax.ScaleByRmsState:
            lambda state, params_axes: optax.ScaleByRmsState(nu=params_axes),
        optax.ScaleByRStdDevState:
            lambda state, params_axes: optax.ScaleByRStdDevState(
                mu=params_axes, nu=params_axes),
        optax.ScaleBySM3State:
            lambda state, params_axes: optax.ScaleBySM3State(
                mu=params_axes, nu=params_axes),
        optax.ScaleByTrustRatioState:
            lambda state, params_axes: optax.ScaleByTrustRatioState(),
        optax.ScaleByScheduleState:
            lambda state, params_axes: optax.ScaleByScheduleState(count=None),
        optax.ZeroNansState:
            lambda state, params_axes: optax.ZeroNansState(found_nan=None),
        # FactoredState

        # Recursive, Combinator Optax States:

        # MaskedState
        optax.MaskedState:
            lambda state, params_axes: optax.MaskedState(
                inner_state=OptaxStatePartitionRules.derive_optax_logical_axes(
                    state.inner_state, params_axes)),
        optax.InjectHyperparamsState:
            lambda state, params_axes: optax.InjectHyperparamsState(
                count=None,
                hyperparams=jax.tree_map(lambda x: None, state.hyperparams),
                inner_state=OptaxStatePartitionRules.derive_optax_logical_axes(
                    state.inner_state, params_axes)),
        optax.MultiStepsState:
            lambda state, params_axes: optax.MultiStepsState(
                mini_step=None,
                gradient_step=None,
                inner_opt_state=OptaxStatePartitionRules.
                derive_optax_logical_axes(  # pylint: disable=line-too-long
                    state.inner_opt_state, params_axes),
                acc_grads=params_axes),
        optax.ApplyIfFiniteState:
            lambda state, params_axes: optax.ApplyIfFiniteState(
                notfinite_count=None,
                last_finite=None,
                total_notfinite=None,
                inner_state=OptaxStatePartitionRules.derive_optax_logical_axes(
                    state.inner_state, params_axes)),
        optax.MaybeUpdateState:
            lambda state, params_axes: optax.MaybeUpdateState(
                inner_state=OptaxStatePartitionRules.derive_optax_logical_axes(
                    state.inner_state, params_axes),
                step=None),
        optax.MultiTransformState:
            lambda state, params_axes: optax.MultiTransformState(
                inner_states=OptaxStatePartitionRules.derive_optax_logical_axes(
                    state.inner_states, params_axes)),
        # LookaheadState
        # SplitRealAndImaginaryState
    }
    # pylint: enable=g-long-lambda

    @classmethod
    def _is_optax_state(cls, x):
        """Returns true if an object is an optax state.

        Note that in optax states are simply derived from NamedTuple, so we have to
        do some hacky name matching.

        Args:
          x: object.

        Returns:
          True if x is an optax state.
        """
        # A solution from stack overflow. Note that isinstance(x, NamedTuple) would
        # not work.
        is_named_tuple = (
            isinstance(x, tuple) and hasattr(x, '_asdict') and
            hasattr(x, '_fields'))
        result = is_named_tuple and type(x).__name__.endswith('State')
        return result

    @classmethod
    def derive_optax_logical_axes(cls, optax_state, params_axes):
        """Derived logical axes for optax state."""
        # Flatten the optax state but do not go into the registered states.
        flattened_state, tree_def = jax.tree_util.tree_flatten(
            optax_state, is_leaf=cls._is_optax_state)

        def derive_fn(x):
            if type(x) not in cls._RULES:
                if cls._is_optax_state(x):
                    raise ValueError(
                        f'Encountered unregistered optax state type {type(x).__name__}')
                return None
            return cls._RULES[type(x)](x, params_axes)

        flattened_axes = [derive_fn(x) for x in flattened_state]
        derived_axes = jax.tree_util.tree_unflatten(tree_def, flattened_axes)
        return derived_axes

class OptimizerDef:
    """Base class for an optimizer definition."""

    def __init__(self):
        pass
    
    def apply_gradient(self, hyper_params, params, state, grads):
        """Applies a gradient for a set of parameters."""
        raise NotImplementedError()

    def init_state(self, params):
        raise NotImplementedError()

    def create(self, target):
        """Creates a new optimizer for the given target.

        Args:
          target: the object to be optimized. This is typically a variable dict
            returned by `flax.linen.Module.init()`, but it can also be a container
            of variables dicts, e.g. `(v1, v2)` and  `('var1': v1, 'var2': v2)` are
            valid inputs as well.

        Returns:
          An instance of `Optimizer`.
        """
        opt_def = self
        state = opt_def.init_state(target)
        return Optimizer(opt_def, state, target)

    def state_dict(self, target, state):
        return to_state_dict({
            'target': to_state_dict(target),
            'state': to_state_dict(state)
        })

    def restore_state(self, opt_target, opt_state, state_dict):
        """Restore the optimizer target and state from the state dict.

        Args:
          opt_target: the optimizer target.
          opt_state: the optimizer state.
          state_dict: the state dict containing the desired new state of the
            optimizer.

        Returns:
          a tuple of the optimizer target and state with the restored values from
          the state dict.
        """

        opt_target = from_state_dict(opt_target, state_dict['target'])
        opt_state = from_state_dict(opt_state, state_dict['state'])
        return opt_target, opt_state


class Optimizer(struct.PyTreeNode):
    """Legacy flax optimizer class.

    Optimizer carries the target and optimizer state. The optimizer is updated
    using the method apply_gradient.

    Attributes:
      optimizer_def: The optimizer definition.
      state: The initial state of the optimizer.
      target: The target to optimizer.
    """

    optimizer_def: OptimizerDef = struct.field(pytree_node=False)
    state: Any = struct.field(pytree_node=True) # state is the optimizer state
    target: Any = struct.field(pytree_node=True) # target is the model parameters

    def apply_gradient(self, grads):
        """Applies a pytree of gradients to the target.

        Args:
          grads: A pytree of gradients.
          **hyper_param_overrides: the hyper parameters passed to apply_gradient
            will override the defaults specified in the `OptimizerDef`. Pass
            `hyper_params=...` to replace all hyper parameters.

        Returns:
          A new optimizer with the updated target and state.
        """
        new_target, new_state = self.optimizer_def.apply_gradient(
            self.target,
            self.state,
            grads
        )
        return self.replace(target=new_target, state=new_state)

    def state_dict(self):
        return self.optimizer_def.state_dict(self.target, self.state)

    def restore_state(self, state):
        target, state = self.optimizer_def.restore_state(self.target, self.state,
                                                         state)
        return self.replace(target=target, state=state)


# Transitional Type Definitions

OptimizerType = Optimizer
OptimizerStateType = Union[OptimizerState, Mapping[str, Any]]
OptimizerDefType = OptimizerDef



class OptaxWrapper(OptimizerDef):
    """Wrapper to make optax optimizer compatible with T5X."""

    def __init__(self, optax_optimizer: optax.GradientTransformation):
        """Initializer.

        Args:
          optax_optimizer: An optax optimizer.
        """
        self.optax_optimizer = optax_optimizer
        super().__init__()

    def init_state(self, params):
        """Create initial state based on the params to optimize.

        Args:
          params: PyTree of parameters to optimize.

        Returns:
          Initial optimizer state.
        """
        state = OptimizerState(
            step=0, param_states=self.optax_optimizer.init(params))
        return state

    def apply_gradient(self, params, state, grads):
        """Applies gradient.

        Args:
          params: PyTree of the parameters.
          state: A named tuple containing the state of the optimizer.
          grads: PyTree of the gradients for the parameters.

        Returns:
          A tuple containing the new parameters and the new optimizer state.
        """
        updates, new_optax_state = self.optax_optimizer.update(
            grads, state.param_states, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, OptimizerState(
            step=state.step + 1, param_states=new_optax_state)

    def derive_logical_axes(self, optimizer, param_logical_axes):
        """Derives optimizer state logical axes from params logical axes.

        Args:
          optimizer: `optimizers.Optimizer` instance.
          param_logical_axes: A PyTree where each leaf is a t5x PartitionSpec.

        Returns:
          An `optimizers.Optimizer` instance, with all the leafs replaced by t5x
          PartitionSpec or None (no partition).
        """
        optimizer_logical_axes = jax.tree_map(lambda x: None,
                                              optimizer.state_dict())
        optimizer_logical_axes['target'] = param_logical_axes

        optax_state_axes = OptaxStatePartitionRules.derive_optax_logical_axes(
            optimizer.state.param_states, param_logical_axes)

        optimizer_logical_axes['state']['param_states'] = (
            serialization.to_state_dict(optax_state_axes))

        return optimizer.restore_state(frozen_dict.unfreeze(optimizer_logical_axes))

    def state_dict(self, target, state):
        """Override state dict function.

        We need to override this function because many optax transformations use
        `optax.EmptyState`, which produces empty dict in the state dict. This causes
        the T5 training loop to fail in multiple places. As a remedy, we will
        filter out the generated state dict so that there are no empty dict in the
        output.

        The restore_state function is also overridden to reconstruct those empty
        dict.

        Args:
          target: Pytree of target variables.
          state: Pytree of optimizer state.

        Returns:
          A nested state.
        """
        state_dict = to_state_dict(state)

        # This step removes any empty dict (recursively) in the state dict.
        state_dict = traverse_util.unflatten_dict(
            traverse_util.flatten_dict(state_dict, sep='/'), sep='/')

        return to_state_dict({
            'target': to_state_dict(target),
            'state': state_dict,
        })

    def restore_state(self, opt_target, opt_state, state_dict):
        """Override to restore empty dicts corresponding to `optax.EmptyState`.

        Args:
          opt_target: the optimizer target.
          opt_state: the optimizer state.
          state_dict: the state dict containing the desired new state of the
            optimizer.

        Returns:
          a tuple of the optimizer target and state with the restored values from
          the state dict.
        """
        opt_target = from_state_dict(opt_target, state_dict['target'])

        # Get all the possible keys in the reference optimizer state.
        flat_ref_opt_state_dict = traverse_util.flatten_dict(
            to_state_dict(opt_state), keep_empty_nodes=True, sep='/')

        flat_src_opt_state_dict = dict(
            traverse_util.flatten_dict(state_dict['state'], sep='/'))
        # Adding the empty paths back to flat_src_opt_state_dict.
        for k, v in flat_ref_opt_state_dict.items():
            if k in flat_src_opt_state_dict:
                continue
            # The key is not in the input state dict, presumably because it
            # corresponds to an empty dict.
            if v != traverse_util.empty_node:
                raise ValueError(
                    f'Failed to restore optimizer state, path {k} is not present '
                    'in the input optimizer state dict.')
            flat_src_opt_state_dict[k] = v

        # Restore state from the enhanced state dict.
        opt_state = from_state_dict(
            opt_state,
            traverse_util.unflatten_dict(flat_src_opt_state_dict, sep='/'))
        return opt_target, opt_state


# Optax wrapper and elementary wrapped optax optimizers.


def wrap_optax_optimizer(optax_optimizer):
    """Converts optax optimizer constructor to a wrapped T5X-compatible optimizer.

    Args:
      optax_optimizer: an optax optimizer creation function that returns an optax
        GradientTransformation.

    Returns:
      A function that takes the same arguments as the original optax creation
      function but instead returns a wrapped OptimizerDef-compatible interface for
      using the optimizer with T5X.
    """

    @functools.wraps(optax_optimizer)
    def wrapped_optimizer(*args, **kwargs) -> OptimizerDef:
        return OptaxWrapper(optax_optimizer(*args, **kwargs))

    return wrapped_optimizer


def chain(
    transformations: Sequence[optax.GradientTransformation]
) -> optax.GradientTransformation:
    return optax.chain(*transformations)


chain = wrap_optax_optimizer(chain)
adabelief = wrap_optax_optimizer(optax.adabelief)
adagrad = wrap_optax_optimizer(optax.adagrad)
adam = wrap_optax_optimizer(optax.adam)
adamw = wrap_optax_optimizer(optax.adamw)
amos = wrap_optax_optimizer(amos.amos)
fromage = wrap_optax_optimizer(optax.fromage)
lars = wrap_optax_optimizer(optax.lars)
lamb = wrap_optax_optimizer(optax.lamb)
noisy_sgd = wrap_optax_optimizer(optax.noisy_sgd)
radam = wrap_optax_optimizer(optax.radam)
rmsprop = wrap_optax_optimizer(optax.rmsprop)
sgd = wrap_optax_optimizer(optax.sgd)
yogi = wrap_optax_optimizer(optax.yogi)
dpsgd = wrap_optax_optimizer(optax.dpsgd)
