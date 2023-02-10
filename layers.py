import dataclasses
import functools
import operator
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np


# from flax.linen.partitioning import param_with_axes, with_sharding_constraint
param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint


# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]
InitializerAxis = Union[int, Tuple[int, ...]]
NdInitializer = Callable[
    [PRNGKey, Shape, DType, InitializerAxis, InitializerAxis], Array]

default_embed_init = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal', out_axis=0)


# ------------------------------------------------------------------------------
# Temporary inlined JAX N-d initializer code
# TODO(levskaya): remove once new JAX release is out.
# ------------------------------------------------------------------------------
def _compute_fans(shape: jax.core.NamedShape, in_axis=-2, out_axis=-1):
    """Inlined JAX `nn.initializer._compute_fans`."""
    if isinstance(in_axis, int):
        in_size = shape[in_axis]
    else:
        in_size = int(np.prod([shape[i] for i in in_axis]))
    if isinstance(out_axis, int):
        out_size = shape[out_axis]
    else:
        out_size = int(np.prod([shape[i] for i in out_axis]))
    receptive_field_size = shape.total / in_size / out_size
    fan_in = in_size * receptive_field_size
    fan_out = out_size * receptive_field_size
    return fan_in, fan_out


def variance_scaling(scale, mode, distribution, in_axis=-2, out_axis=-1,
                     dtype=jnp.float_):
    """Inlined JAX `nn.initializer.variance_scaling`."""

    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        shape = jax.core.as_named_shape(shape)
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == 'fan_in':
            denominator = fan_in
        elif mode == 'fan_out':
            denominator = fan_out
        elif mode == 'fan_avg':
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                'invalid mode for variance scaling initializer: {}'.format(mode))
        variance = jnp.array(scale / denominator, dtype=dtype)

        if distribution == 'truncated_normal':
            # constant is stddev of standard normal truncated to (-2, 2)
            stddev = jnp.sqrt(variance) / jnp.array(.87962566103423978, dtype)
            return random.truncated_normal(key, -2, 2, shape, dtype) * stddev
        elif distribution == 'normal':
            return random.normal(key, shape, dtype) * jnp.sqrt(variance)
        elif distribution == 'uniform':
            return random.uniform(key, shape, dtype, -1) * jnp.sqrt(3 * variance)
        else:
            raise ValueError('invalid distribution for variance scaling '
                             'initializer: {}'.format(distribution))
    return init
# ------------------------------------------------------------------------------


def nd_dense_init(scale, mode, distribution):
    """Initializer with in_axis, out_axis set at call time."""
    def init_fn(key, shape, dtype, in_axis, out_axis):
        fn = variance_scaling(
            scale, mode, distribution, in_axis, out_axis)
        return fn(key, shape, dtype)
    return init_fn


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)

# ------------------------------------------------------------------------------
# DenseGeneral for attention layers.
# ------------------------------------------------------------------------------
class DenseGeneral(nn.Module):
    """A linear transformation (without bias) with flexible axes.
      Attributes:
        features: tuple with numbers of output features.
        axis: tuple with axes to apply the transformation on.
        dtype: the dtype of the computation (default: float32).
        kernel_init: initializer function for the weight matrix.
    """
    features: Union[Iterable[int], int]
    axis: Union[Iterable[int], int] = -1
    dtype: DType = jnp.float32
    kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'truncated_normal')
    kernel_axes: Tuple[str, ...] = ()

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along multiple dimensions.
        Args:
          inputs: The nd-array to be transformed.
        Returns:
          The transformed input.
        """
        features = _canonicalize_tuple(self.features)
        axis = _canonicalize_tuple(self.axis)

        inputs = jnp.asarray(inputs, self.dtype)
        axis = _normalize_axes(axis, inputs.ndim)

        kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + features
        kernel_in_axis = np.arange(len(axis))
        kernel_out_axis = np.arange(len(axis), len(axis) + len(features))
        kernel = param_with_axes(
            'kernel',
            self.kernel_init,
            kernel_shape,
            jnp.float32,
            kernel_in_axis,
            kernel_out_axis,
            axes=self.kernel_axes)
        kernel = jnp.asarray(kernel, self.dtype)

        contract_ind = tuple(range(0, len(axis)))
        return lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))

class LayerNorm(nn.Module):
  """T5 Layer normalization operating on the last axis of the input data."""
  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  scale_init: Initializer = nn.initializers.ones

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    scale = param_with_axes(
        'scale', self.scale_init, (features,), jnp.float32, axes=('embed',))

    scale = jnp.asarray(scale, self.dtype)
    return y * scale
