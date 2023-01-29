import logging
from jax.experimental.pjit import pjit
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union
from flax.linen import partitioning as flax_partitioning
from jax.experimental.maps import Mesh

from flax import traverse_util

TpuMesh = Tuple[int, int, int, int]  # (x, y, z, num_cores).
OtherMesh = Tuple[int, int]
HardwareMesh = Union[TpuMesh, OtherMesh]
LogicalAxisRules = Sequence[Tuple[str, Optional[str]]]

PartitionedCallable = Callable[..., Any]

# T5X pjit partitioning
class PjittedFnWithContext(PartitionedCallable):
    """Wraps pjitted function to apply the appropriate contexts."""

    def __init__(self,
                 pjitted_fn,
                 partition_mesh: Mesh,
                 logical_axis_rules: flax_partitioning.LogicalRules = ()):
        self._pjitted_fn = pjitted_fn
        self._mesh = partition_mesh
        self._logical_axis_rules = logical_axis_rules

    def __call__(self, *args):
        with Mesh(self._mesh.devices,
                  self._mesh.axis_names), flax_partitioning.axis_rules(
                      self._logical_axis_rules):
            return self._pjitted_fn(*args)

    def lower(self, *args):
        with Mesh(self._mesh.devices, self._mesh.axis_names), flax_partitioning.axis_rules(self._logical_axis_rules):
            return self._pjitted_fn.lower(*args)


class Partitioner:
    def __init__(
        self,
        num_partitions: Optional[int] = None,
        model_parallel_submesh: Optional[HardwareMesh] = None,
        params_on_devices: bool = True,
        backend: Optional[str] = None,
        logical_axis_rules: Optional[LogicalAxisRules] = None,
        use_cpu_pjit: Optional[bool] = False
    ):
        if not num_partitions and not model_parallel_submesh:
            raise ValueError('At least one of `num_partitions` or '
                       '`model_parallel_submesh` must be set.')

        if model_parallel_submesh is not None and len(model_parallel_submesh) != 4:
            logging.error(
                '`model_parallel_submesh` must be either None or a 4-tuple. Got '
                'Got `num_partitions=%s`. A ValueError will be raised beginning '
                'March 1, 2022.', model_parallel_submesh)

        if bool(num_partitions) and bool(model_parallel_submesh):
            logging.error(
                'At most one of `num_partitions` or `model_parallel_submesh` can be '
                'set. Got `num_partitions=%s` and `model_parallel_submesh`=%s. A '
                'ValueError will be raised beginning March 21, 2022.', num_partitions,
                model_parallel_submesh)
        
        self._num_partitions = num_partitions
        self._model_parallel_submesh = model_parallel_submesh
        self._params_on_devices = params_on_devices
        self._data_axis = 'data'
        self._backend = backend

        assert logical_axis_rules is not None, "logical_axis_rules must be set."
                
        self._logical_axis_rules = tuple(logical_axis_rules)
        self._data_axis, = flax_partitioning.logical_to_mesh_axes(
            ['batch'], logical_axis_rules)
        self._use_cpu_pjit = use_cpu_pjit

    def partition(
        self,
        fn: Callable,  # pylint: disable=g-bare-generic
        in_axis_resources,
        out_axis_resources,
        static_argnums: Union[int, Sequence[int]] = (),
        donate_argnums: Union[int, Sequence[int]] = ()
    ):
        pjitted = pjit(
            fn,
            in_axis_resources=in_axis_resources,
            out_axis_resources=out_axis_resources,
            static_argnums=static_argnums,
            donate_argnums=donate_argnums,
            backend=self._backend
        )

        return PjittedFnWithContext(pjitted, self.mesh, self._logical_axis_rules)
    def get_mesh_axes(self, train_state: TrainState) -> TrainState:
        """Returns a copy of TrainState with Optional[PartitionSpecs] as leaves."""
        logical_axes = self.get_logical_axes(train_state)

        def _logical_to_mesh_axes(param_name, logical_axes):
            if logical_axes is None:
                return None
            elif logical_axes is traverse_util.empty_node:
                return traverse_util.empty_node
            try:
                return flax_partitioning.logical_to_mesh_axes(logical_axes,
                                                            self._logical_axis_rules)
            except ValueError as e:
                raise ValueError(f'Failed to map logical axes for {param_name}') from e

        flat_logical_axes = traverse_util.flatten_dict(
            logical_axes.state_dict(), keep_empty_nodes=True, sep='/')
        flat_mesh_axes = {
            k: _logical_to_mesh_axes(k, v) for k, v in flat_logical_axes.items()
        }
        
        return logical_axes.restore_state(
            traverse_util.unflatten_dict(flat_mesh_axes, sep='/'))
