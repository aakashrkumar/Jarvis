from enum import Enum
from typing import List, Iterable, Optional, Union, Tuple, Dict, Any
import jax.numpy as jnp
from pydantic import BaseModel
from flax import struct

class PaLMConfig(struct.PyTreeNode):
    num_tokens:         int = 20000
    
    dim:                int = 512
    depth:              int = 10
    
    heads:              int = 8
    dim_head:           int = 64
    
    ff_mult:            int = 4
    
    lr:                 float = 1e-4
    
    seed:               int = 0
    batch_size:         int = 16