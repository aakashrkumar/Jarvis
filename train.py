import jax
import flax
import optax
from t5x.partitioning import PjitPartitioner
from palm_main import PaLM
from config import PaLMConfig

def train():
    config = PaLMConfig
    model = PaLM(config = config)
    
    