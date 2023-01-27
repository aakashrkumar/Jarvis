"""
Codebase from Lucidrains' PaLM Model model
"""
import numpy
import flax.linen as nn
from einops import rearrange
import jax.numpy as jnp
from jax.numpy import einsum
from typing import Callable
from config import PaLMConfig
from flax.linen import partitioning as nn_partitioning

ATTN_MASK_VALUE = -1e10

scan_with_axes = nn_partitioning.scan_with_axes
remat = nn_partitioning.remat
ScanIn = nn_partitioning.ScanIn

class PreNorm(nn.Module):
    fn: Callable

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.LayerNorm(epsilon = 1e-5, use_bias = False)(x)
        return self.fn(x, **kwargs)

# rotary positional embedding
# https://arxiv.org/abs/2104.09864

class RotaryEmbedding(nn.Module):
    dim_head: int         

    @nn.compact
    def __call__(self, max_seq_len):
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.dim_head, 2) / self.dim_head))
        seq = jnp.arange(max_seq_len)
        freqs = einsum("i , j -> i j", seq, inv_freq)
        return jnp.concatenate((freqs, freqs), axis = -1)

def jax_unstack(x, axis = 0):
    return jnp.moveaxis(x, axis, 0)

def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j = 2)
    x1, x2 = jax_unstack(x, axis = -2)
    return jnp.concatenate((-x2, x1), axis = -1)

def apply_rotary_pos_emb(pos, t):
    return (t * jnp.cos(pos)) + (rotate_half(t) * jnp.sin(pos))

# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202

class SwiGLU(nn.Module):
    @nn.compact
    def __call__(self, x):
        x, gate = x.split(2, axis = -1)
        return jnp.multiply(nn.swish(gate), x)

# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame

class ParallelTransformerBlock(nn.Module):
    config: PaLMConfig

    @nn.compact
    def __call__(self, x):
        attn_inner_dim = self.config.dim_head * self.config.heads
        ff_inner_dim = self.config.dim * self.config.ff_mult
        fused_dims = (attn_inner_dim, self.config.dim_head, self.config.dim_head, (ff_inner_dim * 2))

        scale = self.config.dim_head ** -0.5

        n = x.shape[1]

        split_indices = numpy.cumsum(fused_dims[:-1])
        # attention queries, keys, values, and feedforward inner
        fused_attn_ff_proj = nn.Dense(features = sum(fused_dims), use_bias=False)(x) # kernel shape: (dim, sum(fused_dims))
        # print("Dense_0 kernel shape: ", (self.config.dim, sum(fused_dims)))
        q, k, v, ff = jnp.split(fused_attn_ff_proj, split_indices, axis = -1)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150
        q = rearrange(q, "b n (h d) -> b h n d", h = self.config.heads)
        
        # rotary embeddings
        positions = RotaryEmbedding(self.config.dim_head)(n)

        if positions is not None and positions.shape[-2] >= n:
            positions = positions[:n]
            
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale
        q = q * scale

        # similarity
        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # causal mask
        mask = jnp.tril(jnp.ones((n, n)))

        if mask is not None and mask.shape[-1] >= n:
            mask = mask[:n, :n]

        sim = jnp.where(mask, sim, ATTN_MASK_VALUE)

        # attention
        attn = nn.softmax(sim, axis = -1)

        # aggregate values
        attn_out = einsum("b h i j, b j d -> b h i d", attn, v)

        # attention out
        attn_out = rearrange(attn_out, "b h n d -> b n (h d)")
        attn_out = nn.Dense(self.config.dim, use_bias=False)(attn_out) # kernel shape: (sum(fused_dims), dim)
        # print("Dense_1 kernel shape: ", (sum(fused_dims), self.config.dim))


        # feedforward out
        ff_out = SwiGLU()(ff)
        ff_out = nn.Dense(self.config.dim, use_bias=False)(ff_out) # kernel shape: (sum(fused_dims), dim)
        # print("Dense_2 kernel shape: ", (sum(fused_dims), self.config.dim))

        # merge heads
        merge_heads = attn_out + ff_out
        return merge_heads

# transformer

class ParallelTransformer(nn.Module):
    config: PaLMConfig

    @nn.compact
    def __call__(self, x):
        layers = []
        for _ in range(self.config.depth):
            layers.append(
                PreNorm(ParallelTransformerBlock(config=self.config))
            )
        for block in layers:
            x = block(x) + x
        return x

# model

class PaLMModel(nn.Module): 
    config: PaLMConfig

    @nn.compact
    def __call__(self, x):
        embed = nn.Embed(num_embeddings=self.config.num_tokens, features=self.config.dim, embedding_init = nn.initializers.normal(stddev=0.02))
        x = embed(x)
        x = ParallelTransformer(config=self.config)(x)
        x = nn.LayerNorm(epsilon = 1e-5, use_bias = False)(x)
        out = embed.attend(x)
        return out    


if __name__ == "__main__":

    import jax
    import numpy

    key = jax.random.PRNGKey(0)

    seq = jax.random.randint(key, (1, 2048), 0, 20000)
    config = PaLMConfig(
        num_tokens = 20000,
        dim = 512,
        depth = 1,
        heads = 8,
        dim_head = 64
    )
    
    model = PaLMModel(
        config
    )

    init_rngs = {'params': jax.random.PRNGKey(1), 
                'dropout': jax.random.PRNGKey(2)}

    params = model.init(init_rngs, seq)
    output = model.apply(params, seq)
    print(output.shape) # (1, 2048, 20000)

    n_params_flax = sum(
        jax.tree_leaves(jax.tree_map(lambda x: numpy.prod(x.shape), params))
    )
    print(f"Number of parameters in Flax model: {n_params_flax}") # 55073280
