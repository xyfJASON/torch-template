import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange


def sinusoidal_embedding(
    idx: Tensor,
    dim: int,
    base: float = 10000,
    style: str = "cos-first",
) -> Tensor:
    """idx (*) -> embed (*, dim)"""
    assert dim % 2 == 0
    half_dim = dim // 2
    freqs = torch.arange(half_dim, dtype=torch.float32) / half_dim
    freqs = torch.exp(-math.log(base) * freqs).to(device=idx.device)
    embed = idx.float()[..., None] * freqs
    if style == "cos-first":
        embed = torch.cat([torch.cos(embed), torch.sin(embed)], dim=-1)
    elif style == "sin-first":
        embed = torch.cat([torch.sin(embed), torch.cos(embed)], dim=-1)
    else:
        raise ValueError(f"Unknown sinusoidal embedding style: {style}")
    return embed


def get_1d_sinusoidal_positional_embedding(
    seq_len: int,
    dim: int,
    base: float = 10000,
    max_seq_len: float = None,
    style: str = "cos-first",
) -> Tensor:
    """return embed (seq_len, dim)"""
    max_seq_len = max_seq_len or seq_len
    idx = torch.arange(seq_len, dtype=torch.float32) / seq_len * max_seq_len
    embed = sinusoidal_embedding(idx, dim, base=base, style=style)
    return embed


def get_2d_sinusoidal_positional_embedding(
    height: int,
    width: int,
    dim: int,
    base: float = 10000,
    max_height: float = None,
    max_width: float = None,
    style: str = "cos-first",
    width_first: bool = False,
) -> Tensor:
    """return embed (height, width, dim)"""
    max_height = max_height or height
    max_width = max_width or width
    grid_h = torch.arange(height, dtype=torch.float32) / height * max_height
    grid_w = torch.arange(width, dtype=torch.float32) / width * max_width
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    embed_h = sinusoidal_embedding(grid[0], dim // 2, base=base, style=style)
    embed_w = sinusoidal_embedding(grid[1], dim // 2, base=base, style=style)
    embed = torch.cat([embed_w, embed_h], dim=-1) if width_first else torch.cat([embed_h, embed_w], dim=-1)
    return embed


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", torch.ones(dim), persistent=False)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """x (B, L, D) -> (B, L, D)"""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop: float = 0.0,
        bias: bool = True,
        multiple_of: int = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if multiple_of is not None:
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """x (B, L, D) -> (B, L, D)"""
        return self.w2(self.ffn_dropout(F.silu(self.w1(x)) * self.w3(x)))


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, frequency_embedding_dim: int = 256, input_scaling: float = 1.0):
        super().__init__()
        self.frequency_embedding_dim = frequency_embedding_dim
        self.input_scaling = input_scaling
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

    def forward(self, t: Tensor) -> Tensor:
        """t (B, ) -> (B, D)"""
        t = t * self.input_scaling
        t_freq = sinusoidal_embedding(t, self.frequency_embedding_dim).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class PatchEmbedder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, patch_size: int, bias: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """x (B, C, H, W) -> (B, L=(H/P)*(W/P), D)"""
        x = self.proj(x)
        x = rearrange(x, "B D H W -> B (H W) D")
        return x


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
    ):
        super().__init__()
        self.num_heads = num_heads

        assert dim % num_heads == 0
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        """x (B, L, D) -> (B, L, D)"""
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        x = rearrange(x, "B H L D -> B L (H D)")

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        self.attn = SelfAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=False,
        )
        self.mlp = SwiGLUFFN(hidden_dim, int(hidden_dim * mlp_ratio))

    def forward(self, x: Tensor) -> Tensor:
        """x (B, L, D) -> (B, L, D)"""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(hidden_dim, elementwise_affine=False)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """x (B, L, D) -> (B, L, P*P*C)"""
        return self.linear(self.norm(x))


class DiT(nn.Module):
    def __init__(
        self,
        img_input_size: int,
        img_patch_size: int,
        img_in_channels: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.img_input_size = img_input_size
        self.img_patch_size = img_patch_size
        self.img_in_channels = img_in_channels
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.img_grid_size = img_input_size // img_patch_size

        # image embedding
        self.img_embedder = PatchEmbedder(img_in_channels, hidden_dim, img_patch_size)
        img_pe = get_2d_sinusoidal_positional_embedding(self.img_grid_size, self.img_grid_size, hidden_dim)
        self.register_buffer("img_pe", img_pe.reshape(self.img_grid_size ** 2, hidden_dim))

        # timestep embedding (repeated 16 times)
        self.t_embedder = TimestepEmbedder(hidden_dim, input_scaling=1000.0)
        t_pe = get_1d_sinusoidal_positional_embedding(16, hidden_dim)
        self.register_buffer("t_pe", t_pe)

        # transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            ) for _ in range(depth)
        ])

        # final layer
        self.final_layer = FinalLayer(hidden_dim, img_patch_size, img_in_channels)

        # initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # apply basic init
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # image embedding
        w = self.img_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.img_embedder.proj.bias, 0)

        # timestep embedding
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # final layer
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: Tensor) -> Tensor:
        """x (B, L, D=P*P*C) -> (B, H, W, C)"""
        c = self.img_in_channels
        p = self.img_patch_size
        h = w = self.img_grid_size
        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape((x.shape[0], c, h * p, h * p))
        return x

    def forward(self, img: Tensor, t: Tensor) -> Tensor:
        """img (B, C, H, W), t (B, ) -> (B, C, H, W)"""
        # image embedding
        img = self.img_embedder(img)
        img = img + self.img_pe.unsqueeze(0)
        L = img.shape[1]

        # timestep embedding (repeated 16 times)
        t = self.t_embedder(t).unsqueeze(1).repeat(1, 16, 1)
        t = t + self.t_pe.unsqueeze(0)
        T = t.shape[1]

        # transformer blocks
        x = torch.cat([img, t], dim=1)
        for block in self.blocks:
            x = block(x)

        # final layer
        x = self.final_layer(x)
        img, t = torch.split(x, [L, T], dim=1)

        img = self.unpatchify(img)
        return img
