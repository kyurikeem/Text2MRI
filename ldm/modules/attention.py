from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from ldm.modules.diffusionmodules.util import checkpoint
import numpy as np


try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x, context=None, mask=None):
        '''
        x.shape: (b,4096,128) Not Fixed size
        context.shape(b,77,768)
        q, k, v shape: torch.Size([b, 4096(64*64, hw), 128]) torch.Size([b, 77, 128]) torch.Size([b, 77, 128])
        -> q,k,v shape: (?, hw, 16=128/8=self.head), (?, 77, 16=128/8=self.head), (?, 77, 16=128/8=self.head)

        [Visualization]
        1. aggregate all attention map across the "timesteps" and "heads"
        2. Normalization divided by "max" with respecto to "each token"
        '''

        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        # print('v1.shape:', v.shape)                        # v: torch.Size([32, 77, 16])

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v)) # v: torch.Size([32, 77, 16])

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        _sim = sim.softmax(dim=-1)                           # attention_map torch.Size([32, 4096, 77])
        # print('_sim.shape:', _sim.shape)                    # _sim: torch.Size([32, 4096, 77]
        # print('v2.shape:', v.shape)                        # v: torch.Size([32, 77, 16])

        out = einsum('b i j, b j d -> b i d', _sim, v)       # cross-attention output (feature update) torch.Size([32, 4096, 16])
        # print('out1.shape:', out.shape)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h) # torch.Size([4, 4096, 128])
        # print('out2.shape:', out.shape)

        return self.to_out(out), _sim #out


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None): # context torch.Size([4, 77, 768])
        ''' 
        < Original Version >
            x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
            x = self.attn2(self.norm2(x), context=context) + x
            x = self.ff(self.norm3(x)) + x
            return x
        '''
        x_attention, self_attention_map = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None)  # self.disable_self_attn == False
        # x += x_attention
        x = x + x_attention 
        x_attention, cross_attention_map = self.attn2(self.norm2(x), context=context)
        # x += x_attention 
        x = x + x_attention 
        x = self.ff(self.norm3(x)) + x                     # torch.Size([4, 1024, 256])
        return x, self_attention_map, cross_attention_map  # torch.Size([32, 1024, 1024]), torch.Size([32, 1024, 77])


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear


    def average_map(self, attmap, token_idx=0):
        """
        ssim: [32, 4096, 77] -> [4*8, 4096, 77] -> [4, 8, 4096, 77]
        out: [4, 4096, 128] -> [4, 4096, 16*8] -> [4, 4096, 16, 8] -> [4, 8, 4096, 16]
        attmap.shape: similarity matrix.
        token_idx: index of token for visualizing, 77: [SOS, ...text..., EOS]
        """
        # ssim version (attention map)
        attmap = rearrange(attmap, '(b h) n d -> b h n d', h=8) # TODO: h == self.n_heads == 8 (4, 8, 4096, 77)
        
        # out version (cross attention map)
        # attmap = rearrange(attmap, 'b n (d h) -> b n d h', h=8)
        # attmap = rearrange(attmap, 'b n d h -> b h n d', h=8) 

        attmap_sm = F.softmax(attmap.float(), dim=-1) #F.softmax(torch.Tensor(attmap).float(), dim=-1) # (4, 8, hw, context_dim)
        att_map_mean = torch.mean(attmap_sm, dim=1)  # (4, hw, context_dim)

        b, hw, context_dim = att_map_mean.shape
        h = int(math.sqrt(hw))
        w = h        
        return att_map_mean.view(b,h,w,context_dim)  # (4, h, w, context_dim)

    def forward(self, x, context=None):
        '''
        note: if no context is given, cross-attention defaults to self-attention
        '''
        # print('context:', context.shape) # torch.Size([b, 77, 768]
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()

        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x, self_attention_map, cross_attention_map = block(x, context=context[i])
            # print('self_attention_map:', self_attention_map.shape)     # torch.Size([b, 4096, 128])
            # print('cross_attention_map:', cross_attention_map.shape)     # torch.Size([b, 4096, 128])

            # # Attention Map TODO: Saving OPTION !
            # import time
            # # if self.get_attention_map:
            # t = time.time()
            # self.save_dir = './attention_maps-mutant'

            # if cross_attention_map.shape[1] == 64:
            #     np.save(os.path.join(self.save_dir, "crossatt-64-" + str(t)), self.average_map(cross_attention_map).detach().cpu().numpy() )

            # if cross_attention_map.shape[1] == 256: 
            #     np.save(os.path.join(self.save_dir, "crossatt-256-" + str(t)), self.average_map(cross_attention_map).detach().cpu().numpy() )

            # if cross_attention_map.shape[1] == 1024: 
            #     np.save(os.path.join(self.save_dir, "crossatt-1024-" + str(t)), self.average_map(cross_attention_map).detach().cpu().numpy() )

            # if cross_attention_map.shape[1] == 4096: 
            #     np.save(os.path.join(self.save_dir, "crossatt-4096-" + str(t)), self.average_map(cross_attention_map).detach().cpu().numpy() )


        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        
        return x + x_in

