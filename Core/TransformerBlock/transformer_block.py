# transformer_block.py - CORRIGÉ
import torch
import torch.nn as nn

from attention import MultiHeadAttention, RMSNorm
from feedforward import FeedForward

class TransformerBlock(nn.Module):
    """
    Transformer Block avec RMSNorm + RoPE + SwiGLU + GQA + Flash Attention
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, 
                 use_rope=True, max_seq_len=2048,
                 use_yarn=False, yarn_scale=1.0, yarn_original_max_len=1024,
                 use_swiglu=True, n_kv_heads=None, use_qk_norm=False, use_flash_attn=True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_swiglu = use_swiglu
        self.n_kv_heads = n_kv_heads
        self.use_qk_norm = use_qk_norm
        self.use_flash_attn = use_flash_attn
        
        # RMSNorm
        self.ln1 = RMSNorm(embed_dim)
        
        # Multi-Head Attention
        self.attention = MultiHeadAttention(
            embed_dim, 
            num_heads, 
            dropout,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            use_yarn=use_yarn,
            yarn_scale=yarn_scale,
            yarn_original_max_len=yarn_original_max_len,
            n_kv_heads=n_kv_heads,
            use_qk_norm=use_qk_norm,
            use_flash_attn=use_flash_attn
        )
        
        # RMSNorm
        self.ln2 = RMSNorm(embed_dim)
        
        # Feed-Forward
        self.ffn = FeedForward(embed_dim, dropout, use_swiglu=use_swiglu)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            mask: [seq_len, seq_len] - Masque causal
        
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        # Attention block
        residual = x
        x = self.ln1(x)
        x = self.attention(x, mask)
        x = residual + x
        
        # Feed-Forward block
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


def create_causal_mask(seq_len, device='cpu'):
    """Crée un masque causal triangulaire"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask