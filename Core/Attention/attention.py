# attention.py - CORRIGÉ
"""
Multi-Head Attention avec RoPE + YaRN + Flash Attention
✅ BUGS FIXÉS:
- Flash Attention masque correct (batch, heads, seq_len, seq_len)
- YaRN scaling identique Flash/Fallback
- QK-Norm avec RMSNorm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================
# RMSNorm (Root Mean Square Normalization)
# ============================================

class RMSNorm(nn.Module):
    """RMSNorm - Plus rapide et simple que LayerNorm"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ============================================
# RoPE + YaRN
# ============================================

class RotaryPositionalEmbedding(nn.Module):
    """RoPE (Rotary Position Embedding) avec YaRN"""
    def __init__(self, dim, max_seq_len=2048, base=10000, device=None,
                 use_yarn=False, yarn_scale=1.0, yarn_original_max_len=1024):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.use_yarn = use_yarn
        self.yarn_scale = yarn_scale
        self.yarn_original_max_len = yarn_original_max_len
        
        # ✅ VALIDATION YaRN SCALE
        if use_yarn:
            assert 0.1 <= yarn_scale <= 16.0, \
                f"yarn_scale must be in [0.1, 16.0], got {yarn_scale}"
        
        # Calculer les fréquences
        if use_yarn:
            inv_freq = self._compute_yarn_frequencies()
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None
    
    def _compute_yarn_frequencies(self):
        """Calcule les fréquences avec YaRN"""
        freqs = torch.arange(0, self.dim, 2).float() / self.dim
        inv_freq_base = 1.0 / (self.base ** freqs)
        
        if self.yarn_scale == 1.0:
            return inv_freq_base
        
        alpha = self.yarn_scale
        beta = 32
        
        dims = torch.arange(0, self.dim, 2).float()
        scale = torch.where(
            dims < beta,
            torch.ones_like(dims),
            1 + (alpha - 1) * (dims - beta) / (self.dim - beta)
        )
        
        inv_freq_yarn = inv_freq_base / scale
        return inv_freq_yarn
    
    def _update_cos_sin_cache(self, seq_len, device, dtype):
        """Met à jour le cache cos/sin si nécessaire"""
        if (seq_len != self._seq_len_cached or
            self._cos_cached is None or
            self._cos_cached.device != device or
            self._cos_cached.dtype != dtype):
            self._seq_len_cached = seq_len
            
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(dtype))
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
        
        return self._cos_cached, self._sin_cached
    
    def rotate_half(self, x):
        """Rotation de la moitié des dimensions"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q, k):
        """Applique RoPE/YaRN à Q et K"""
        seq_len = q.shape[2]
        cos, sin = self._update_cos_sin_cache(seq_len, q.device, q.dtype)
        
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_rot, k_rot
    
    def forward(self, q, k):
        return self.apply_rotary_pos_emb(q, k)


# ============================================
# Multi-Head Attention - CORRIGÉ
# ============================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention avec RoPE + YaRN + GQA + QK-Norm + Flash Attention
    
    ✅ BUGS FIXÉS:
    - Flash Attention masque correct [batch, heads, seq_len, seq_len]
    - YaRN scaling identique Flash/Fallback
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, 
                 use_rope=True, max_seq_len=2048,
                 use_yarn=False, yarn_scale=1.0, yarn_original_max_len=1024,
                 n_kv_heads=None, use_qk_norm=False, use_flash_attn=True):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim doit être divisible par num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.use_flash_attn = use_flash_attn
        
        # ✅ GQA
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else num_heads
        assert num_heads % self.n_kv_heads == 0, \
            f"num_heads ({num_heads}) doit être divisible par n_kv_heads ({self.n_kv_heads})"
        self.num_queries_per_kv = num_heads // self.n_kv_heads
        
        # Dimension KV
        self.kv_dim = self.n_kv_heads * self.head_dim
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.kv_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # ✅ QK-Norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        
        # RoPE
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                self.head_dim, 
                max_seq_len,
                use_yarn=use_yarn,
                yarn_scale=yarn_scale,
                yarn_original_max_len=yarn_original_max_len
            )
        else:
            self.rope = None
        
        # Flash Attention check
        self._flash_attn_available = False
        if use_flash_attn:
            try:
                F.scaled_dot_product_attention
                self._flash_attn_available = True
            except AttributeError:
                print("⚠️  Flash Attention non disponible (PyTorch < 2.0)")
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            mask: [seq_len, seq_len] - Masque causal BOOLEAN (True = masqué)
        
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Transpose
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # ✅ QK-Norm
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # RoPE
        if self.use_rope:
            q, k = self.rope(q, k)
        
        # ✅ GQA: Répéter K et V
        if self.n_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # ✅ FLASH ATTENTION ou fallback
        if self.use_flash_attn and self._flash_attn_available:
            # ✅ CORRIGÉ: Masque avec bonne shape [batch, heads, seq_len, seq_len]
            attn_mask = None
            if mask is not None:
                attn_mask = torch.zeros(
                    batch_size, self.num_heads, seq_len, seq_len,
                    dtype=q.dtype,
                    device=q.device
                )
                attn_mask.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # ✅ YaRN scaling (identique au fallback)
            scale = None
            if self.use_rope and self.rope.use_yarn and self.rope.yarn_scale > 1.0:
                scale = math.sqrt(self.rope.yarn_scale) / math.sqrt(self.head_dim)
            else:
                scale = 1.0 / math.sqrt(self.head_dim)
            
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=scale,
            )
        else:
            # ✅ FALLBACK: Attention standard
            scores = torch.matmul(q, k.transpose(-2, -1))
            
            # ✅ YaRN scaling (IDENTIQUE à Flash)
            if self.use_rope and self.rope.use_yarn and self.rope.yarn_scale > 1.0:
                scores = scores * math.sqrt(self.rope.yarn_scale) / math.sqrt(self.head_dim)
            else:
                scores = scores / math.sqrt(self.head_dim)
            
            # Masque
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # Softmax
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            
            output = torch.matmul(attn_weights, v)
        
        # Transpose et reshape
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, embed_dim)
        
        # Projection finale
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output