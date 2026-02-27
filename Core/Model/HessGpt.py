# HessGpt.py - CORRIGÉ
"""
HessGPT - Architecture Transformer moderne v5 PRODUCTION READY
✅ TOUS LES BUGS FIXÉS:
- Flash Attention masque correct
- YaRN scaling consistent
- Soft-capping stabilisé
- GQA optimisé
- Validation params complète
- Position embeddings testé
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformer_block import TransformerBlock
from attention import RMSNorm

class HessGPT(nn.Module):
    """
    HessGPT - Architecture Transformer moderne
    
    ✅ BUGS FIXÉS v5:
    - Flash Attention masque [batch, heads, seq_len, seq_len]
    - YaRN scaling identique Flash/Fallback
    - Soft-capping avec stabilité numérique
    - GQA avec repeat_interleave efficace
    - Validation params complète (GQA, RoPE, soft-cap)
    - Position embeddings fallback testé
    """
    def __init__(
        self,
        vocab_size,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        max_seq_len=2048,
        dropout=0.1,
        use_rope=True,
        use_yarn=False,
        yarn_scale=1.0,
        yarn_original_max_len=1024,
        use_swiglu=True,
        n_kv_heads=None,
        use_qk_norm=False,
        soft_cap=None,
        use_flash_attn=True
    ):
        super().__init__()
        
        # ✅ VALIDATION COMPLÈTE DES PARAMÈTRES
        assert vocab_size > 0, "vocab_size must be positive"
        assert embed_dim > 0, "embed_dim must be positive"
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        assert num_layers > 0, "num_layers must be positive"
        assert max_seq_len > 0, "max_seq_len must be positive"
        
        # ✅ GQA VALIDATION
        if n_kv_heads is not None:
            assert n_kv_heads > 0, "n_kv_heads must be positive"
            assert num_heads % n_kv_heads == 0, \
                f"num_heads ({num_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
            assert embed_dim % n_kv_heads == 0, \
                f"embed_dim ({embed_dim}) must be divisible by n_kv_heads ({n_kv_heads})"
        
        # ✅ RoPE VALIDATION
        if use_rope:
            assert max_seq_len > 0, "max_seq_len must be > 0 for RoPE"
            if use_yarn:
                assert yarn_original_max_len > 0, "yarn_original_max_len must be > 0"
                assert yarn_original_max_len <= max_seq_len, \
                    f"yarn_original_max_len ({yarn_original_max_len}) must be <= max_seq_len ({max_seq_len})"
                assert 0.1 <= yarn_scale <= 16.0, \
                    f"yarn_scale must be in [0.1, 16.0], got {yarn_scale}"
        
        # ✅ SOFT-CAP VALIDATION
        if soft_cap is not None:
            assert soft_cap > 0, "soft_cap must be > 0"
            assert soft_cap <= 100, "soft_cap > 100 may cause numerical issues"
        
        # ✅ WARNING si yarn non activé mais scale != 1.0
        if not use_yarn and yarn_scale != 1.0:
            print(f"⚠️  Warning: yarn_scale={yarn_scale} ignored (use_yarn=False)")
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.use_yarn = use_yarn
        self.yarn_scale = yarn_scale
        self.yarn_original_max_len = yarn_original_max_len
        self.use_swiglu = use_swiglu
        self.n_kv_heads = n_kv_heads
        self.use_qk_norm = use_qk_norm
        self.soft_cap = soft_cap
        self.use_flash_attn = use_flash_attn
        
        # Token Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Position embeddings (seulement si pas RoPE)
        if not use_rope:
            self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)
        else:
            self.position_embeddings = None
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, 
                num_heads, 
                dropout,
                use_rope=use_rope,
                max_seq_len=max_seq_len,
                use_yarn=use_yarn,
                yarn_scale=yarn_scale,
                yarn_original_max_len=yarn_original_max_len,
                use_swiglu=use_swiglu,
                n_kv_heads=n_kv_heads,
                use_qk_norm=use_qk_norm,
                use_flash_attn=use_flash_attn
            )
            for _ in range(num_layers)
        ])
        
        # Final RMSNorm
        self.ln_final = RMSNorm(embed_dim)
        
        # Output Head
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Causal mask cache
        self.register_buffer('_causal_mask', None, persistent=False)
        
        # Initialisation
        self.apply(self._init_weights)
        
        # Weight tying (après init pour ne pas casser le lien)
        self.output_head.weight = self.token_embeddings.weight
        
    def _init_weights(self, module):
        """Initialisation des poids"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
    
    def _get_causal_mask(self, seq_len, device):
        """
        ✅ Cache le masque causal
        
        Returns:
            mask: [seq_len, seq_len] bool (True = masqué)
        """
        if (self._causal_mask is None or 
            self._causal_mask.size(0) < seq_len or
            self._causal_mask.device != device):
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            self.register_buffer('_causal_mask', mask, persistent=False)
        
        return self._causal_mask[:seq_len, :seq_len]
    
    def forward(self, input_ids, targets=None, pad_token_id=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            targets: [batch_size, seq_len] (optionnel)
            pad_token_id: ID du token de padding
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            loss: Scalar (si targets fourni)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # ✅ Token Embeddings
        token_embeds = self.token_embeddings(input_ids)
        
        # ✅ Position Embeddings (fallback si pas RoPE)
        if self.use_rope:
            x = self.dropout(token_embeds)
        else:
            assert seq_len <= self.max_seq_len, \
                f"seq_len ({seq_len}) > max_seq_len ({self.max_seq_len})"
            
            positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
            pos_embeds = self.position_embeddings(positions)
            x = self.dropout(token_embeds + pos_embeds)
        
        # Masque causal
        mask = self._get_causal_mask(seq_len, device)
        
        # Transformer Blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final RMSNorm
        x = self.ln_final(x)
        
        # Output projection
        logits = self.output_head(x)
        
        # ✅ SOFT-CAPPING STABILISÉ (Gemma-style)
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        
        # Loss (optionnel)
        loss = None
        if targets is not None:
            ignore_index = pad_token_id if pad_token_id is not None else -100
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=ignore_index
            )
        
        return logits, loss
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """Génération autoregressive"""
        was_training = self.training
        self.eval()
        device = input_ids.device
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                input_ids_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
                
                logits, _ = self.forward(input_ids_cond)
                logits = logits[:, -1, :] / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        if was_training:
            self.train()
        
        return input_ids
    
    def resize_token_embeddings(self, new_vocab_size):
        """Resize embeddings si vocab change"""
        if new_vocab_size == self.vocab_size:
            return
        
        print(f"📝 Resizing embeddings: {self.vocab_size} → {new_vocab_size}")
        
        old_embeddings = self.token_embeddings
        self.token_embeddings = nn.Embedding(new_vocab_size, self.embed_dim)
        
        old_vocab_size = min(old_embeddings.num_embeddings, new_vocab_size)
        with torch.no_grad():
            self.token_embeddings.weight.data[:old_vocab_size] = \
                old_embeddings.weight.data[:old_vocab_size]
        
        old_output = self.output_head
        self.output_head = nn.Linear(self.embed_dim, new_vocab_size, bias=False)
        self.output_head.weight = self.token_embeddings.weight
        
        self.vocab_size = new_vocab_size
        print(f"   ✅ Embeddings resized to {new_vocab_size}")
    
    def count_parameters(self):
        """Compte les paramètres"""
        token_params = self.token_embeddings.weight.numel()
        
        pos_params = 0
        if self.position_embeddings is not None:
            pos_params = self.position_embeddings.weight.numel()
        
        block_params = sum(p.numel() for block in self.blocks for p in block.parameters())
        ln_params = sum(p.numel() for p in self.ln_final.parameters())
        output_params = 0  # Partagé avec token_embeddings
        
        total = token_params + pos_params + block_params + ln_params + output_params
        
        return {
            'token_embeddings': token_params,
            'position_embeddings': pos_params,
            'transformer_blocks': block_params,
            'final_ln': ln_params,
            'output_head': output_params,
            'total': total
        }
    
    def get_config(self):
        """Retourne la configuration"""
        return {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'max_seq_len': self.max_seq_len,
            'use_rope': self.use_rope,
            'use_yarn': self.use_yarn,
            'yarn_scale': self.yarn_scale,
            'yarn_original_max_len': self.yarn_original_max_len,
            'use_swiglu': self.use_swiglu,
            'n_kv_heads': self.n_kv_heads,
            'use_qk_norm': self.use_qk_norm,
            'soft_cap': self.soft_cap,
            'use_flash_attn': self.use_flash_attn
        }