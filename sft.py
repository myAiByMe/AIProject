#!/usr/bin/env python3
"""
🔥 HessGPT - SFT avec LoRA v6 (LLAMA-3 NATIVE CHAT TOKENS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ LLAMA-3 TOKENIZER (128k vocab - meilleure compression anglais)
✅ Tokens de chat NATIFS LLaMA-3 (embeddings déjà initialisés)
✅ Response-only loss masking — MULTI-TURN CORRECT
✅ LoRA Rank 64, Alpha 128 (ALL modules: Q,K,V,O + MLP)
✅ YaRN activé : 1024 → 4096 tokens (extension x4)
✅ Dynamic padding (NO WASTE)
✅ 5 datasets SFT premium (~150k samples total)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FORMAT DE CONVERSATION (LLaMA-3 natif):
  <|begin_of_text|>
  <|start_header_id|>system<|end_header_id|>
  You are helpful.<|eot_id|>
  <|start_header_id|>user<|end_header_id|>
  Hello<|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>
  Hi!<|eot_id|>

TOKENS CUSTOM (pas d'équivalent natif):
  <code>   — délimite les blocs de code dans les réponses
  <think>  — début du raisonnement chain-of-thought
  </think> — fin du raisonnement chain-of-thought

DATASETS SFT (~150k total):
- SmolTalk:              70,000 (polite and concise tone)
- WikiHow SFT:           20,000 (task decomposition)
- Tulu 3 SFT Personas:   30,000 (stable personality)
- LongAlign-10k:         10,000 (YaRN calibration)
- Magpie-Reasoning-V2:   20,000 (DeepSeek-R1 <think> traces, chain-of-thought)

USAGE:
    python sft_hessgpt_lora.py
    python sft_hessgpt_lora.py --pretrain-checkpoint ./checkpoints/HessGpt_pretrain.pt
    python sft_hessgpt_lora.py --dry-run
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
import os
import time
import math
import json
import gc
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from datetime import datetime
import traceback
from typing import Dict, List, Optional, Tuple
import re
import warnings
warnings.filterwarnings('ignore')

sys.path.append('./Core/Model')

# ============================================
# TOKENS SPÉCIAUX CUSTOM
# Les tokens de chat LLaMA-3 (<|start_header_id|>, <|end_header_id|>, <|eot_id|>,
# <|begin_of_text|>, <|end_of_text|>) sont NATIFS dans le vocab LLaMA-3 base.
# On n'ajoute que les tokens sans équivalent natif.
# ============================================
CUSTOM_TOKEN_STRINGS = [
    '<code>',    # délimite les blocs de code
    '<think>',   # début du raisonnement chain-of-thought
    '</think>',  # fin du raisonnement chain-of-thought
]

# IDs des tokens natifs LLaMA-3 (connus, stables)
# Ces valeurs sont fixes dans Meta-Llama-3-8B — pas besoin de les résoudre dynamiquement
LLAMA3_NATIVE = {
    'bos':            '<|begin_of_text|>',   # 128000
    'eos':            '<|end_of_text|>',     # 128001
    'start_header':   '<|start_header_id|>', # 128006
    'end_header':     '<|end_header_id|>',   # 128007
    'eot':            '<|eot_id|>',          # 128009
}

# Sera rempli dynamiquement dans main() avec les IDs réels
SPECIAL_TOKENS: Dict[str, int] = {}

# ============================================
# ARGS
# ============================================
parser = argparse.ArgumentParser(description='HessGPT SFT with LoRA v5')
parser.add_argument('--pretrain-checkpoint', type=str, 
                    default='./checkpoints/HessGpt_pretrain.pt',
                    help='Checkpoint du pré-entraînement')
parser.add_argument('--output-checkpoint', type=str,
                    default='./checkpoints/HessGpt_sft_lora.pt',
                    help='Checkpoint de sortie SFT')
parser.add_argument('--dry-run', action='store_true',
                    help='Vérifie les datasets sans entraîner')
parser.add_argument('--num-samples', type=int, default=None,
                    help='Limite le nombre de samples (pour test rapide)')
args = parser.parse_args()

print("=" * 80)
print("🔥 HessGPT v6 — SFT LoRA + LLaMA-3 Native Chat Tokens")
print("   LLaMA-3 128k natif | YaRN 4x | LoRA R64 | Dynamic Padding | Response-only")
print("=" * 80)

# ============================================
# CONFIGURATION SFT v6
# ============================================
CONFIG = {
    # --- Model (YaRN Extension) ---
    # vocab_size sera mis à jour dynamiquement après tokenizer load
    # LLaMA-3 128256 + 3 custom tokens (<code>, <think>, </think>)
    'vocab_size':    128256 + len(CUSTOM_TOKEN_STRINGS),
    'embed_dim':     1280,
    'num_heads':     20,
    'num_layers':    22,
    'max_seq_len':   4096,
    'dropout':       0.1,
    'use_rope':      True,
    'use_yarn':      True,
    'yarn_scale':    4.0,
    'yarn_original_max_len': 1024,
    'use_swiglu':    True,
    'n_kv_heads':    5,
    'use_qk_norm':   True,
    'soft_cap':      30.0,
    'use_flash_attn': True,

    # --- LoRA ---
    'lora_r':        64,
    'lora_alpha':    128,
    'lora_dropout':  0.05,
    'lora_target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                             'gate_proj', 'up_proj', 'down_proj'],

    # --- Training ---
    'batch_size':              8,
    'gradient_accumulation':   4,
    'max_grad_norm':           1.0,
    'learning_rate':           2e-4,
    'weight_decay':            0.01,
    'adam_beta1':              0.9,
    'adam_beta2':              0.95,
    'adam_eps':                1e-8,

    # --- Data ---
    'val_split':     0.05,

    # --- LR Schedule ---
    'warmup_ratio':  0.05,
    'num_epochs':    3,

    # --- Validation ---
    'validate_every_steps': 200,
    'val_batches':          50,

    # --- Checkpoint ---
    'save_every_epochs':   1,

    # --- System ---
    'use_compile':    False,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\n📊 CONFIGURATION SFT v6:")
print(f"   Vocab size : {CONFIG['vocab_size']:,} (LLaMA-3 128k + 3 custom tokens)")
print(f"   Max seq len: {CONFIG['max_seq_len']} (YaRN x4)")
print(f"   YaRN scale : {CONFIG['yarn_scale']}")
print(f"   Q heads    : {CONFIG['num_heads']}")
print(f"   KV heads   : {CONFIG['n_kv_heads']} (GQA)")
print(f"   LoRA rank  : {CONFIG['lora_r']}")
print(f"   LoRA alpha : {CONFIG['lora_alpha']}")
print(f"   LoRA modules: {len(CONFIG['lora_target_modules'])} (ALL)")
print(f"   Dropout    : {CONFIG['dropout']}")
print(f"   LR         : {CONFIG['learning_rate']:.0e}")

print(f"\n🗣️  TOKENS NATIFS LLaMA-3 (chat format — embeddings déjà initialisés):")
for k, v in LLAMA3_NATIVE.items():
    print(f"   {v}")
print(f"\n🔧 TOKENS CUSTOM (3 — ajoutés après vocab natif):")
for tok in CUSTOM_TOKEN_STRINGS:
    print(f"   {tok}")

# ============================================
# LORA LAYER
# ============================================
class LoRALayer(nn.Module):
    """LoRA: Low-Rank Adaptation"""
    def __init__(self, in_features, out_features, r=1, alpha=1, dropout=0.05):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        self.dropout = nn.Dropout(dropout)
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        x = self.dropout(x)
        result = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return result * self.scaling


class LinearWithLoRA(nn.Module):
    """Linear layer with LoRA adaptation"""
    def __init__(self, base_layer, r=1, alpha=1, dropout=0.05):
        super().__init__()
        self.base_layer = base_layer
        self.lora = LoRALayer(
            base_layer.in_features,
            base_layer.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout
        )
    
    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = self.lora(x)
        return base_out + lora_out


def apply_lora_to_model(model, r=1, alpha=1, dropout=0.05, target_modules=['q_proj', 'v_proj']):
    """Applique LoRA aux modules cibles"""
    print(f"\n🔧 Applying LoRA (r={r}, alpha={alpha})...")
    
    for param in model.parameters():
        param.requires_grad = False
    
    lora_params = 0
    
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                
                lora_layer = LinearWithLoRA(module, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, lora_layer)
                
                lora_params += r * (module.in_features + module.out_features)
                
                print(f"   ✅ {name}: {module.in_features} → {module.out_features} "
                      f"(+{r * (module.in_features + module.out_features)} LoRA params)")
    
    print(f"\n📊 LoRA Statistics:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total params:     {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")
    print(f"   Trainable ratio:  {trainable_params/total_params*100:.2f}%")
    
    return model, trainable_params


# ============================================
# DATASET SFT — FORMAT LLAMA-3 NATIF
# ============================================
class SFTDataset(Dataset):
    """
    Dataset SFT avec format LLaMA-3 natif.

    Format de chaque séquence :
      <|begin_of_text|>
      <|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>
      <|start_header_id|>user<|end_header_id|>\n{user}<|eot_id|>
      <|start_header_id|>assistant<|end_header_id|>\n{response}<|eot_id|>

    Masking : on entraîne uniquement sur les tokens de réponse assistant,
    entre <|end_header_id|> et <|eot_id|> pour le tour assistant.
    Machine à états : PROMPT → IN_ASSISTANT_HEADER → IN_ASSISTANT_BODY → PROMPT → ...
    """
    def __init__(self, data, tokenizer, max_seq_len=4096):
        self.data        = data
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        text   = self._format_llama3(sample)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        input_ids  = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:],  dtype=torch.long)

        # IDs natifs LLaMA-3 — résolus dans main() et stockés dans SPECIAL_TOKENS
        start_header_id = SPECIAL_TOKENS['<|start_header_id|>']
        end_header_id   = SPECIAL_TOKENS['<|end_header_id|>']
        eot_id          = SPECIAL_TOKENS['<|eot_id|>']

        # Machine à états pour le masking multi-turn
        # States: 'prompt' | 'in_header' | 'in_assistant_body'
        labels = torch.full_like(target_ids, -100)
        state  = 'prompt'
        pending_header_tokens = []  # accumule les tokens entre start_header et end_header

        for i, token_id in enumerate(input_ids):
            token_id = token_id.item()

            if state == 'prompt':
                if token_id == start_header_id:
                    state = 'in_header'
                    pending_header_tokens = []

            elif state == 'in_header':
                if token_id == end_header_id:
                    # Décode les tokens du header pour savoir si c'est 'assistant'
                    header_text = self.tokenizer.decode(pending_header_tokens).strip()
                    if header_text == 'assistant':
                        state = 'in_assistant_body'
                    else:
                        state = 'prompt'
                    pending_header_tokens = []
                else:
                    pending_header_tokens.append(token_id)

            elif state == 'in_assistant_body':
                if token_id == eot_id:
                    # On inclut le <|eot_id|> dans la loss (le modèle doit apprendre à s'arrêter)
                    labels[i] = target_ids[i]
                    state = 'prompt'
                else:
                    labels[i] = target_ids[i]

        return input_ids, labels

    def _format_llama3(self, sample):
        """Formate un sample en format LLaMA-3 natif."""
        if 'messages' in sample:
            return self._format_from_messages(sample['messages'])

        if 'instruction' in sample and 'output' in sample:
            system    = sample.get('system', 'You are a helpful assistant.')
            user      = sample['instruction']
            assistant = sample['output']
            return self._build_conversation(system, [(user, assistant)])

        system    = sample.get('system', 'You are a helpful assistant.')
        user      = sample.get('user', sample.get('prompt', ''))
        assistant = sample.get('assistant', sample.get('response', ''))
        return self._build_conversation(system, [(user, assistant)])

    def _build_conversation(self, system: str, turns: list) -> str:
        """Construit une séquence complète au format LLaMA-3."""
        bos   = '<|begin_of_text|>'
        sh    = '<|start_header_id|>'
        eh    = '<|end_header_id|>'
        eot   = '<|eot_id|>'

        result = bos
        result += f"{sh}system{eh}\n{system}{eot}"
        for user_msg, assistant_msg in turns:
            result += f"{sh}user{eh}\n{user_msg}{eot}"
            result += f"{sh}assistant{eh}\n{assistant_msg}{eot}"
        return result

    def _format_from_messages(self, messages) -> str:
        """Formate depuis une liste de messages OpenAI-style."""
        bos   = '<|begin_of_text|>'
        sh    = '<|start_header_id|>'
        eh    = '<|end_header_id|>'
        eot   = '<|eot_id|>'

        result = bos
        for msg in messages:
            role    = msg.get('role', '')
            content = msg.get('content', '')
            if role in ('system', 'user', 'assistant'):
                result += f"{sh}{role}{eh}\n{content}{eot}"
        return result


# ✅ FIX v5: Dynamic padding collate function — pad_token_id injecté depuis main()
def make_collate_fn(pad_token_id: int):
    """
    Collate function with DYNAMIC padding.
    Pads to max length in THIS batch, not global max_seq_len.
    pad_token_id est celui du tokenizer (LLaMA-3: eos_token_id, pas 0).
    """
    def collate_fn(batch):
        input_ids = [item[0] for item in batch]
        labels    = [item[1] for item in batch]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        return input_ids, labels
    return collate_fn


def load_sft_datasets(tokenizer, max_seq_len=4096, num_samples_limit=None):
    """Charge les 5 datasets SFT (150k total)"""
    print(f"\n📥 Loading SFT datasets...")
    
    all_datasets = []
    
    # 1. SmolTalk (70k)
    print(f"\n1️⃣  SmolTalk (70k)...")
    try:
        smol_magpie = load_dataset("HuggingFaceTB/smol-magpie-ultra", split="train")
        smol_magpie = smol_magpie.select(range(min(40000, len(smol_magpie))))
        
        everyday = load_dataset("HuggingFaceTB/everyday-conversations", split="train")
        everyday = everyday.select(range(min(30000, len(everyday))))
        
        smoltalk = concatenate_datasets([smol_magpie, everyday])
        all_datasets.append(smoltalk)
        
        print(f"   ✅ SmolTalk: {len(smoltalk):,} samples")
    except Exception as e:
        print(f"   ❌ Erreur SmolTalk: {e}")
    
    # 2. WikiHow SFT (20k)
    print(f"\n2️⃣  WikiHow SFT (20k)...")
    try:
        wikihow = load_dataset("b-mc2/wikihow_lists", split="train")
        wikihow = wikihow.select(range(min(20000, len(wikihow))))
        all_datasets.append(wikihow)
        
        print(f"   ✅ WikiHow: {len(wikihow):,} samples")
    except Exception as e:
        print(f"   ❌ Erreur WikiHow: {e}")
    
    # 3. Tulu 3 SFT Personas (30k)
    print(f"\n3️⃣  Tulu 3 SFT Personas (30k)...")
    try:
        tulu = load_dataset("allenai/tulu-3-sft-personas", split="train")
        tulu = tulu.select(range(min(30000, len(tulu))))
        all_datasets.append(tulu)
        
        print(f"   ✅ Tulu 3: {len(tulu):,} samples")
    except Exception as e:
        print(f"   ❌ Erreur Tulu 3: {e}")
    
    # 4. LongAlign-10k (YaRN calibration)
    print(f"\n4️⃣  LongAlign-10k (10k - YaRN calibration)...")
    try:
        longalign = load_dataset("THUDM/LongAlign-10k", split="train")
        longalign = longalign.select(range(min(10000, len(longalign))))
        all_datasets.append(longalign)
        
        print(f"   ✅ LongAlign: {len(longalign):,} samples")
    except Exception as e:
        print(f"   ❌ Erreur LongAlign: {e}")
    
    # 5. Magpie-Reasoning-V2 (20k - DeepSeek-R1 <think> traces)
    print(f"\n5️⃣  Magpie-Reasoning-V2 (20k - Chain-of-Thought <think> traces)...")
    try:
        magpie_raw = load_dataset("Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B", split="train")
        magpie_raw = magpie_raw.select(range(min(20000, len(magpie_raw))))
        
        # Convert to unified format with <think> tags in assistant response
        def format_magpie_reasoning(example):
            """
            Magpie-Reasoning-V2 schema:
              - 'instruction'   : user question
              - 'response'      : full assistant answer (may already contain <think>...</think>)
              - 'thinking'      : isolated chain-of-thought (if present as separate field)
            We reconstruct: <think>{thinking}</think>\n{response_without_think}
            """
            instruction = example.get('instruction', example.get('prompt', ''))
            response    = example.get('response', example.get('output', ''))
            thinking    = example.get('thinking', None)

            # If the dataset exposes a separate 'thinking' field, wrap it explicitly
            if thinking and thinking.strip():
                # Strip any pre-existing <think> block from response to avoid duplication
                response_clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
                assistant_text = f"<think>{thinking.strip()}</think>\n{response_clean}"
            else:
                # The <think> block is already embedded in 'response' — keep as-is
                assistant_text = response

            return {
                'messages': [
                    {'role': 'system',    'content': 'You are a helpful assistant. Think step by step inside <think> tags before answering.'},
                    {'role': 'user',      'content': instruction},
                    {'role': 'assistant', 'content': assistant_text},
                ]
            }
        
        magpie = magpie_raw.map(format_magpie_reasoning, remove_columns=magpie_raw.column_names)
        all_datasets.append(magpie)
        
        print(f"   ✅ Magpie-Reasoning-V2: {len(magpie):,} samples  🧠 <think> traces inclus")
    except Exception as e:
        print(f"   ❌ Erreur Magpie-Reasoning-V2: {e}")
    
    if len(all_datasets) == 0:
        raise ValueError("Aucun dataset chargé !")
    
    full_dataset = concatenate_datasets(all_datasets)
    full_dataset = full_dataset.shuffle(seed=42)
    
    if num_samples_limit is not None:
        full_dataset = full_dataset.select(range(min(num_samples_limit, len(full_dataset))))
    
    print(f"\n✅ Total dataset: {len(full_dataset):,} samples  (target ~150k)")
    
    # Split train/val
    val_size = int(len(full_dataset) * CONFIG['val_split'])
    train_size = len(full_dataset) - val_size
    
    train_data = full_dataset.select(range(train_size))
    val_data = full_dataset.select(range(train_size, len(full_dataset)))
    
    print(f"   Train: {len(train_data):,} samples")
    print(f"   Val:   {len(val_data):,} samples")
    
    # Create PyTorch datasets
    train_dataset = SFTDataset(train_data, tokenizer, max_seq_len)
    val_dataset = SFTDataset(val_data, tokenizer, max_seq_len)
    
    return train_dataset, val_dataset


# ============================================
# WSD SCHEDULER
# ============================================
class WSDScheduler:
    """Warmup – Stable – Decay"""
    def __init__(self, optimizer, max_lr, total_steps,
                 warmup_ratio=0.05, decay_ratio=0.1, min_lr_ratio=0.1):
        self.optimizer   = optimizer
        self.max_lr      = max_lr
        self.min_lr      = max_lr * min_lr_ratio
        self.total_steps = total_steps

        self.warmup_steps = int(total_steps * warmup_ratio)
        self.decay_steps  = int(total_steps * decay_ratio)
        self.stable_steps = total_steps - self.warmup_steps - self.decay_steps

        self.current_step = 0

        print(f"\n📈 WSD LR SCHEDULE:")
        print(f"   ├─ Warmup : {self.warmup_steps:>8,} steps ({warmup_ratio*100:.1f}%)")
        print(f"   ├─ Stable : {self.stable_steps:>8,} steps ({self.stable_steps/total_steps*100:.1f}%)")
        print(f"   ├─ Decay  : {self.decay_steps:>8,} steps ({decay_ratio*100:.1f}%)")
        print(f"   └─ Total  : {self.total_steps:>8,} steps")
        print(f"   LR: {self.min_lr:.2e} → {self.max_lr:.2e}")

    def get_lr(self):
        step = self.current_step

        if step < self.warmup_steps:
            return self.max_lr * (step / max(self.warmup_steps, 1))
        elif step < self.warmup_steps + self.stable_steps:
            return self.max_lr
        else:
            decay_step = step - self.warmup_steps - self.stable_steps
            progress   = min(decay_step / max(self.decay_steps, 1), 1.0)
            cosine     = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine

    def step(self):
        lr = self.get_lr()
        self.current_step += 1
        
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def get_last_lr(self):
        return [self.get_lr()]

    def state_dict(self):
        return {'current_step': self.current_step}

    def load_state_dict(self, sd):
        self.current_step = sd['current_step']


# ============================================
# VALIDATION
# ============================================
@torch.no_grad()
def validate(model, val_loader, device, pad_token_id, max_batches=50):
    """Validation loss"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    try:
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
            
            x = x.to(device)
            y = y.to(device)
            
            autocast_device = 'cuda' if device == 'cuda' else 'cpu'
            with torch.amp.autocast(autocast_device, dtype=torch.bfloat16):
                _, loss = model(x, targets=y, pad_token_id=pad_token_id)
            
            total_loss += loss.item()
            num_batches += 1
    finally:
        model.train()  # ✅ FIX v5: toujours restauré, même si exception

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = math.exp(min(avg_loss, 10))
    
    return perplexity, avg_loss


# ============================================
# CHECKPOINT
# ============================================
class CheckpointManager:
    def __init__(self, path):
        self.path = path
        dir_name = os.path.dirname(path)
        if dir_name:  # ✅ FIX v5: évite makedirs('') qui lève une erreur
            os.makedirs(dir_name, exist_ok=True)

    def save(self, model, optimizer, scheduler, metadata):
        lora_state = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                lora_state[name] = param.data.cpu()
        
        checkpoint = {
            'lora_state_dict':      lora_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch':                metadata['epoch'],
            'global_step':          metadata['global_step'],
            'training_history':     metadata['training_history'],
            'config':               CONFIG,
            'last_save':            datetime.now().isoformat(),
        }
        
        tmp = self.path + '.tmp'
        torch.save(checkpoint, tmp)
        os.replace(tmp, self.path)
        print(f"      💾 Checkpoint → {self.path}")

    def load(self):
        if not os.path.exists(self.path):
            return None
        print(f"\n📂 Checkpoint: {self.path}")
        # weights_only=False nécessaire car le checkpoint contient CONFIG (dict Python)
        # et training_history (objets non-tenseurs). Sécurisé car fichier généré par nous.
        cp = torch.load(self.path, map_location='cpu', weights_only=False)
        print(f"   ✅ Epoch: {cp['epoch']}")
        print(f"   ✅ Step:  {cp['global_step']:,}")
        return cp


# ============================================
# TRAIN EPOCH
# ============================================
def train_epoch(
    model, train_loader, optimizer, scheduler,
    val_loader, checkpoint_manager, training_history,
    epoch, global_step
):
    """Entraîne une epoch"""
    print(f"\n{'=' * 80}")
    print(f"📦 EPOCH {epoch + 1}/{CONFIG['num_epochs']}")
    print(f"   LR: {scheduler.get_last_lr()[0]:.2e}")
    print(f"{'=' * 80}")
    
    model.train()
    epoch_loss = 0.0
    valid_batches = 0
    t_start = time.time()
    accumulated_steps = 0  # ✅ FIX v5: tracker pour flush fin d'epoch
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['num_epochs']}", leave=True)
    autocast_device = 'cuda' if device == 'cuda' else 'cpu'  # ✅ FIX v5: device dynamique
    
    for batch_idx, (x, y) in enumerate(pbar):
        try:
            x = x.to(device)
            y = y.to(device)
            
            with torch.amp.autocast(autocast_device, dtype=torch.bfloat16):
                logits, loss = model(x, targets=y, pad_token_id=tokenizer.pad_token_id)
                loss = loss / CONFIG['gradient_accumulation']
            
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                accumulated_steps = 0
                continue
            
            loss.backward()
            accumulated_steps += 1
            
            is_last_batch = (batch_idx + 1 == len(train_loader))
            should_step = (
                accumulated_steps % CONFIG['gradient_accumulation'] == 0
                or is_last_batch  # ✅ FIX v5: flush derniers batches fin d'epoch
            )
            
            if should_step and accumulated_steps > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CONFIG['max_grad_norm']
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                accumulated_steps = 0
                
                global_step += 1
                
                if global_step % CONFIG['validate_every_steps'] == 0 and val_loader is not None:
                    val_ppl, val_loss = validate(
                        model, val_loader, device, tokenizer.pad_token_id, CONFIG['val_batches']
                    )
                    print(f"\n      {'─' * 65}")
                    print(f"      📊 Step {global_step:,} | PPL {val_ppl:7.2f} | "
                          f"Val Loss {val_loss:.4f} | LR {scheduler.get_last_lr()[0]:.2e}")
                    print(f"      {'─' * 65}\n")
                    
                    training_history['validations'].append({
                        'step':       global_step,
                        'epoch':      epoch + 1,
                        'perplexity': val_ppl,
                        'val_loss':   val_loss,
                        'train_loss': loss.item() * CONFIG['gradient_accumulation'],
                        'lr':         scheduler.get_last_lr()[0],
                    })
            
            epoch_loss += loss.item() * CONFIG['gradient_accumulation']
            valid_batches += 1
            
            if batch_idx % 20 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item() * CONFIG["gradient_accumulation"]:.4f}',
                    'lr':   f'{scheduler.get_last_lr()[0]:.2e}',
                    'step': f'{global_step:,}',
                })
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n      ❌ OOM batch {batch_idx}")
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                accumulated_steps = 0  # ✅ FIX v5: reset sinon step prématuré avec 0 gradients
                gc.collect()
                continue
            raise
    
    pbar.close()
    
    avg_loss = epoch_loss / max(valid_batches, 1)
    
    val_ppl, val_loss = (None, None)
    if val_loader is not None:
        val_ppl, val_loss = validate(model, val_loader, device, tokenizer.pad_token_id, CONFIG['val_batches'])
    
    epoch_time = time.time() - t_start
    
    print(f"\n   {'─' * 70}")
    print(f"   ✅ EPOCH {epoch + 1} TERMINÉE")
    print(f"      Train Loss: {avg_loss:.4f}")
    if val_ppl is not None:
        print(f"      Val PPL:    {val_ppl:.2f}")
        print(f"      Val Loss:   {val_loss:.4f}")
    print(f"      Time:       {epoch_time / 60:.1f} min")
    print(f"      LR:         {scheduler.get_last_lr()[0]:.2e}")
    print(f"   {'─' * 70}")
    
    training_history['epochs'].append({
        'epoch':      epoch + 1,
        'train_loss': avg_loss,
        'val_loss':   val_loss,
        'val_ppl':    val_ppl,
        'global_step': global_step,
        'lr':         scheduler.get_last_lr()[0],
        'time_s':     epoch_time,
    })
    
    if (epoch + 1) % CONFIG['save_every_epochs'] == 0:
        checkpoint_manager.save(
            model, optimizer, scheduler,
            metadata={
                'epoch':            epoch + 1,
                'global_step':      global_step,
                'training_history': training_history,
            }
        )
    
    return global_step


# ============================================
# MAIN
# ============================================
def main():
    from HessGpt import HessGPT
    
    print("\n" + "=" * 80)
    print("🤖 LOADING PRETRAINED MODEL v5")
    print("=" * 80)
    
    # Load tokenizer
    global tokenizer
    print(f"\n📝 Loading LLaMA-3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

    # Ajoute uniquement les tokens custom (les tokens de chat sont déjà natifs)
    tokenizer.add_special_tokens({
        'additional_special_tokens': CUSTOM_TOKEN_STRINGS
    })
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Résolution dynamique de TOUS les IDs (natifs + custom)
    # Tokens natifs LLaMA-3
    SPECIAL_TOKENS['<|begin_of_text|>']   = tokenizer.convert_tokens_to_ids('<|begin_of_text|>')
    SPECIAL_TOKENS['<|end_of_text|>']     = tokenizer.convert_tokens_to_ids('<|end_of_text|>')
    SPECIAL_TOKENS['<|start_header_id|>'] = tokenizer.convert_tokens_to_ids('<|start_header_id|>')
    SPECIAL_TOKENS['<|end_header_id|>']   = tokenizer.convert_tokens_to_ids('<|end_header_id|>')
    SPECIAL_TOKENS['<|eot_id|>']          = tokenizer.convert_tokens_to_ids('<|eot_id|>')
    # Tokens custom
    for tok in CUSTOM_TOKEN_STRINGS:
        SPECIAL_TOKENS[tok] = tokenizer.convert_tokens_to_ids(tok)

    # vocab_size réel après add_special_tokens
    CONFIG['vocab_size'] = len(tokenizer)

    print(f"   ✅ LLaMA-3 tokenizer: {len(tokenizer)} tokens")
    print(f"   pad_token_id: {tokenizer.pad_token_id}")
    print(f"   Tokens natifs:")
    for tok in ('<|begin_of_text|>', '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>'):
        print(f"      {tok:30s} → {SPECIAL_TOKENS[tok]}")
    print(f"   Tokens custom:")
    for tok in CUSTOM_TOKEN_STRINGS:
        print(f"      {tok:30s} → {SPECIAL_TOKENS[tok]}")
    
    # Load datasets
    if args.dry_run:
        print("\n📋 DRY RUN - Checking datasets...")
        train_dataset, val_dataset = load_sft_datasets(
            tokenizer, CONFIG['max_seq_len'], num_samples_limit=1000
        )
        print(f"\n✅ Datasets OK!")
        return
    
    train_dataset, val_dataset = load_sft_datasets(
        tokenizer, CONFIG['max_seq_len'], num_samples_limit=args.num_samples
    )
    
    # Create model
    print(f"\n🏗️  Creating HessGPT with YaRN extension...")
    model = HessGPT(
        vocab_size=CONFIG['vocab_size'],
        embed_dim=CONFIG['embed_dim'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers'],
        max_seq_len=CONFIG['max_seq_len'],
        dropout=CONFIG['dropout'],
        use_rope=CONFIG['use_rope'],
        use_yarn=CONFIG['use_yarn'],
        yarn_scale=CONFIG['yarn_scale'],
        yarn_original_max_len=CONFIG['yarn_original_max_len'],
        use_swiglu=CONFIG['use_swiglu'],
        n_kv_heads=CONFIG['n_kv_heads'],
        use_qk_norm=CONFIG['use_qk_norm'],
        soft_cap=CONFIG['soft_cap'],
        use_flash_attn=CONFIG['use_flash_attn']
    ).to(device)
    
    # Load pretrain checkpoint
    if os.path.exists(args.pretrain_checkpoint):
        print(f"\n📂 Loading pretrain checkpoint: {args.pretrain_checkpoint}")
        checkpoint = torch.load(args.pretrain_checkpoint, map_location='cpu', weights_only=True)
        
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=True)
        print(f"   ✅ Pretrain weights loaded")
    else:
        print(f"\n⚠️  Pretrain checkpoint not found: {args.pretrain_checkpoint}")
        print(f"   Starting from scratch...")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Params: {total_params / 1e6:.1f}M")
    
    # Apply LoRA
    model, trainable_params = apply_lora_to_model(
        model,
        r=CONFIG['lora_r'],
        alpha=CONFIG['lora_alpha'],
        dropout=CONFIG['lora_dropout'],
        target_modules=CONFIG['lora_target_modules']
    )
    
    # DataLoaders with DYNAMIC PADDING
    print(f"\n📥 Creating DataLoaders with dynamic padding...")
    collate = make_collate_fn(tokenizer.pad_token_id)  # ✅ FIX v5: pad_token_id réel LLaMA-3
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )
    
    print(f"   ✅ Train: {len(train_loader):,} batches (dynamic padding)")
    print(f"   ✅ Val:   {len(val_loader):,} batches (dynamic padding)")
    
    # Optimizer
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params_list,
        lr=CONFIG['learning_rate'],
        betas=(CONFIG['adam_beta1'], CONFIG['adam_beta2']),
        eps=CONFIG['adam_eps'],
        weight_decay=CONFIG['weight_decay'],
        fused=(device == 'cuda'),  # fused optimizer only on CUDA
    )
    
    # Scheduler
    total_steps = len(train_loader) * CONFIG['num_epochs'] // CONFIG['gradient_accumulation']
    scheduler = WSDScheduler(
        optimizer,
        max_lr=CONFIG['learning_rate'],
        total_steps=total_steps,
        warmup_ratio=CONFIG['warmup_ratio'],
    )
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(args.output_checkpoint)
    
    training_history = {
        'config':      CONFIG,
        'special_tokens': SPECIAL_TOKENS,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_steps': total_steps,
        'epochs':      [],
        'validations': [],
        'start_time':  datetime.now().isoformat(),
    }
    
    global_step = 0
    
    print("\n" + "=" * 80)
    print("🚀 TRAINING START")
    print("=" * 80)
    
    for epoch in range(CONFIG['num_epochs']):
        try:
            global_step = train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                val_loader=val_loader,
                checkpoint_manager=checkpoint_manager,
                training_history=training_history,
                epoch=epoch,
                global_step=global_step,
            )
        except KeyboardInterrupt:
            print("\n\n⚠️  CTRL+C")
            checkpoint_manager.save(
                model, optimizer, scheduler,
                metadata={
                    'epoch':            epoch,
                    'global_step':      global_step,
                    'training_history': training_history,
                }
            )
            return
        except Exception as e:
            print(f"\n❌ ERREUR:")
            print(traceback.format_exc())
            checkpoint_manager.save(
                model, optimizer, scheduler,
                metadata={
                    'epoch':            epoch,
                    'global_step':      global_step,
                    'training_history': training_history,
                }
            )
            raise
    
    # Save final
    checkpoint_manager.save(
        model, optimizer, scheduler,
        metadata={
            'epoch':            CONFIG['num_epochs'],
            'global_step':      global_step,
            'training_history': training_history,
        }
    )
    
    print("\n" + "=" * 80)
    print("🎉 SFT TRAINING TERMINÉ !")
    print("=" * 80)
    print(f"\n📊 RÉSULTATS:")
    print(f"   Epochs:  {len(training_history['epochs'])}/{CONFIG['num_epochs']}")
    print(f"   Steps:   {global_step:,}")
    
    if training_history['validations']:
        last = training_history['validations'][-1]
        print(f"   PPL:     {last['perplexity']:.2f}")
        print(f"   Loss:    {last['val_loss']:.4f}")
    
    print(f"\n💾 Checkpoint: {checkpoint_manager.path}")
    
    history_path = args.output_checkpoint.replace('.pt', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2, default=str)
    print(f"📝 History: {history_path}")
    print("\n✅ DONE !")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrompu")
    except Exception as e:
        print(f"\n❌ ERREUR:")
        print(traceback.format_exc())
    finally:
        print("\n👋")