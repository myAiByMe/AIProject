#!/usr/bin/env python3
"""
HessGPT Pre-Training — LLaMA-3 Tokenizer

USAGE:
    python pretrain_hessgpt_fixed.py
    python pretrain_hessgpt_fixed.py --total-chunks 10
    python pretrain_hessgpt_fixed.py --dry-run
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys
import os
import time
import math
import json
import gc
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from datetime import datetime
import traceback

sys.path.append('./Core/Model')

# ============================================
# TOKENS SPÉCIAUX
# Les tokens de chat LLaMA-3 (<|start_header_id|>, <|end_header_id|>, <|eot_id|>, etc.)
# sont NATIFS dans le vocab LLaMA-3 base — pas besoin de les ajouter.
# Le pretrain tokenize du texte brut, donc seuls les 3 tokens domain sont utiles.
# ============================================
SPECIAL_TOKENS = [
    '<code>',
    '<think>',
    '</think>',
]

# ============================================
# ARGS
# ============================================
parser = argparse.ArgumentParser(description='HessGPT Pre-Training')
parser.add_argument('--total-chunks', type=int, default=None,
                    help='Override nombre de chunks')
parser.add_argument('--dry-run', action='store_true',
                    help='Vérifie sans training')
parser.add_argument('--data-dir', type=str, default='./data/ultra_filtered',
                    help='Directory des chunks')
parser.add_argument('--checkpoint', type=str, default='./checkpoints/HessGpt_pretrain.pt',
                    help='Path du checkpoint')
args = parser.parse_args()

print("=" * 80)
print("🔥 HessGPT v5 — LLaMA-3 Tokenizer + RMSNorm + Flash + 22 Layers")
print("   LLaMA-3 128k | RMSNorm | Flash Attn | QK-Norm | WSD | Soft-cap")
print("=" * 80)

# ============================================
# CONFIGURATION OPTIMISÉE v5
# ============================================
CONFIG = {
    # Model — vocab_size défini après chargement tokenizer (voir plus bas)
    'vocab_size':    None,
    'embed_dim':     1280,
    'num_heads':     20,
    'num_layers':    22,
    'max_seq_len':   1024,
    'dropout':       0.0,
    'use_rope':      True,
    'use_yarn':      False,
    'yarn_scale':    4.0,
    'yarn_original_max_len': 1024,
    'use_swiglu':    True,
    'n_kv_heads':    5,
    'use_qk_norm':   True,
    'soft_cap':      30.0,
    'use_flash_attn': True,

    # Training
    'batch_size':              32,
    'gradient_accumulation':   4,
    'max_grad_norm':           1.0,
    'learning_rate':           4e-4,
    'weight_decay':            0.1,
    'adam_beta1':              0.9,
    'adam_beta2':              0.95,
    'adam_eps':                1e-8,

    # Data
    'data_dir':     args.data_dir,
    'val_chunk_id': 0,

    # WSD LR Schedule
    'warmup_ratio':  0.03,
    'decay_ratio':   0.15,
    'min_lr_ratio':  0.1,

    # Validation
    'validate_every_steps': 500,
    'val_batches':          50,

    # Checkpoint
    'checkpoint_file':   args.checkpoint,
    'save_every_epochs': 5,

    # System
    'use_compile':  True,
    'compile_mode': 'default',
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\n📊 CONFIGURATION (LLaMA-3 Tokenizer):")
print(f"   Vocab size : sera défini après chargement tokenizer")
print(f"   Embed dim  : {CONFIG['embed_dim']}")
print(f"   Layers     : {CONFIG['num_layers']}")
print(f"   Q heads    : {CONFIG['num_heads']}")
print(f"   KV heads   : {CONFIG['n_kv_heads']} (GQA ratio {CONFIG['num_heads']//CONFIG['n_kv_heads']}:1)")
print(f"   Dropout    : {CONFIG['dropout']}")
print(f"   Weight decay: {CONFIG['weight_decay']}")
print(f"   Use RoPE   : ✅")
print(f"   Use SwiGLU : ✅")
print(f"   GQA        : ✅")
print(f"   RMSNorm    : ✅")
print(f"   QK-Norm    : ✅")
print(f"   Soft-cap   : {CONFIG['soft_cap']} ✅")
print(f"   Flash Attn : {CONFIG['use_flash_attn']}")

print(f"\n🔧 TOKENS CUSTOM ({len(SPECIAL_TOKENS)} — ajoutés après vocab natif LLaMA-3):")
for tok in SPECIAL_TOKENS:
    print(f"   {tok}")

# ============================================
# SCAN CHUNKS
# ============================================
def scan_available_chunks(data_dir):
    """Scanne chunks avec stats.json"""
    available = []
    if not os.path.exists(data_dir):
        return available

    for entry in sorted(os.listdir(data_dir)):
        if not entry.startswith('chunk'):
            continue
        
        chunk_dir = os.path.join(data_dir, entry)
        if not os.path.isdir(chunk_dir):
            continue
        
        stats_file = os.path.join(chunk_dir, 'stats.json')
        if not os.path.exists(stats_file):
            continue
        
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            pt_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.npy')])
            
            if len(pt_files) > 0:
                if entry.startswith('chunk_'):
                    chunk_id = int(entry.split('_')[1])
                else:
                    chunk_id = int(entry.replace('chunk', ''))
                available.append({
                    'id': chunk_id,
                    'dir': chunk_dir,
                    'files': pt_files,
                    'stats': stats,
                })
        except Exception as e:
            print(f"   ⚠️  Skip {entry}: {e}")
            continue

    available.sort(key=lambda x: x['id'])
    return available

print(f"\n🔍 Scan chunks...")
ALL_CHUNKS = scan_available_chunks(CONFIG['data_dir'])

if args.total_chunks is not None:
    ALL_CHUNKS = ALL_CHUNKS[:args.total_chunks]

NUM_CHUNKS_TOTAL = len(ALL_CHUNKS)
print(f"   ✅ {NUM_CHUNKS_TOTAL} chunks trouvés")

TRAIN_CHUNKS = ALL_CHUNKS

NUM_TRAIN_CHUNKS = len(TRAIN_CHUNKS)

if NUM_TRAIN_CHUNKS == 0:
    print("\n❌ Aucun chunk de training !")
    sys.exit(1)

print(f"\n📊 RÉSUMÉ :")
print(f"   • Training chunks: {NUM_TRAIN_CHUNKS}")
print(f"   • Each chunk: ~985M train + ~15M val (auto-split)")
print(f"   • Total training tokens: ~{NUM_TRAIN_CHUNKS * 985}M ({NUM_TRAIN_CHUNKS * 0.985:.2f}B)")
print(f"   • Validation: 15M tokens per chunk (from same chunk)")

if args.dry_run:
    print("\n📋 DRY RUN :")
    total_tokens = 0
    for chunk in TRAIN_CHUNKS:
        tokens = chunk['stats']['total_tokens']
        total_tokens += tokens
        print(f"   chunk_{chunk['id']:03d}: {tokens/1e9:.2f}B tokens")
    print(f"\n   Total: {total_tokens/1e9:.2f}B tokens")
    sys.exit(0)

# ============================================
# CALCUL STEPS
# ============================================
def calculate_steps_per_chunk(chunk_stats):
    """Calcule steps basé sur taille réelle"""
    total_tokens = chunk_stats['total_tokens']
    samples = total_tokens // (CONFIG['max_seq_len'] + 1)
    batches = math.ceil(samples / CONFIG['batch_size'])
    steps = math.ceil(batches / CONFIG['gradient_accumulation'])
    return max(steps, 1)

STEPS_PER_CHUNK = []
for chunk in TRAIN_CHUNKS:
    steps = calculate_steps_per_chunk(chunk['stats'])
    STEPS_PER_CHUNK.append(steps)
    print(f"   Chunk {chunk['id']:03d}: {steps:,} steps")

TOTAL_STEPS = sum(STEPS_PER_CHUNK)

print(f"\n📈 TRAINING PLAN :")
print(f"   Chunks     : {NUM_TRAIN_CHUNKS}")
print(f"   Total steps: {TOTAL_STEPS:,}")
print(f"   Tokens     : {sum(c['stats']['total_tokens'] for c in TRAIN_CHUNKS)/1e9:.2f}B")

# ============================================
# TOKENIZER LLAMA-3
# ============================================
print(f"\n✅ Device : {device}")
if device == 'cuda':
    print(f"   GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    if torch.cuda.is_bf16_supported():
        print(f"   BF16: ✅")

print(f"\n📝 Loading LLaMA-3 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

tokenizer.add_special_tokens({
    'additional_special_tokens': SPECIAL_TOKENS
})

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Vocab size défini dynamiquement après ajout des special tokens
CONFIG['vocab_size'] = len(tokenizer)

print(f"   ✅ LLaMA-3 tokenizer: {len(tokenizer)} tokens")
print(f"   ✅ Special tokens assignés automatiquement:")
for tok in SPECIAL_TOKENS:
    tid = tokenizer.convert_tokens_to_ids(tok)
    print(f"      {tok:20s} → {tid}")

# ============================================
# WSD SCHEDULER
# ============================================
class WSDScheduler:
    """Warmup – Stable – Decay"""
    def __init__(self, optimizer, max_lr, total_steps,
                 warmup_ratio=0.02, decay_ratio=0.08, min_lr_ratio=0.1):
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
# DATASET WITH TRAIN/VAL SPLIT
# ============================================
class ChunkSubset(Dataset):
    """
    Subset d'un chunk (train ou val)
    
    Utilisé après split du chunk principal
    """
    def __init__(self, tokens, seq_len, pad_token_id, mode='train'):
        self.tokens = tokens
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.mode = mode
        self.num_samples = len(tokens) // (seq_len + 1)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        
        if len(chunk) < self.seq_len + 1:
            pad_len = self.seq_len + 1 - len(chunk)
            chunk = torch.cat([
                chunk,
                torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
            ])
        
        return chunk[:-1], chunk[1:]


class LazyChunkDataset:
    """
    Charge un chunk en RAM et le split en train/val
    
    Args:
        chunk_info: Info du chunk (id, dir, files)
        seq_len: Longueur de séquence
        pad_token_id: Token de padding
        val_tokens: Nombre de tokens pour validation (défaut: 15M)
    
    Usage:
        chunk_dataset = LazyChunkDataset(chunk_info, seq_len, pad_token_id)
        train_dataset = chunk_dataset.get_train_dataset()
        val_dataset = chunk_dataset.get_val_dataset()
    """
    def __init__(self, chunk_info, seq_len, pad_token_id, val_tokens=15_000_000):
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.val_tokens = val_tokens
        
        self.tokens_train = None
        self.tokens_val = None
        self.num_samples_train = 0
        self.num_samples_val = 0
        
        self._load(chunk_info)
    
    def _load(self, chunk_info):
        print(f"   📥 Loading chunk_{chunk_info['id']:03d}...")
        t0 = time.time()
        
        import numpy as np
        all_tensors = []
        for fname in chunk_info['files']:
            fpath = os.path.join(chunk_info['dir'], fname)
            try:
                data_np = np.load(fpath)
                data = torch.from_numpy(data_np).long()
                all_tensors.append(data)
            except Exception as e:
                print(f"      ⚠️  {fname}: {e}")
                continue
        
        if len(all_tensors) == 0:
            raise ValueError(f"Chunk {chunk_info['id']} vide !")
        
        all_tokens   = torch.cat(all_tensors)
        total_tokens = len(all_tokens)

        val_size   = min(self.val_tokens, int(total_tokens * 0.05))
        train_size = total_tokens - val_size

        self.tokens_train = all_tokens[:train_size]
        self.tokens_val   = all_tokens[train_size:]
        
        self.num_samples_train = len(self.tokens_train) // (self.seq_len + 1)
        self.num_samples_val = len(self.tokens_val) // (self.seq_len + 1)
        
        elapsed = time.time() - t0
        
        print(f"   ✅ {total_tokens/1e6:.1f}M tokens loaded ({elapsed:.1f}s)")
        print(f"      📊 Train: {len(self.tokens_train)/1e6:.1f}M tokens ({self.num_samples_train:,} samples)")
        print(f"      📊 Val:   {len(self.tokens_val)/1e6:.1f}M tokens ({self.num_samples_val:,} samples)")
    
    def get_train_dataset(self):
        """Retourne le dataset de training"""
        return ChunkSubset(self.tokens_train, self.seq_len, self.pad_token_id, mode='train')
    
    def get_val_dataset(self):
        """Retourne le dataset de validation"""
        return ChunkSubset(self.tokens_val, self.seq_len, self.pad_token_id, mode='val')
    
    def unload(self):
        """Libère la RAM"""
        if self.tokens_train is not None:
            del self.tokens_train
            self.tokens_train = None
        if self.tokens_val is not None:
            del self.tokens_val
            self.tokens_val = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ============================================
# CHECKPOINT
# ============================================
class CheckpointManager:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def save(self, model, optimizer, scheduler, metadata):
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        checkpoint = {
            'model_state_dict':     m.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step':          metadata['global_step'],
            'next_train_chunk_idx': metadata['next_train_chunk_idx'],
            'training_history':     metadata['training_history'],
            'total_training_time':  metadata.get('total_training_time', 0),
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
        cp = torch.load(self.path, map_location='cpu', weights_only=False)
        print(f"   ✅ Step: {cp['global_step']:,}")
        print(f"   ✅ Next chunk: {cp['next_train_chunk_idx']}")
        return cp

# ============================================
# VALIDATION
# ============================================
@torch.no_grad()
def validate(model, val_loader, device, pad_token_id, max_batches=50):
    """Validation avec calcul correct de la loss"""
    model.eval()
    total_loss   = 0.0
    num_batches  = 0
    autocast_enabled = (device == 'cuda')
    autocast_dtype   = torch.bfloat16 if autocast_enabled else torch.float32

    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        
        x = x.to(device)
        y = y.to(device)
        
        with torch.amp.autocast(device, dtype=autocast_dtype, enabled=autocast_enabled):
            _, loss = model(x, targets=y, pad_token_id=pad_token_id)
        
        total_loss  += loss.item()
        num_batches += 1

    avg_loss   = total_loss / max(num_batches, 1)
    perplexity = math.exp(min(avg_loss, 10))
    
    return perplexity, avg_loss

# ============================================
# TRAIN ONE CHUNK
# ============================================
def train_one_chunk(
    model, chunk_info, optimizer, scheduler,
    checkpoint_manager, training_history,
    global_step, total_training_time, train_chunk_idx
):
    epoch_num = train_chunk_idx + 1

    print(f"\n{'=' * 80}")
    print(f"📦 EPOCH {epoch_num}/{NUM_TRAIN_CHUNKS} — chunk_{chunk_info['id']:03d}")
    print(f"   LR: {scheduler.get_last_lr()[0]:.2e}")
    print(f"{'=' * 80}")

    try:
        chunk_dataset = LazyChunkDataset(
            chunk_info,
            CONFIG['max_seq_len'],
            tokenizer.pad_token_id,
            val_tokens=15_000_000
        )
        train_dataset = chunk_dataset.get_train_dataset()
        val_dataset   = chunk_dataset.get_val_dataset()

    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return global_step, total_training_time

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    num_batches = len(train_loader)
    print(f"   📊 Train: {num_batches:,} batches ({len(train_dataset):,} samples)")
    print(f"   📊 Val:   {len(val_loader):,} batches ({len(val_dataset):,} samples)")

    model.train()
    epoch_loss = 0.0
    valid_batches = 0
    t_start = time.time()
    running_loss = 0.0
    running_batches = 0

    autocast_enabled = (device == 'cuda')
    autocast_dtype   = torch.bfloat16 if autocast_enabled else torch.float32

    pbar = tqdm(train_loader, desc=f"Epoch {epoch_num}/{NUM_TRAIN_CHUNKS}", leave=True)

    for batch_idx, (x, y) in enumerate(pbar):
        try:
            x = x.to(device)
            y = y.to(device)

            with torch.amp.autocast(device, dtype=autocast_dtype, enabled=autocast_enabled):
                logits, loss = model(x, targets=y, pad_token_id=tokenizer.pad_token_id)
                loss = loss / CONFIG['gradient_accumulation']

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            is_last_batch = (batch_idx + 1 == num_batches)
            should_step   = ((batch_idx + 1) % CONFIG['gradient_accumulation'] == 0) or is_last_batch

            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CONFIG['max_grad_norm']
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % CONFIG['validate_every_steps'] == 0:
                    val_ppl, val_loss = validate(
                        model, val_loader, device, tokenizer.pad_token_id, CONFIG['val_batches']
                    )
                    model.train()
                    avg_so_far = running_loss / max(running_batches, 1)
                    train_ppl  = math.exp(min(avg_so_far, 10))
                    print(f"\n      {'─' * 70}")
                    print(f"      📊 Step {global_step:,} | "
                          f"Train Loss {avg_so_far:.4f} | Train PPL {train_ppl:.2f} | "
                          f"Val Loss {val_loss:.4f} | Val PPL {val_ppl:.2f} | "
                          f"LR {scheduler.get_last_lr()[0]:.2e}")
                    print(f"      {'─' * 70}\n")

                    training_history['validations'].append({
                        'step':        global_step,
                        'epoch':       epoch_num,
                        'chunk_id':    chunk_info['id'],
                        'perplexity':  val_ppl,
                        'val_loss':    val_loss,
                        'train_loss':  avg_so_far,
                        'train_ppl':   train_ppl,
                        'lr':          scheduler.get_last_lr()[0],
                    })

            raw_loss = loss.item() * CONFIG['gradient_accumulation']
            epoch_loss      += raw_loss
            running_loss    += raw_loss
            valid_batches   += 1
            running_batches += 1

            if batch_idx % 20 == 0:
                avg_loss_so_far = running_loss / max(running_batches, 1)
                ppl_so_far      = math.exp(min(avg_loss_so_far, 10))
                pbar.set_postfix({
                    'loss': f'{raw_loss:.4f}',
                    'avg':  f'{avg_loss_so_far:.4f}',
                    'ppl':  f'{ppl_so_far:.2f}',
                    'lr':   f'{scheduler.get_last_lr()[0]:.2e}',
                    'step': f'{global_step:,}',
                })

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n      ❌ OOM batch {batch_idx}")
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                model.train()
                continue
            raise

    elapsed = time.time() - t_start
    total_training_time += elapsed

    avg_loss = epoch_loss / max(valid_batches, 1)
    
    print(f"\n{'─' * 80}")
    print(f"   Epoch {epoch_num} terminé:")
    print(f"   Loss: {avg_loss:.4f} | Time: {elapsed/60:.1f}min")
    print(f"{'─' * 80}")

    training_history['epochs'].append({
        'epoch':      epoch_num,
        'chunk_id':   chunk_info['id'],
        'train_loss': avg_loss,
        'time':       elapsed,
        'batches':    valid_batches,
        'global_step': global_step,
    })

    if epoch_num % CONFIG['save_every_epochs'] == 0:
        checkpoint_manager.save(
            model, optimizer, scheduler,
            metadata={
                'global_step':         global_step,
                'next_train_chunk_idx': train_chunk_idx + 1,
                'training_history':    training_history,
                'total_training_time': total_training_time,
            }
        )

    chunk_dataset.unload()
    del chunk_dataset, train_dataset, val_dataset, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return global_step, total_training_time

# ============================================
# OPTIMIZER CONFIG
# ============================================
def configure_optimizers(model, learning_rate, weight_decay, betas, eps):
    """Configure optimizer avec weight decay groups"""
    decay = set()
    no_decay = set()
    
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, (torch.nn.Linear,)):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, (torch.nn.Embedding,)):
                no_decay.add(fpn)
            else:
                no_decay.add(fpn)
    
    param_dict = {pn: p for pn, p in model.named_parameters()}
    
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
    ]
    
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
        fused=(device == 'cuda'),
    )
    
    print(f"\n⚙️  OPTIMIZER GROUPS:")
    print(f"   • With decay:    {len(decay):>5} params")
    print(f"   • Without decay: {len(no_decay):>5} params")
    
    return optimizer

# ============================================
# MAIN
# ============================================
def main():
    from HessGpt import HessGPT

    print("\n" + "=" * 80)
    print("🤖 CRÉATION MODÈLE")
    print("=" * 80)

    checkpoint_manager = CheckpointManager(CONFIG['checkpoint_file'])

    print(f"\n🏗️  HessGPT v5 (LLaMA-3 tokenizer)...")
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
    
    print(f"   ✅ FP32 weights")
    print(f"   ✅ BF16 autocast")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Params: {total_params / 1e6:.1f}M")
    
    if hasattr(model, 'count_parameters'):
        p = model.count_parameters()
        print(f"\n📊 Architecture:")
        print(f"   • Vocab size:   {CONFIG['vocab_size']:,}")
        print(f"   • Token emb:    {p['token_embeddings'] / 1e6:.1f}M")
        print(f"   • Position emb: {p['position_embeddings'] / 1e6:.1f}M (RoPE=0)")
        print(f"   • Blocks:       {p['transformer_blocks'] / 1e6:.1f}M")

    if CONFIG['use_compile'] and device == 'cuda':
        print(f"\n⚡ torch.compile...")
        try:
            model = torch.compile(model, mode=CONFIG['compile_mode'])
            print(f"   ✅ Compilé")
        except Exception as e:
            print(f"   ⚠️  Échec: {e}")

    optimizer = configure_optimizers(
        model,
        CONFIG['learning_rate'],
        CONFIG['weight_decay'],
        (CONFIG['adam_beta1'], CONFIG['adam_beta2']),
        CONFIG['adam_eps']
    )

    scheduler = WSDScheduler(
        optimizer,
        max_lr=CONFIG['learning_rate'],
        total_steps=TOTAL_STEPS,
        warmup_ratio=CONFIG['warmup_ratio'],
        decay_ratio=CONFIG['decay_ratio'],
        min_lr_ratio=CONFIG['min_lr_ratio'],
    )

    training_history = {
        'config':          CONFIG,
        'special_tokens':  SPECIAL_TOKENS,
        'total_params':    total_params,
        'num_train_chunks': NUM_TRAIN_CHUNKS,
        'total_steps':     TOTAL_STEPS,
        'epochs':          [],
        'validations':     [],
        'start_time':      datetime.now().isoformat(),
    }

    global_step = 0
    start_train_chunk_idx = 0
    total_training_time = 0

    checkpoint = checkpoint_manager.load()
    if checkpoint:
        print("\n♻️  REPRISE")
        unwrapped = model._orig_mod if hasattr(model, '_orig_mod') else model
        unwrapped.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        current_lr = scheduler.get_lr()
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr

        global_step = checkpoint['global_step']
        start_train_chunk_idx = checkpoint['next_train_chunk_idx']
        training_history = checkpoint['training_history']
        total_training_time = checkpoint.get('total_training_time', 0)
        print(f"   ▶️  Chunk {start_train_chunk_idx}, step {global_step:,}")

    print("\n" + "=" * 80)
    print("🚀 TRAINING START")
    print(f"   Chunks: {start_train_chunk_idx} → {NUM_TRAIN_CHUNKS}")
    print(f"   Each chunk: 985M train + 15M val (1.5% for validation)")
    print("=" * 80)

    overall_start = time.time()

    for train_chunk_idx in range(start_train_chunk_idx, NUM_TRAIN_CHUNKS):
        chunk_info = TRAIN_CHUNKS[train_chunk_idx]

        try:
            global_step, total_training_time = train_one_chunk(
                model=model,
                chunk_info=chunk_info,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_manager=checkpoint_manager,
                training_history=training_history,
                global_step=global_step,
                total_training_time=total_training_time,
                train_chunk_idx=train_chunk_idx,
            )
        except KeyboardInterrupt:
            print("\n\n⚠️  CTRL+C")
            checkpoint_manager.save(
                model, optimizer, scheduler,
                metadata={
                    'global_step':         global_step,
                    'next_train_chunk_idx': train_chunk_idx,
                    'training_history':    training_history,
                    'total_training_time': total_training_time,
                }
            )
            return
        except Exception as e:
            print(f"\n❌ ERREUR:")
            print(traceback.format_exc())
            checkpoint_manager.save(
                model, optimizer, scheduler,
                metadata={
                    'global_step':         global_step,
                    'next_train_chunk_idx': train_chunk_idx,
                    'training_history':    training_history,
                    'total_training_time': total_training_time,
                }
            )
            raise

    overall_time = time.time() - overall_start

    checkpoint_manager.save(
        model, optimizer, scheduler,
        metadata={
            'global_step':         global_step,
            'next_train_chunk_idx': NUM_TRAIN_CHUNKS,
            'training_history':    training_history,
            'total_training_time': total_training_time,
        }
    )

    print("\n" + "=" * 80)
    print("🎉 TRAINING TERMINÉ !")
    print("=" * 80)
    print(f"\n📊 RÉSULTATS:")
    print(f"   Epochs:  {len(training_history['epochs'])}/{NUM_TRAIN_CHUNKS}")
    print(f"   Steps:   {global_step:,}")
    print(f"   Train:   {total_training_time / 3600:.2f}h")
    print(f"   Real:    {overall_time / 3600:.2f}h")

    if training_history['validations']:
        last = training_history['validations'][-1]
        print(f"   PPL:     {last['perplexity']:.2f}")
        print(f"   Loss:    {last['val_loss']:.4f}")

    print(f"\n💾 Checkpoint: {checkpoint_manager.path}")

    history_path = CONFIG['checkpoint_file'].replace('.pt', '_history.json')
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