"""
Medical BERT Preprocessing and Model Implementation
Based on specifications:
- Vocabulary size: (ICD diagnosis, ICD procedure, medication codes)
- Time embeddings: 7-day non-overlapping windows
- Medical code type embeddings: diseases, symptoms, procedures, special tokens
- BERT architecture: 5 layers, hidden size 320, 5 attention heads, 15M parameters
- Pre-training: MLM with 15% masking probability
- Training: AdamW, batch size 512, 100 epochs, LR 5e-4 with reduce on plateau
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import paths


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, default_collate
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json
from datetime import datetime, timedelta
import time
import math
from input_data_format import load_data_from_csv, calculate_vocab_size
from tqdm import tqdm

class MedicalCodeTokenizer:
    """Tokenizes medical codes and builds vocabulary"""
    
    def __init__(self, vocab_size: Optional[int] = None):
        self.vocab_size = vocab_size
        self.code_to_idx: Dict[str, int] = {}
        self.idx_to_code: Dict[int, str] = {}
        self.special_tokens = {
            '[PAD]': 0,
            '[MASK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[UNK]': 4
        }
        self._initialize_special_tokens()
    
    def _initialize_special_tokens(self):
        """Initialize special tokens in vocabulary"""
        for token, idx in self.special_tokens.items():
            self.code_to_idx[token] = idx
            self.idx_to_code[idx] = token
    
    def build_vocab(self, codes: List[List[str]]):
        """Build vocabulary from list of code sequences"""
        code_counts = defaultdict(int)
        for seq in codes:
            for code in seq:
                code_counts[code] += 1
        print(f"DEBUG: Found {len(code_counts)} unique codes in input sequences.") 
        
        # Sort by frequency and take top vocab_size codes
        sorted_codes = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add codes to vocabulary (starting after special tokens)
        start_idx = len(self.special_tokens)
        for code, count in sorted_codes:
            if code not in self.code_to_idx:
                idx = len(self.code_to_idx)
                self.code_to_idx[code] = idx
                self.idx_to_code[idx] = code
        
        # --- NEW: Update vocab_size to actual number of filled tokens ---
        # This prevents out-of-bounds indexing during random masking
        self.vocab_size = len(self.code_to_idx)
    
    def encode(self, codes: List[str]) -> List[int]:
        """Encode list of codes to indices"""
        return [self.code_to_idx.get(code, self.special_tokens['[UNK]']) for code in codes]
    
    def decode(self, indices: List[int]) -> List[str]:
        """Decode indices to codes"""
        return [self.idx_to_code.get(idx, '[UNK]') for idx in indices]
    
    def save(self, path: str):
        """Save tokenizer to file"""
        with open(path, 'w') as f:
            json.dump({
                'code_to_idx': self.code_to_idx,
                'idx_to_code': {str(k): v for k, v in self.idx_to_code.items()},
                'vocab_size': self.vocab_size
            }, f, indent=2)
    
    def load(self, path: str):
        """Load tokenizer from file"""
        with open(path, 'r') as f:
            data = json.load(f)
            self.code_to_idx = data['code_to_idx']
            self.idx_to_code = {int(k): v for k, v in data['idx_to_code'].items()}
            self.vocab_size = data['vocab_size']


class TimeEmbedding(nn.Module):
    """Time embedding for 7-day non-overlapping windows"""
    
    def __init__(self, embed_dim: int, window_days: int = 7):
        super().__init__()
        self.window_days = window_days
        self.embed_dim = embed_dim
        # We'll use a learnable embedding table for time windows
        # Maximum 1000 windows (about 19 years)
        self.max_windows = 1000
        self.time_embedding = nn.Embedding(self.max_windows, embed_dim)
    
    def get_window_idx(self, timestamps: List[datetime], start_time: Optional[datetime] = None) -> torch.Tensor:
        """Convert timestamps to window indices relative to start_time"""
        if len(timestamps) == 0:
            return torch.tensor([], dtype=torch.long)
        
        # Use provided start_time or fallback to the earliest in the current list
        reference_time = start_time if start_time is not None else min(timestamps)
        
        # Calculate window indices
        window_indices = []
        for ts in timestamps:
            delta = ts - reference_time
            # Ensure index is non-negative even if ts < reference_time (shouldn't happen with correct start_time)
            window_idx = max(0, delta.days // self.window_days)
            window_idx = min(window_idx, self.max_windows - 1)  # Clamp to max
            window_indices.append(window_idx)
        
        return torch.tensor(window_indices, dtype=torch.long)
    
    def forward(self, timestamps: List[datetime], start_time: Optional[datetime] = None) -> torch.Tensor:
        """Get time embeddings for timestamps"""
        window_indices = self.get_window_idx(timestamps, start_time)
        if window_indices.numel() == 0:
            raise ValueError("Empty timestamps provided to TimeEmbedding. At least one timestamp is required.")
        
        # 确保索引张量和 Embedding 层在同一个设备上
        window_indices = window_indices.to(self.time_embedding.weight.device)
        return self.time_embedding(window_indices)


class MedicalCodeTypeEmbedding(nn.Module):
    """Embedding for medical code types (diseases, symptoms, procedures, special tokens)"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Define code type categories
        self.code_types = {
            'diagnosis': 0,      # ICD diagnosis codes
            'procedure': 1,      # ICD procedure codes
            'medication': 2,     # Medication prescriptions
            'special': 3,       # Special tokens
            'unknown': 4
        }
        self.num_types = len(self.code_types)
        self.type_embedding = nn.Embedding(self.num_types, embed_dim) # random initializing embedding
    
    def get_code_type(self, code: str, code_type: Optional[str] = None) -> int:
        """Determine code type from code string or use provided code_type
        
        Args:
            code: Code string
            code_type: Optional code type string ('diagnosis', 'procedure', 'medication')
                      If provided, use this directly instead of inferring from code string
        """
        # If code_type is provided, use it directly
        if code_type is not None:
            return self.code_types.get(code_type.lower(), self.code_types['unknown'])
        
        # Fallback: infer from code string (for special tokens or when code_type not available)
        if code in ['[PAD]', '[MASK]', '[CLS]', '[SEP]', '[UNK]']:
            return self.code_types['special']
        elif code.startswith('ICD9') or code.startswith('ICD10') or code.startswith('E') or code.startswith('V'):
            # Check if it's a procedure code (usually starts with specific prefixes)
            if any(code.startswith(prefix) for prefix in ['ICD9PROC', 'ICD10PROC', 'PROC']):
                return self.code_types['procedure']
            else:
                return self.code_types['diagnosis']
        elif code.startswith('MED') or code.startswith('RX') or code.startswith('DRUG'):
            return self.code_types['medication']
        else:
            return self.code_types['unknown']
    
    def forward(self, codes: List[str], code_types: Optional[List[str]] = None) -> torch.Tensor:
        """Get type embeddings for codes
        
        Args:
            codes: List of code strings
            code_types: Optional list of code type strings ('diagnosis', 'procedure', 'medication')
                       If provided, use these directly instead of inferring from code strings
        """
        if code_types is not None:
            # Use provided code types directly
            type_indices = [self.code_types.get(code_type, self.code_types['unknown']) for code_type in code_types]
        else:
            # If code_types is not provided, we should raise an error rather than guessing
            raise ValueError("code_types must be provided to MedicalCodeTypeEmbedding. Fallback inference is disabled for accuracy.")
        
        type_indices = torch.tensor(type_indices, dtype=torch.long) # turn input from a list into a tensor, as embedding layer requires tensor input
        type_indices = type_indices.to(self.type_embedding.weight.device)
        return self.type_embedding(type_indices) # still: give three type indices, output three type embeddings


class TransformerBlock(nn.Module):
    """Single transformer encoder block"""
    
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int = None, dropout: float = 0.1):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = hidden_size * 4
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class MedicalBERT(nn.Module):
    """Custom BERT model for medical codes"""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 320,
        num_layers: int = 5,
        num_heads: int = 5,
        max_seq_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        
        # Embeddings
        self.code_embedding = nn.Embedding(vocab_size, hidden_size)
        self.time_embedding = TimeEmbedding(hidden_size, window_days=7)
        self.type_embedding = MedicalCodeTypeEmbedding(hidden_size)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        
        # Layer norm and dropout
        self.embedding_norm = nn.LayerNorm(hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # MLM head
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        code_indices: torch.Tensor,
        timestamps: List[List[datetime]],
        first_timestamps: Optional[List[datetime]] = None,
        original_codes: Optional[List[List[str]]] = None,
        code_types: Optional[List[List[str]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Forward pass
        Args:
            code_indices: [batch_size, seq_len] tensor of code indices
            timestamps: List of lists of datetime objects for each sequence
            first_timestamps: List of the very first timestamp for each patient (for time embedding)
            original_codes: List of lists of original code strings
            code_types: List of lists of code type strings
            attention_mask: [batch_size, seq_len] mask (True for padding)
            return_embeddings: If True, returns hidden states instead of logits
        """
        batch_size, seq_len = code_indices.shape
        
        # Code embeddings
        code_emb = self.code_embedding(code_indices)
        
        # Time embeddings
        time_emb_list = []
        if first_timestamps is None:
            raise ValueError("first_timestamps must be provided to MedicalBERT for accurate time window calculation.")
            
        for i, ts_list in enumerate(timestamps):
            if len(ts_list) < seq_len:
                # Pad with last timestamp
                ts_list = ts_list + [ts_list[-1] if ts_list else datetime.now()] * (seq_len - len(ts_list))
            elif len(ts_list) > seq_len:
                # Truncate to max length
                ts_list = ts_list[:seq_len]
            
            start_time = first_timestamps[i]
            time_emb = self.time_embedding(ts_list, start_time=start_time)
            time_emb_list.append(time_emb)
        time_emb = torch.stack(time_emb_list, dim=0)
        
        # Type embeddings
        type_emb_list = []
        if code_types is None:
            raise ValueError("code_types must be provided to MedicalBERT. Fallback inference is disabled for accuracy.")
            
        for types in code_types: # for each sequence in the batch
            if len(types) < seq_len:
                types = types + ['special'] * (seq_len - len(types))
            elif len(types) > seq_len:
                types = types[:seq_len]
            
            # Create dummy codes list (not used when code_types is provided)
            dummy_codes = [''] * len(types)
            type_emb = self.type_embedding(dummy_codes, code_types=types)
            type_emb_list.append(type_emb)
        type_emb = torch.stack(type_emb_list, dim=0)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=code_indices.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)  # [batch, seq_len, hidden]
        
        # Combine embeddings
        embeddings = code_emb + time_emb + type_emb + pos_emb
        embeddings = self.embedding_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        # Transformer blocks
        x = embeddings
        for transformer in self.transformer_blocks:
            x = transformer(x, mask=attention_mask)
        
        if return_embeddings:
            return x  # [batch, seq_len, hidden_size]
            
        # MLM head
        logits = self.mlm_head(x)  # [batch, seq_len, vocab_size]
        
        return logits
    
    def _decode_indices_for_type(self, indices: List[int]) -> List[str]:
        """Helper to decode indices for type embedding (fallback)"""
        # This is a fallback - in practice, original_codes should be provided
        return [f"CODE_{idx}" for idx in indices]


class EHRDataset(Dataset):
    """Dataset for EHR sequences"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: MedicalCodeTokenizer,
        max_length: int = 512,
        mask_prob: float = 0.15
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get codes, timestamps, and code_types
        # Fix Point 1: Match keys from load_data_from_csv
        codes = item['codes']
        timestamps = item['timestamps']
        code_types = item.get('code_types', None)

         
        # --- NEW: Sequence Integrity Guard ---
        if any(ts is None or pd.isna(ts) for ts in timestamps):
            raise ValueError(f"❌ Data Integrity Error: Patient index {idx} has missing timestamps (NaT). "
                             f"This will break the medical sequence logic.")
        
        # Ensure it's sorted (Double check)
        for i in range(len(timestamps) - 1):
            if timestamps[i] > timestamps[i+1]:
                raise ValueError(f"❌ Sorting Error: Sequence at index {idx} is NOT sorted. "
                                 f"Check the data loading and sorting logic.")
        
        # Fix Point 2: Store the true first timestamp of this patient
        first_timestamp = timestamps[0] if timestamps else datetime.now()

        if code_types is None:
            raise ValueError(f"Data item is missing code_types. All medical codes must have an associated type.")

        # Sort all three by timestamp
        sorted_pairs = sorted(zip(codes, timestamps, code_types), key=lambda x: x[1])
        codes, timestamps, code_types = zip(*sorted_pairs)
        codes = list(codes)
        timestamps = list(timestamps)
        code_types = list(code_types)
        
        # Randomly select a sequence of up to 512 consecutive codes (paper requirement)
        if len(codes) > self.max_length:
            start_idx = np.random.randint(0, len(codes) - self.max_length + 1)
            codes = codes[start_idx:start_idx + self.max_length]
            timestamps = timestamps[start_idx:start_idx + self.max_length]
            if code_types is not None:
                code_types = code_types[start_idx:start_idx + self.max_length] 
        
        # Encode codes (after random selection)
        code_indices = self.tokenizer.encode(codes)
        
        # Store original codes and code types before padding
        original_codes_list = codes.copy()
        original_code_types_list = code_types.copy() if code_types is not None else None
        
        # Pad to max_length if needed (for sequences shorter than max_length)
        if len(code_indices) < self.max_length:
            padding_length = self.max_length - len(code_indices)
            code_indices = code_indices + [self.tokenizer.special_tokens['[PAD]']] * padding_length
            # Pad timestamps with last timestamp
            if not timestamps:
                raise ValueError("Empty timestamps encountered during padding. Each sequence must have at least one timestamp.")
            last_ts = timestamps[-1] 
            timestamps = timestamps + [last_ts] * padding_length
            # Pad original codes with [PAD]
            original_codes_list = original_codes_list + ['[PAD]'] * padding_length
            # Pad code types with 'special' for [PAD] tokens
            if original_code_types_list is None:
                raise ValueError("original_code_types_list is None during padding. This should not happen if data is correctly loaded.")
            original_code_types_list = original_code_types_list + ['special'] * padding_length
        
        # Create attention mask
        attention_mask = [idx != self.tokenizer.special_tokens['[PAD]'] for idx in code_indices]
        
        # MLM masking
        masked_indices = code_indices.copy()
        labels = [-100] * len(code_indices)  # -100 is ignored in loss
        
        for i in range(len(code_indices)):
            if code_indices[i] == self.tokenizer.special_tokens['[PAD]']:
                continue
            
            if np.random.random() < self.mask_prob: # under the scenario of masking, we have the following scenarios
                labels[i] = code_indices[i]  # Store original for calculating loss: they are the correct answers
                
                # 80% of the time, replace with [MASK]
                # 10% of the time, replace with random token
                # 10% of the time, keep original
                rand = np.random.random()
                if rand < 0.8:
                    masked_indices[i] = self.tokenizer.special_tokens['[MASK]']
                elif rand < 0.9:
                    # Robust Fix: Use len(code_to_idx) to ensure we stay within bounds
                    current_actual_vocab = len(self.tokenizer.code_to_idx)
                    masked_indices[i] = np.random.randint(len(self.tokenizer.special_tokens), current_actual_vocab)
                # else keep original
        
        result = {
            'code_indices': torch.tensor(masked_indices, dtype=torch.long), # including (e.g. [MASK]) and code_indices to ignore (e.g. pad, rest 85% cases))
            'labels': torch.tensor(labels, dtype=torch.long),
            'timestamps': timestamps,
            'first_timestamp': first_timestamp,
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
            'original_codes': original_codes_list
        }
        
        # Add code_types if available
        if original_code_types_list is not None:
            result['code_types'] = original_code_types_list
        
        return result


def medical_collate_fn(batch):
    """
    Custom collate function to handle datetime objects and strings
    """
    # Separate keys that should not be converted to Tensors
    # These will be kept as raw Python lists
    list_keys = ['timestamps', 'first_timestamp', 'original_codes', 'code_types']
    
    collated = {}
    for key in batch[0].keys():
        if key in list_keys:
            # Keep as a simple list of objects
            collated[key] = [d[key] for d in batch]
        else:
            # Use default PyTorch collation for Tensors and numbers
            collated[key] = default_collate([d[key] for d in batch])
    return collated


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_medbert(
    model: MedicalBERT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    num_epochs: int = 100,
    learning_rate: float = 5e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    patience: int = 2,
    reduce_factor: float = 2.0
):
    """Train Medical BERT model"""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    best_val_loss = float('inf')
    patience_counter = 0
    current_lr = learning_rate
    
    for epoch in range(num_epochs):
        # Training
        print(f"processing epoch {epoch+1}/{num_epochs}...")
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for batch in tqdm(train_loader):
            code_indices = batch['code_indices'].to(device)
            labels = batch['labels'].to(device)
            timestamps = batch['timestamps']
            first_timestamps = batch.get('first_timestamp', None)
            attention_mask = batch['attention_mask'].to(device)
            original_codes = batch.get('original_codes', None)
            code_types = batch.get('code_types', None)
            
            # Convert attention mask to format expected by model (True = padding)
            attention_mask = ~attention_mask
            
            optimizer.zero_grad()
            
            logits = model(
                code_indices, 
                timestamps, 
                first_timestamps=first_timestamps,
                original_codes=original_codes, 
                code_types=code_types, 
                attention_mask=attention_mask
            )
            
            # Reshape for loss calculation
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                code_indices = batch['code_indices'].to(device)
                labels = batch['labels'].to(device)
                timestamps = batch['timestamps']
                first_timestamps = batch.get('first_timestamp', None)
                attention_mask = batch['attention_mask'].to(device)
                original_codes = batch.get('original_codes', None)
                code_types = batch.get('code_types', None)
                attention_mask = ~attention_mask
                
                logits = model(
                    code_indices, 
                    timestamps, 
                    first_timestamps=first_timestamps,
                    original_codes=original_codes, 
                    code_types=code_types, 
                    attention_mask=attention_mask
                )
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Learning rate reduction on plateau
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = f'best_medbert_model_{timestamp}.pt'
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                current_lr /= reduce_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                patience_counter = 0
                print(f"  Reduced learning rate to {current_lr:.6f}")
        
        print()

    # Final Testing Phase
    if test_loader is not None:
        import glob
        ckpts = sorted(glob.glob('best_medbert_model_*.pt'))
        if ckpts:
            best_ckpt = ckpts[-1]
            model.load_state_dict(torch.load(best_ckpt, map_location=device))
            print(f"Reloaded best model from {best_ckpt}")
        
        model.eval()
    
        print("\n" + "="*30)
        print("Final Testing Phase")
        print("="*30)
        model.eval()
        test_loss = 0.0
        test_steps = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                code_indices = batch['code_indices'].to(device)
                labels = batch['labels'].to(device)
                timestamps = batch['timestamps']
                first_timestamps = batch.get('first_timestamp', None)
                attention_mask = batch['attention_mask'].to(device)
                original_codes = batch.get('original_codes', None)
                code_types = batch.get('code_types', None)
                attention_mask = ~attention_mask
                
                logits = model(
                    code_indices, 
                    timestamps, 
                    first_timestamps=first_timestamps,
                    original_codes=original_codes, 
                    code_types=code_types, 
                    attention_mask=attention_mask
                )
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                test_loss += loss.item()
                test_steps += 1
        
        avg_test_loss = test_loss / test_steps
        print(f"🏁 Final Test Loss: {avg_test_loss:.4f}")

if __name__ == "__main__":
    # Fix Random Seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 1. Configuration & Data Loading
    # III
    diag_file_III = str(paths.data_root() / "MIMIC_III_sEHR_cleaned/diagnose_icd10_mapped_50.csv")
    proc_file_III = str(paths.data_root() / "MIMIC_III_sEHR_cleaned/procedure_icd10_mapped_50.csv")
    med_file_III  = str(paths.data_root() / "MIMIC_III_sEHR_cleaned/prescriptions_cleaned_50.csv")

    # IV
    diag_file_IV = str(paths.data_root() / "MIMIC_IV_sEHR_cleaned/diagnose_icd10_mapped_50.csv")
    proc_file_IV = str(paths.data_root() / "MIMIC_IV_sEHR_cleaned/procedure_icd10_mapped_50.csv")
    med_file_IV  = str(paths.data_root() / "MIMIC_IV_sEHR_cleaned/prescriptions_cleaned_50.csv")

    # Sanity check: ensure no '0' / '0.0' in the prescriptions
    for ds_name, med_file in [('III', med_file_III), ('IV', med_file_IV)]:
        df = pd.read_csv(med_file, usecols=['ndc'], low_memory=False)
        ndc_num = pd.to_numeric(df['ndc'], errors='coerce')
        n_zero = ((ndc_num == 0) | ndc_num.isna()).sum()
        print(f"[sanity] {ds_name} prescriptions: {len(df)} rows, "
            f"NDC=0/null: {n_zero}")
        assert n_zero == 0, f"{ds_name} still has {n_zero} rows with NDC=0 or null!"

    data_III = load_data_from_csv(diag_file_III, proc_file_III, med_file_III)
    data_IV = load_data_from_csv(
        diag_file_IV, proc_file_IV, med_file_IV,
        proc_time_col='chartdate',
        med_time_col='starttime',
    )
    data = data_III + data_IV
    
    # 2. Vocab Audit & Tokenizer Initialization
    # We call this once to audit the raw files and get an accurate vocab size
    vocab_size_III = calculate_vocab_size(diag_file_III, proc_file_III, med_file_III)
    vocab_size_IV = calculate_vocab_size(
        diag_file_IV, proc_file_IV, med_file_IV,
        proc_time_col='chartdate',
        med_time_col='starttime',
    )
    vocab_size_mmc = vocab_size_III + vocab_size_IV 
    
    print("Building tokenizer vocabulary...")
    tokenizer = MedicalCodeTokenizer(vocab_size=None)
    
    # Build vocabulary from actual data
    # Extract all code sequences from data
    all_code_sequences = [item['codes'] for item in data]
    tokenizer.build_vocab(all_code_sequences)
    
        # --- NEW: Check Vocabulary Coverage ---
    # Encode all codes to see how many were forced into [UNK]
    all_encoded_indices = []
    for seq in all_code_sequences:
        all_encoded_indices.extend(tokenizer.encode(seq))
    
    if all_encoded_indices:
        unk_idx = tokenizer.special_tokens['[UNK]']
        unk_count = sum(1 for idx in all_encoded_indices if idx == unk_idx)
        coverage = 1 - unk_count / len(all_encoded_indices)
        print(f"✓ Vocabulary Coverage: {coverage:.2%}")
        print(f"  (Total tokens: {len(all_encoded_indices)}, UNK tokens: {unk_count})")
    # --------------------------------------
    
    # Point 4: Use actual tokenizer size for model
    final_vocab_size = len(tokenizer.code_to_idx)
    print(f"Final vocabulary size: {final_vocab_size}")
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    print("\nCreating model...")
    model = MedicalBERT(
        vocab_size=final_vocab_size,
        hidden_size=320,
        num_layers=5,
        num_heads=5,
        max_seq_length=512
    )
    model = model.to(device) # Move model to GPU immediately
    
    # Save tokenizer so we can use it for inference later
    tokenizer.save(str(paths.data_root() / "medical_tokenizer_combined.json"))
    print("✓ Tokenizer saved to medical_tokenizer_combined.json")
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create dataset and dataloader
    print("\nCreating datasets and dataloaders using global splits...")
    
    # 1. Load the global split lists
    SPLIT_DIR = str(paths.data_root() / "splits/sEHR_splits_combined")
    try:
        with open(f"{SPLIT_DIR}/splits.json") as f:
            splits = json.load(f)

        train_pids = set(splits['train']['III']) | set(splits['train']['IV'])
        val_pids   = set(splits['val']['III'])   | set(splits['val']['IV'])
        print(f"✓ Loaded global splits (Train/Val/Test) from {SPLIT_DIR}")
    except FileNotFoundError:
        print(f"❌ Error: Split files not found in {SPLIT_DIR}. Run export_patient_splits.py first.")
        exit(1)

    print("=== Patient ID format check ===")
    print(f"Split train pid sample (III): {[p for p in list(splits['train']['III'])[:3]]}")
    print(f"Split train pid sample (IV):  {[p for p in list(splits['train']['IV'])[:3]]}")
    print(f"Data pid sample (first 5):    {[d['patient_id'] for d in data[:5]]}")
    print(f"Split pid type: {type(splits['train']['III'][0])}")
    print(f"Data pid type:  {type(data[0]['patient_id'])}")

    # 2. Filter data into splits based on global PID lists
    train_data = [d for d in data if d['patient_id'] in train_pids]
    val_data = [d for d in data if d['patient_id'] in val_pids]
    
    print(f"Total formatted patients: {len(data)}")
    print(f"Matched Train size: {len(train_data)}")
    print(f"Matched Val size: {len(val_data)}")
    
    if len(train_data) == 0 or len(val_data) == 0:
        print("❌ Error: No patients matched the split lists. Check if IDs in CSV match IDs in patient_history_db.joblib.")
        exit(1)
    
    train_dataset = EHRDataset(train_data, tokenizer, max_length=512, mask_prob=0.15)
    val_dataset = EHRDataset(val_data, tokenizer, max_length=512, mask_prob=0.15)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=128,
        shuffle=True, 
        collate_fn=medical_collate_fn,
        num_workers=8,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=128,
        shuffle=False, 
        collate_fn=medical_collate_fn,
        num_workers=8,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    train_medbert(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader = None,
        num_epochs=100,
        learning_rate=5e-4,
        device=device
    )
    
    print("\nTraining completed!")


