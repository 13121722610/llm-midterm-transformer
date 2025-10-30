# src/model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, causal=False):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.causal = causal
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B, T, C = q.size()
        
        q = self.w_q(q).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.w_k(k).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.w_v(v).view(B, T, self.n_head, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # 因果掩码（用于decoder self-attention）
        if self.causal:
            causal_mask = torch.tril(torch.ones(T, T, device=q.device)).view(1, 1, T, T)
            scores = scores.masked_fill(causal_mask == 0, -1e9)
        
        # 外部掩码（用于padding mask等）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_head, causal=True)  # 因果注意力
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention (causal)
        self_attn_out = self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), mask)
        x = x + self.dropout(self_attn_out)
        
        # Feedforward
        ff_out = self.ff(self.ln2(x))
        x = x + self.dropout(ff_out)
        return x

class DecoderOnlyTransformer(nn.Module):
    """Decoder-Only Transformer (类似GPT架构)"""
    def __init__(self, vocab_size, d_model=256, n_layer=6, n_head=8, d_ff=1024, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 使用简化的DecoderBlock，只包含self-attention和feedforward
        self.layers = nn.ModuleList([DecoderBlock(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x, mask=None):
        B, T = x.size()
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        x = self.tok_emb(x) + self.pos_emb(positions)
        x = self.dropout(x)
        
        # 创建因果掩码
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        
        for layer in self.layers:
            x = layer(x, causal_mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
