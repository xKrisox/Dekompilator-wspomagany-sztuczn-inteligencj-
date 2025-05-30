import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import time

# 1. Dataset
class CodeHexDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        with open(file_path, "r") as f:
            data = json.load(f)
        self.source = [item["hex"] for item in data]
        self.target = [item["source"] for item in data]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src = self.tokenizer(self.source[idx], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        tgt = self.tokenizer(self.target[idx], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": src["input_ids"].squeeze(0),
            "target_ids": tgt["input_ids"].squeeze(0),
        }

# 2. Prosty Positional Encoding (standardowy z Attention is All You Need)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# 3. Model bazujący na torch.nn.Transformer
class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # since PyTorch 1.8
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):
        # src, tgt: [batch, seq]
        src_emb = self.embedding(src)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pos_encoder(tgt_emb)
        output = self.transformer(
            src_emb,
            tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask
        )
        logits = self.fc_out(output)
        return logits

def generate_square_subsequent_mask(sz):
    # Maskuje przyszłe tokeny (do dekodera)
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

# 4. Tokenizer (możesz np. użyć transformers.AutoTokenizer lub własnego)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small") # lub inny, np. BPE

vocab_size = tokenizer.vocab_size

# 5. Dataset i DataLoader
dataset = CodeHexDataset("dataset_20000.json", tokenizer, max_length=512)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 6. Model, optimizer, loss
model = TransformerSeq2Seq(vocab_size, d_model=256, nhead=4, num_encoder_layers=3, num_decoder_layers=3, max_len=512)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
start_time = time.time()
# 7. Trening (prosta pętla)
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in dataloader:
        src = batch["input_ids"].to(device)
        tgt = batch["target_ids"].to(device)

        # przesuwamy target o jeden w prawo do inputów dekodera, a label to już pełny target (patrz: teacher forcing)
        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]

        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        src_pad_mask = (src == tokenizer.pad_token_id)
        tgt_pad_mask = (tgt_input == tokenizer.pad_token_id)

        logits = model(src, tgt_input, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask, tgt_mask=tgt_mask)
        logits = logits.reshape(-1, logits.size(-1))
        tgt_label = tgt_label.reshape(-1)

        loss = criterion(logits, tgt_label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1}, loss: {loss.item()}")
total_time = time.time() - start_time    # <- Stop licznika
print(f"Czas trenowania: {total_time:.2f} sekund ({total_time/60:.2f} minut)")

# 8. Zapis modelu
torch.save(model.state_dict(), "transformer_decompiler_20000.pth")
