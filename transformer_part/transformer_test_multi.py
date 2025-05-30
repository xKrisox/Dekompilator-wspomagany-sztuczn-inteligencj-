import json
import torch
from tqdm import tqdm
import difflib
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re
from difflib import SequenceMatcher

def normalize_code_preserve_numbers(code: str) -> str:
    if not code:
        return ""

    C_KEYWORDS = {
        "int", "return", "if", "else", "for", "while", "switch", "case", "break",
        "continue", "void", "main", "char", "float", "double", "struct", "typedef"
    }

    code = code.lower()
    code = re.sub(r"[{}\[\];\(\)\n\r\t]", " ", code)
    code = re.sub(r"\s+", " ", code)

    tokens = code.split()
    normalized = []
    for token in tokens:
        if token in C_KEYWORDS:
            normalized.append(token)
        elif re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", token):
            normalized.append("X")
        elif re.fullmatch(r"\d+", token):
            normalized.append(token)
        else:
            normalized.append(token)

    return " ".join(normalized).strip()

def semantically_similar_with_numbers(expected: str, generated: str) -> float:
    norm_expected = normalize_code_preserve_numbers(expected)
    norm_generated = normalize_code_preserve_numbers(generated)

    ratio = SequenceMatcher(None, norm_expected, norm_generated).ratio()

    expected_nums = set(re.findall(r"\b\d+\b", expected))
    generated_nums = set(re.findall(r"\b\d+\b", generated))

    if expected_nums != generated_nums:
        ratio *= 0.85

    return ratio * 100


def greedy_decode(model, tokenizer, src_hex, max_length=512, device="cpu"):
    model.eval()
    src = tokenizer(src_hex, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)
    src_input = src["input_ids"].to(device)
    src_pad_mask = (src_input == tokenizer.pad_token_id)

    decoded_tokens = [tokenizer.pad_token_id]

    for i in range(max_length):
        tgt_input = torch.tensor([decoded_tokens], dtype=torch.long).to(device)
        tgt_pad_mask = (tgt_input == tokenizer.pad_token_id)
        tgt_mask = torch.triu(torch.full((tgt_input.size(1), tgt_input.size(1)), float('-inf')), diagonal=1).to(device)

        with torch.no_grad():
            output = model(
                src_input,
                tgt_input,
                src_key_padding_mask=src_pad_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                tgt_mask=tgt_mask
            )
            next_token_logits = output[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1).item()
            decoded_tokens.append(next_token)
            if hasattr(tokenizer, "eos_token_id") and next_token == tokenizer.eos_token_id:
                break

    out = tokenizer.decode(decoded_tokens[1:], skip_special_tokens=True)
    return out

def compute_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio() * 100

def compute_bleu(ref, hyp):
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=SmoothingFunction().method1) * 100

# ---- Model, tokenizer, device ----

from transformers import AutoTokenizer
import torch.nn as nn

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

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048, max_len=512, dropout=0.1):
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
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):
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

tokenizer = AutoTokenizer.from_pretrained("t5-small")
vocab_size = tokenizer.vocab_size
model = TransformerSeq2Seq(vocab_size, d_model=256, nhead=4, num_encoder_layers=3, num_decoder_layers=3, max_len=512)
model.load_state_dict(torch.load("transformer_decompiler_20000.pth", map_location="cpu"))
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ---- Wczytaj dane ----

with open("dataset_test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# ---- Testuj i wypisuj ----

similarity_scores = []
bleu_scores = []
semantic_scores = []

for example in tqdm(test_data):
    hex_input = example["hex"]
    gold_source = example["source"].strip()
    prediction = greedy_decode(model, tokenizer, hex_input, max_length=512, device=device).strip()

    similarity = compute_similarity(gold_source, prediction)
    bleu = compute_bleu(gold_source, prediction)
    semantic_score = semantically_similar_with_numbers(gold_source, prediction)
    
    semantic_scores.append(semantic_score)


    similarity_scores.append(similarity)
    bleu_scores.append(bleu)

    print(f"HEX: {hex_input}")
    print(f"Expected:\n{gold_source}\n")
    print(f"Generated: {prediction}\n")
    print(f"Similarity: {similarity:.2f}%")
    print(f"BLEU Score: {bleu:.2f}%\n")
    print(f"Semantic Similarity: {semantic_score:.2f}%")
    print("="*60)

# ---- Podsumowanie ----

mean_similarity = sum(similarity_scores) / len(similarity_scores)
mean_bleu = sum(bleu_scores) / len(bleu_scores)
avg_semantic = sum(semantic_scores) / len(semantic_scores)

print(f"\n==== PODSUMOWANIE OGÓLNE ====")
print(f"Średnia Similarity: {mean_similarity:.2f}%")
print(f"Średni BLEU Score: {mean_bleu:.2f}%")
print(f"Średni wynik semantycznej zgodności: {avg_semantic:.2f}%")
