import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time 
# 1. Klasa Dataset dla danych treningowych
class CodeHexDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        with open(file_path, "r") as f:
            data = json.load(f)
        self.source = [item["hex"] for item in data]  # Hex code as input
        self.target = [item["source"] for item in data]  # C code as output
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        source_encoding = self.tokenizer(
            self.source[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_encoding = self.tokenizer(
            self.target[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = target_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens
        return {
            "input_ids": source_encoding["input_ids"].squeeze(0),
            "attention_mask": source_encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }

# 2. Inicjalizacja modelu i tokenizer
model_name = "t5-small"  # Możesz użyć większego modelu, np. "t5-large" lub CodeT5
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 3. Przygotowanie danych
dataset = CodeHexDataset(file_path="dataset_20000.json", tokenizer=tokenizer, max_length=512)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 4. Definicja optymalizatora
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 5. Trenowanie modelu
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
start_time = time.time()
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
total_time = time.time() - start_time    # <- Stop licznika
print(f"Czas trenowania: {total_time:.2f} sekund ({total_time/60:.2f} minut)")
# 6. Zapisanie wytrenowanego modelu
model.save_pretrained("code_decompiler_model_20000")
tokenizer.save_pretrained("code_decompiler_model_20000")
