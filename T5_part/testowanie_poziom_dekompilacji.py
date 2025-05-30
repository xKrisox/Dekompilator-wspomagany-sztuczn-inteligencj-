import json
import re
from difflib import SequenceMatcher
from transformers import T5Tokenizer, T5ForConditionalGeneration

# === 1. Normalizacja kodu (ignorujemy składnię, zostawiamy liczby) ===

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

# === 2. Porównanie z kontrolą wartości liczbowych ===

def semantically_similar_with_numbers(expected: str, generated: str) -> float:
    norm_expected = normalize_code_preserve_numbers(expected)
    norm_generated = normalize_code_preserve_numbers(generated)

    ratio = SequenceMatcher(None, norm_expected, norm_generated).ratio()

    expected_nums = set(re.findall(r"\b\d+\b", expected))
    generated_nums = set(re.findall(r"\b\d+\b", generated))

    if expected_nums != generated_nums:
        ratio *= 0.85  # kara za różne liczby

    return ratio * 100

# === 3. Generowanie kodu na podstawie HEX ===

def generate_code(hex_input, model, tokenizer):
    input_encoding = tokenizer(
        hex_input,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = input_encoding["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# === 4. Ewaluacja całego zbioru testowego z progiem 82% ===

def evaluate_semantic_similarity(model_name, dataset_file="dataset_test.json", threshold=82.0):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()

    with open(dataset_file, "r") as f:
        data = json.load(f)

    total_score = 0.0
    results = []
    fit_count = 0

    for example in data:
        hex_input = example["hex"]
        expected_code = example["source"]

        generated_code = generate_code(hex_input, model, tokenizer)
        score = semantically_similar_with_numbers(expected_code, generated_code)
        total_score += score

        is_fit = score >= threshold
        if is_fit:
            fit_count += 1
            results.append({
                "hex": hex_input,
                "expected": expected_code.strip(),
                "generated": generated_code.strip(),
                "semantic_similarity": score
            })

        print("="*60)
        print(f"HEX:\n{hex_input}")
        print(f"EXPECTED:\n{expected_code.strip()}")
        print(f"GENERATED:\n{generated_code.strip()}")
        print(f"SEMANTIC SIMILARITY: {score:.2f}% — {'FIT ✔️' if is_fit else 'UNFIT ❌'}")

    avg_similarity = total_score / len(data)
    print("\n==== PODSUMOWANIE ====")
    print(f"Liczba przykładów: {len(data)}")
    print(f"Liczba FIT (>= {threshold:.1f}%): {fit_count}")
    print(f"Średnie podobieństwo: {avg_similarity:.2f}%")

    return results, avg_similarity

# === 5. Uruchomienie testu ===

if __name__ == "__main__":
    MODEL_PATH = "code_decompiler_model_20000"
    DATASET_FILE = "dataset_test.json"
    THRESHOLD = 82.0

    results, avg = evaluate_semantic_similarity(MODEL_PATH, DATASET_FILE, THRESHOLD)
    # Możesz zapisać `results` do pliku JSON, jeśli chcesz:
    # with open("results_above_threshold.json", "w") as out:
    #     json.dump(results, out, indent=2)
