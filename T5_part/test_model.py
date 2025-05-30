import json
from difflib import SequenceMatcher
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 1. Ładowanie modelu i tokenizer
model_name = "code_decompiler_model_base_1000"  # Ścieżka do wytrenowanego modelu
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 2. Funkcja generująca kod z dynamicznym dostosowaniem długości
def generate_code(hex_input):
    try:
        input_encoding = tokenizer(
            hex_input,
            max_length=512,  # Maksymalna długość danych wejściowych
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = input_encoding["input_ids"]

        # Dynamiczne ustawienie min_length i max_length
        input_length = len(hex_input)
        if input_length < 30:  # Krótkie dane wejściowe
            min_len = 10
            max_len = 30
        else:  # Dłuższe dane wejściowe
            min_len = 30
            max_len = 512

        output_ids = model.generate(
            input_ids,
            min_length=min_len,
            max_length=max_len,
            num_beams=5,  # Beam search dla lepszej jakości
            repetition_penalty=2.0,  # Kary za powtarzające się tokeny
            early_stopping=True
        )
        generated_code = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return post_process_code(generated_code)  # Dodanie post-processingu
    except Exception as e:
        print(f"Error generating code for HEX input: {hex_input}\nException: {e}")
        return None

# 3. Funkcja do post-processingu kodu
def post_process_code(generated_code):
    # Usuń nadmiarowe średniki i puste linie
    lines = generated_code.split(";")
    cleaned_code = ";".join(line.strip() for line in lines if line.strip())
    return cleaned_code

# 4. Funkcja do obliczania podobieństwa (procentowe)
def calculate_similarity(expected, generated):
    if generated is None:  # Obsługa przypadku, gdy generacja się nie powiodła
        return 0.0
    matcher = SequenceMatcher(None, expected, generated)
    similarity = matcher.ratio() * 100  # Procentowe dopasowanie
    return similarity

# 5. Funkcja do obliczania wskaźnika BLEU
def calculate_bleu(reference, candidate):
    if candidate is None:  # Obsługa przypadku, gdy generacja się nie powiodła
        return 0.0
    smoothie = SmoothingFunction().method4
    score = sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie)
    return score * 100  # BLEU jako procent

# 6. Funkcja do testowania na całym datasetcie
def evaluate_model(dataset_file):
    with open(dataset_file, "r") as f:
        data = json.load(f)
    
    total_similarity = 0
    total_bleu = 0
    results = []
    
    for example in data:
        hex_input = example["hex"]
        expected_output = example["source"]
        
        # Wygenerowany kod
        generated_output = generate_code(hex_input)
        
        # Obliczanie podobieństwa i BLEU
        similarity = calculate_similarity(expected_output, generated_output)
        bleu_score = calculate_bleu(expected_output, generated_output)
        total_similarity += similarity
        total_bleu += bleu_score
        
        # Drukowanie wyników dla danego przykładu
        print(f"HEX: {hex_input}\nExpected: {expected_output}\nGenerated: {generated_output}\nSimilarity: {similarity:.2f}%\nBLEU Score: {bleu_score:.2f}%\n")
        results.append({
            "hex": hex_input,
            "expected": expected_output,
            "generated": generated_output,
            "similarity": similarity,
            "bleu_score": bleu_score
        })
    
    # Średnie podobieństwo i BLEU dla całego datasetu
    average_similarity = total_similarity / len(data)
    average_bleu = total_bleu / len(data)
    print(f"\nŚrednie podobieństwo na całym datasetcie: {average_similarity:.2f}%")
    print(f"Średni BLEU Score na całym datasetcie: {average_bleu:.2f}%")
    return results, average_similarity, average_bleu

# 7. Wywołanie testowania
dataset_file = "dataset_test.json"  # Zmień ścieżkę na lokalizację swojego pliku JSON
results, avg_similarity, avg_bleu = evaluate_model(dataset_file)
