import os
from transformers import AutoTokenizer

def build_llama_vocab(output_dir):
    token = os.environ['HUGGING_FACE_TOKEN']
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=token)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "llama_vocab.txt")
    vocab_items = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])

    with open(output_path, "w", encoding="utf-8") as f:
        for token, _ in vocab_items:
            f.write(token + "\n")

    print(f"Vocabulary saved to {output_path}")
