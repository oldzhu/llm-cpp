#!/usr/bin/env python3
"""Extract SentencePiece .model vocabulary to a simple JSON file.

Usage: python scripts/extract_spm_vocab.py data/TinyLlama-1.1B/tokenizer.model data/spm_vocab.json
"""

import json
import sys

try:
    import sentencepiece as spm
except ImportError:
    print("Error: sentencepiece not installed. Run: pip install sentencepiece")
    sys.exit(1)

def extract(model_path, output_path):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    vocab = {}
    for i in range(sp.GetPieceSize()):
        piece = sp.IdToPiece(i)
        score = sp.GetScore(i)
        vocab[piece] = {"id": i, "score": score}
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=1)
    
    print(f"Extracted {len(vocab)} tokens from {model_path} -> {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/extract_spm_vocab.py <model_path> <output_json>")
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2])
