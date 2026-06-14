from huggingface_hub import snapshot_download
import sys
print("Downloading TinyLlama-1.1B (public, LLaMA-family architecture)...")
# TinyLlama uses LLaMA architecture and is publicly accessible
snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0', local_dir='data/TinyLlama-1.1B',
                   ignore_patterns=['*.msgpack','*.h5','*.pth'])
print("Done")
