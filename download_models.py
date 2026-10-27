"""Run with network access. Inference itself uses local_files_only=True."""
import argparse
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("models", nargs="*", default=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B", "openai/gpt-oss-20b"])
args = parser.parse_args()
for name in args.models:
    print(f"Downloading {name}", flush=True)
    snapshot_download(name,
        allow_patterns=["config.json", "generation_config.json", "model-*.safetensors",
                        "model.safetensors", "model.safetensors.index.json", "tokenizer*",
                        "special_tokens_map.json", "chat_template.jinja", "*.txt", "LICENSE", "README.md"],
        ignore_patterns=["original/*", "metal/*"])
    if name == "openai/gpt-oss-20b":
        # MXFP4 layers request this kernel at load time. Cache it now so the
        # service can keep HF_HUB_OFFLINE=1 during inference.
        from kernels import get_kernel
        get_kernel("kernels-community/triton_kernels")
    print(f"Ready: {name}", flush=True)
