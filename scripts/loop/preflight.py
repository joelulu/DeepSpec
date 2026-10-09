"""Validate existing cache, target metadata and launch shape before spawning GPUs."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from transformers import AutoConfig
from deepspec.data.target_cache_dataset import CacheDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--global-batch", type=int, default=16)
    parser.add_argument("--local-batch", type=int, default=1)
    parser.add_argument("--data-percent", default="100")
    parser.add_argument("--holdout-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-cuda", action="store_true")
    args = parser.parse_args()
    dataset = CacheDataset(cache_dir=args.cache, data_percent=args.data_percent,
        holdout_samples=args.holdout_samples, subset_seed=args.seed)
    manifest = dataset.manifest
    target = AutoConfig.from_pretrained(args.target)
    if target.model_type != "qwen3":
        raise ValueError("The loop pilot currently supports Qwen3 targets only")
    if str(manifest["target_model_name_or_path"]) != args.target:
        raise ValueError("TARGET_MODEL must match target_model_name_or_path in manifest.json")
    if int(manifest["hidden_size"]) != int(target.hidden_size):
        raise ValueError("Cache and target hidden_size differ")
    layer_ids = [int(index) for index in manifest["target_layer_ids"]]
    if any(index < -1 or index >= target.num_hidden_layers - 1 for index in layer_ids):
        raise ValueError("Target features must exclude the final decoder layer; inspect cache metadata")
    if len(dataset) < args.global_batch or min(args.global_batch, args.local_batch) < 1:
        raise ValueError("Need at least one positive full global batch of cached samples")
    sample = dataset[0]
    if sample["input_ids"].numel() < 2 or not sample["loss_mask"].any():
        raise ValueError("First cache sample has no usable response supervision")
    if args.check_cuda:
        count = torch.cuda.device_count()
        if count == 0:
            raise RuntimeError("No visible CUDA GPU; set CUDA_VISIBLE_DEVICES and use a CUDA torch build")
        if args.global_batch % (count * args.local_batch):
            raise ValueError("GLOBAL_BATCH must be divisible by visible GPUs * LOCAL_BATCH")
        print("GPUs:", [torch.cuda.get_device_name(index) for index in range(count)])
    print(json.dumps(dict(target=args.target, source_samples=dataset.source_num_samples, samples=len(dataset),
        data_percent=args.data_percent, holdout_samples=args.holdout_samples, target_layer_ids=layer_ids,
        hidden_size=target.hidden_size, first_sample_tokens=sample["input_ids"].numel(),
        cache_version=manifest["version"]), indent=2))


if __name__ == "__main__":
    main()
