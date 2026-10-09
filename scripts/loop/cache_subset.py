"""Build a deterministic v2 target-cache subset without copying tensor shards."""
import argparse
import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import random
import shutil
import struct
import tempfile
from decimal import Decimal

RECORD = struct.Struct("<QIIQQQQQ")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def create_subset(source, output, fraction="0.03", seed=42):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Subset output must differ from the original cache")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["version"] != 2 or manifest["index_record_size"] != RECORD.size:
        raise ValueError("This tool requires the canonical DeepSpec v2 index format")
    total = int(manifest["num_samples"])
    fraction = Decimal(str(fraction))
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    count = int(total * fraction)
    if count < 1:
        raise ValueError("The requested fraction selects no samples")
    source_index = source / "samples.idx"
    index_stat = source_index.stat()
    if index_stat.st_size != total * RECORD.size:
        raise ValueError("Source index size does not match manifest.num_samples")
    if int(manifest["num_shards"]) != len(manifest["shards"]):
        raise ValueError("Source num_shards does not match the shard list")
    for shard_id, shard in enumerate(manifest["shards"]):
        name = Path(shard["file_name"])
        if name.is_absolute() or ".." in name.parts or shard["shard_id"] != shard_id:
            raise ValueError("Shard names must be relative; shard IDs must be dense")
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    identity = dict(source=str(source), source_manifest_sha256=digest(manifest_path),
        source_index_size=index_stat.st_size, source_index_mtime_ns=index_stat.st_mtime_ns,
        source_num_samples=total, fraction=str(fraction), seed=int(seed), num_samples=count)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Serialize repeated/concurrent launches; publish a complete directory atomically.
    with (output.parent / (output.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            metadata = json.loads((output / "subset.json").read_text())
            if any(metadata.get(key) != value for key, value in identity.items()):
                raise ValueError("Existing subset belongs to different data/settings; use a new output directory")
            if digest(output / "samples.idx") != metadata["index_sha256"]:
                raise ValueError("Existing subset index changed; use a new output directory")
            if json.loads((output / "manifest.json").read_text()) != dict(manifest, num_samples=count):
                raise ValueError("Existing subset manifest changed")
            for shard in manifest["shards"]:
                link = output / shard["file_name"]
                if not link.is_symlink() or link.resolve() != (source / shard["file_name"]).resolve():
                    raise ValueError(f"Invalid subset shard link: {link}")
            return identity
        staging = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=output.parent))
        try:
            indices = sorted(random.Random(int(seed)).sample(range(total), count))
            with source_index.open("rb") as handle:
                with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as index:
                    with (staging / "samples.idx").open("wb") as dest:
                        for new_id, old_id in enumerate(indices):
                            fields = list(RECORD.unpack_from(index, old_id * RECORD.size))
                            if fields[0] != old_id or fields[1] >= len(manifest["shards"]):
                                raise ValueError(f"Invalid source record at {old_id}")
                            fields[0] = new_id
                            dest.write(RECORD.pack(*fields))
            for shard in manifest["shards"]:
                link = staging / shard["file_name"]
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(source / shard["file_name"])
            (staging / "manifest.json").write_text(json.dumps(dict(manifest, num_samples=count), indent=2))
            (staging / "selected_source_ids.json").write_text(json.dumps(indices))
            (staging / "subset.json").write_text(json.dumps(dict(identity,
                index_sha256=digest(staging / "samples.idx")), indent=2))
            os.replace(staging, output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return identity


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fraction", default="0.03")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = create_subset(args.source, args.output, args.fraction, args.seed)
    print(f"Cache subset: {result['num_samples']:,}/{result['source_num_samples']:,} samples; seed={args.seed}")
    print(Path(args.output).resolve())


if __name__ == "__main__":
    main()
