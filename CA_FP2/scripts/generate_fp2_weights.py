#!/usr/bin/env python3
import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_model_root(root: Path) -> Path:
    candidates = [
        root / "FP2_model",
        root.parent / "FP2_model",
        root.parent.parent / "FP2_model",
    ]
    for path in candidates:
        if (path / "workload_config.json").is_file():
            return path
    raise FileNotFoundError("Could not find FP2_model/workload_config.json")


def normalize_group(group: str) -> str:
    group = group.upper()
    if group.startswith("G") and len(group) == 3 and group[1:].isdigit():
        return group
    if group.isdigit():
        return f"G{int(group):02d}"
    raise ValueError(f"invalid group: {group}")


def load_state_dict(path: Path) -> OrderedDict:
    obj = torch.load(path, map_location="cpu")
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    elif isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"{path} did not load to a state_dict-like object")
    return OrderedDict((str(k), v) for k, v in obj.items() if torch.is_tensor(v))


def linear_pairs(state: OrderedDict, expected_layers: int) -> list[tuple[str, str]]:
    weights = [k for k in state if k.endswith(".weight") and state[k].ndim == 2]
    weights.sort(key=lambda k: [int(p) if p.isdigit() else p for p in k.split(".")])
    pairs = []
    for wkey in weights:
        bkey = wkey[:-len(".weight")] + ".bias"
        if bkey not in state:
            raise KeyError(f"missing bias for {wkey}: expected {bkey}")
        pairs.append((wkey, bkey))
    if len(pairs) != expected_layers:
        raise ValueError(f"expected {expected_layers} Linear layers, found {len(pairs)}")
    return pairs


def simulator_array(tensor: torch.Tensor, suffix: str) -> np.ndarray:
    arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    if suffix == "weight":
        if arr.ndim != 2:
            raise ValueError(f"Linear weight must be 2-D, got shape {arr.shape}")
        arr = arr.T
    return np.ascontiguousarray(arr)


def tensor_bytes(tensor: torch.Tensor, suffix: str) -> bytes:
    return simulator_array(tensor, suffix).tobytes(order="C")


def slice_shape(tensor: torch.Tensor, suffix: str) -> list[int]:
    arr = simulator_array(tensor, suffix)
    if arr.ndim == 1:
        return [int(arr.shape[0]), 1]
    if arr.ndim == 2:
        return [int(arr.shape[0]), int(arr.shape[1])]
    return [int(np.prod(arr.shape)), 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="G01")
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--submission-root", type=Path, default=None)
    args = parser.parse_args()

    root = repo_root()
    group = normalize_group(args.group)
    model_root = args.model_root or default_model_root(root)
    submission_root = args.submission_root or root / "FP2" / "submission_g01"
    group_dir = submission_root / f"CA_FP2_{group}"
    weights_dir = group_dir / f"weights_{group}"
    weights_dir.mkdir(parents=True, exist_ok=True)

    config_path = model_root / "workload_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    manifest = {}
    print(f"workload_config: {config_path}")
    print(f"weights_dir    : {weights_dir}")

    for model_id in range(10):
        mlp_key = f"MLP_ID_{model_id}"
        if mlp_key not in config:
            raise KeyError(f"missing {mlp_key} in {config_path}")
        expected_layers = int(config[mlp_key]["total_layers"])
        pt_path = model_root / "mlp_weights" / f"mlp_{model_id}" / f"model_{model_id}.pt"
        state = load_state_dict(pt_path)

        print()
        print(f"========== model {model_id}: {pt_path} ==========")
        for key, tensor in state.items():
            print(f"{key}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")

        pairs = linear_pairs(state, expected_layers)
        bin_name = f"model_{model_id}_weights_{group}.bin"
        bin_path = weights_dir / bin_name
        layer_obj = {}
        offset = 0

        with bin_path.open("wb") as f:
            for layer_idx, (wkey, bkey) in enumerate(pairs):
                for suffix, src_key in (("weight", wkey), ("bias", bkey)):
                    tensor = state[src_key]
                    data = tensor_bytes(tensor, suffix)
                    sim_name = f"model.{layer_idx}.{suffix}"
                    layer_obj[sim_name] = {
                        "0": {
                            "bin_file": bin_name,
                            "bin_offset_bytes": offset,
                            "bin_size_bytes": len(data),
                            "slice_shape": slice_shape(tensor, suffix),
                            "source_key": src_key,
                            "tensor_shape": [int(x) for x in tensor.shape],
                        }
                    }
                    f.write(data)
                    offset += len(data)

        manifest[f"model.{model_id}."] = layer_obj
        print(f"wrote {bin_path} ({offset} bytes)")

    manifest_path = weights_dir / f"memory_map_sim_{group}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
