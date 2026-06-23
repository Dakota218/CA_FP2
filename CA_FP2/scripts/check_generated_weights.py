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
    for path in [root / "FP2_model", root.parent / "FP2_model", root.parent.parent / "FP2_model"]:
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


def as_float32_numpy(tensor: torch.Tensor, sim_name: str) -> np.ndarray:
    arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    if sim_name.endswith(".weight"):
        arr = arr.T
    return np.ascontiguousarray(arr)


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
    weights_dir = submission_root / f"CA_FP2_{group}" / f"weights_{group}"
    manifest_path = weights_dir / f"memory_map_sim_{group}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    failures = 0
    checked = 0
    print(f"manifest   : {manifest_path}")
    print(f"weights_dir: {weights_dir}")

    for model_id in range(10):
        pt_path = model_root / "mlp_weights" / f"mlp_{model_id}" / f"model_{model_id}.pt"
        state = load_state_dict(pt_path)
        layer_key = f"model.{model_id}."
        if layer_key not in manifest:
            raise KeyError(f"manifest missing {layer_key}")

        print()
        print(f"========== model {model_id}: {pt_path} ==========")
        for sim_name, slices in sorted(manifest[layer_key].items()):
            entry = slices["0"]
            source_key = entry.get("source_key")
            if source_key not in state:
                print(f"FAIL {sim_name}: source_key not in .pt: {source_key}")
                failures += 1
                continue

            expected = as_float32_numpy(state[source_key], sim_name)
            bin_path = weights_dir / entry["bin_file"]
            offset = int(entry["bin_offset_bytes"])
            size = int(entry["bin_size_bytes"])
            if size != expected.nbytes:
                print(f"FAIL {sim_name}: size mismatch bin={size} expected={expected.nbytes}")
                failures += 1
                continue

            raw = np.fromfile(bin_path, dtype=np.uint8, count=size, offset=offset)
            actual = raw.tobytes()
            actual_arr = np.frombuffer(actual, dtype=np.float32).reshape(expected.shape)
            exact = np.array_equal(actual_arr, expected)
            close = np.allclose(actual_arr, expected, rtol=0.0, atol=0.0)
            status = "PASS" if exact or close else "FAIL"
            print(
                f"{status} {sim_name} <- {source_key} "
                f"shape={tuple(expected.shape)} offset={offset} bytes={size} exact={exact}"
            )
            checked += 1
            if status != "PASS":
                failures += 1

    print()
    print(f"checked={checked} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
