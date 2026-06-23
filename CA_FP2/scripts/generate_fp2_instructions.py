#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


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
    group_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((model_root / "workload_config.json").read_text(encoding="utf-8"))
    input_offsets = []
    output_offsets = []
    in_acc = 0
    out_acc = 0
    for mid in range(10):
        arch = cfg[f"MLP_ID_{mid}"]["architecture"]
        input_offsets.append(in_acc)
        output_offsets.append(out_acc)
        in_acc += int(arch[0])
        out_acc += int(arch[-1])
    assert in_acc == 2799, in_acc
    assert out_acc == 122, out_acc

    for mid in range(10):
        arch = [int(x) for x in cfg[f"MLP_ID_{mid}"]["architecture"]]
        lines = [
            f"# MLP{mid}: architecture {arch}",
            f"# input TOKEN_I offset={input_offsets[mid]}, output TOKEN_O offset={output_offsets[mid]}",
            f"DRAM,READ,TOKEN_I,0,0,{input_offsets[mid]},{arch[0]}",
        ]
        for layer_idx in range(len(arch) - 1):
            in_dim = arch[layer_idx]
            out_dim = arch[layer_idx + 1]
            is_final = layer_idx == len(arch) - 2
            lines.extend([
                f"# layer {layer_idx}: Linear {in_dim}->{out_dim}",
                f"ARRAY,IS,model.{layer_idx}.weight,{mid},0,{in_dim},{out_dim}",
                f"DRAM,READ,MODEL,model.{layer_idx}.bias,3,0,{mid},0,{out_dim}",
                f"DMA,1,0,2,0,{out_dim}",
                f"VECTOR,4,{out_dim}",
            ])
            if is_final:
                lines.extend([
                    f"DMA,4,0,2,0,{out_dim}",
                    f"VECTOR,2,{out_dim}",
                    f"DRAM,WRITE,TOKEN_O,4,0,{output_offsets[mid]},{out_dim}",
                ])
            else:
                lines.extend([
                    f"DMA,4,0,2,0,{out_dim}",
                    f"VECTOR,3,{out_dim}",
                    f"DMA,4,0,0,0,{out_dim}",
                ])
        (group_dir / f"core_{mid}_{group}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for cid in range(10, 64):
        (group_dir / f"core_{cid}_{group}.csv").write_text("# idle core\n", encoding="utf-8")

    print(f"wrote core_0..core_63 under {group_dir}")
    print(f"input_total={in_acc} output_total={out_acc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
