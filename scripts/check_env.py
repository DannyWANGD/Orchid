"""Record the Orchid interpreter, installed pins and an actual CUDA backward pass."""

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
KEY_PACKAGES = (
    "torch", "triton", "cuda-toolkit", "cuda-bindings", "nvidia-cudnn-cu12",
    "numpy", "PyYAML", "regex", "matplotlib",
)


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def pins(path):
    expected = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not (line := line.strip()) or line.startswith(("#", "--")):
            continue
        name, sep, ver = line.partition("==")
        if not (name and sep and ver):
            raise ValueError(f"Expected a pinned requirement in {path}: {line}")
        expected[name] = ver
    return expected


def probe(verify_pins, requirements=ROOT / "requirements.txt"):
    import torch

    check = subprocess.run(
        [sys.executable, "-I", "-m", "pip", "check"],
        capture_output=True, text=True, check=False,
    )
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--id=0", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        driver = None
    result = {
        "python": platform.python_version(),
        "python_expected": (ROOT / ".python-version").read_text().strip(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "isolated_python_path": bool(sys.flags.ignore_environment or not os.getenv("PYTHONPATH")),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "driver": driver,
        "gpu": None,
        "bf16_native": False,
        "cuda_forward_backward": False,
        "packages": {p: version(p) for p in KEY_PACKAGES},
        "pip_check": {"ok": check.returncode == 0, "output": (check.stdout + check.stderr).strip()},
    }
    if result["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        result["gpu"] = {"name": props.name, "memory_bytes": props.total_memory,
                         "capability": [props.major, props.minor]}
        result["bf16_native"] = torch.cuda.is_bf16_supported(including_emulation=False)
        x = torch.ones(64, 64, device="cuda", requires_grad=True)
        (x @ x).sum().backward()
        torch.cuda.synchronize()
        result["cuda_forward_backward"] = bool((x.grad == 128).all().item())
    if verify_pins:
        expected = pins(requirements)
        missing = {p: {"expected": v, "actual": actual}
                   for p, v in expected.items() if (actual := version(p)) != v}
        result["pinned_requirements"] = {"count": len(expected), "mismatches": missing}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-pins", action="store_true")
    parser.add_argument("--requirements", type=Path, default=ROOT / "requirements.txt",
                        help="Pin file to check (use requirements-cpu.txt for the CPU build)")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = probe(args.verify_pins, args.requirements)
    document = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(document, encoding="utf-8")
    print(document, end="")
    return int(not (
        result["pip_check"]["ok"]
        and (not args.verify_pins or (
            not result["pinned_requirements"]["mismatches"]
            and result["python"] == result["python_expected"]
        ))
        and (not args.require_cuda or result["cuda_forward_backward"])
    ))


if __name__ == "__main__":
    raise SystemExit(main())
