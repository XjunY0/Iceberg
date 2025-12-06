#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import subprocess
import sys
from pathlib import Path

SUPPORTED_ALGOS = {
    # Implemented shell scripts
    "hnsw": "scripts/run_hnsw.sh",
    "nsg": "scripts/run_nsg.sh",
    "ipnsw": "scripts/run_ipnsw.sh",
    "ipnsw+": "scripts/run_ipnsw+.sh",
    "dblsh": "scripts/run_dblsh.sh",
    "fargo": "scripts/run_fargo.sh",
    "mag": "scripts/run_mag.sh",
    "mobius": "scripts/run_mobius.sh",
    "napg": "scripts/run_napg.sh",
    "rabitq": "scripts/run_rabitq.sh",
    "vamana": "scripts/run_vamana.sh",
    "ivfpq": "scripts/run_ivfpq.sh",
    "scann": "scripts/run_scann.sh",
}

def sh(cmd, cwd=None, check=True):
    print("$", cmd if isinstance(cmd, str) else " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), check=check)


def image_exists(tag: str) -> bool:
    try:
        proc = subprocess.run(["docker", "image", "inspect", tag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return proc.returncode == 0
    except Exception:
        return False


def build_image(image: str, context: Path):
    cmd = [
        "docker", "build",
        "-t", image,
        str(context)
    ]
    sh(cmd)


def resolve_data_root(repo_root: Path, dataset: str, user_data: str | None) -> Path:
    """
    Resolve the host data directory to mount into the container at /workspace/data.
    Priority:
      1) If --data is provided and exists, use it.
      2) If --data is provided but is not a valid path (e.g., mistakenly passed a dataset name),
         fall back to <repo_root>/data.
      3) If --data is not provided, use <repo_root>/data.
    All scripts inside the container read data from /workspace/data according to scripts/config_dataset.sh.
    """
    if user_data:
        p = Path(user_data)
        if p.exists():
            return p
        print(f"[INFO] --data path does not exist: {user_data}. Falling back to <repo_root>/data")
    return repo_root / "data"


def run_container(image: str, project_dir: Path, data_dir: Path, algo: str, dataset: str, mode: str, extra_env: dict | None = None, workdir: str = "/workspace"):
    algo_key = algo.lower()
    if algo_key not in SUPPORTED_ALGOS:
        print(f"[ERROR] Unsupported algorithm: {algo}. Supported: {sorted(SUPPORTED_ALGOS.keys())}")
        sys.exit(2)

    inner_script = SUPPORTED_ALGOS[algo_key]

    # Ensure script exists on the host project tree before running the container.
    script_abs = project_dir / inner_script
    if not script_abs.exists():
        print(
            f"[ERROR] Missing algorithm script: {inner_script}\n"
            f"Please provide the script under scripts/, or choose an algorithm with an existing script.\n"
            f"Examples: scripts/run_hnsw.sh, scripts/run_nsg.sh"
        )
        sys.exit(2)

    # Inject OpenBLAS_DIR by default to help CMake find OpenBLAS in Ubuntu 22.04 images.
    if extra_env is None:
        extra_env = {}
    extra_env.setdefault("OpenBLAS_DIR", "/usr/lib/x86_64-linux-gnu/openblas-pthread/cmake/openblas")

    env_args = ["-e", f"MODE={mode}", "-e", f"DATASET={dataset}", "-e", "PROJECT_ROOT=/workspace"]
    for k, v in extra_env.items():
        env_args += ["-e", f"{k}={v}"]

    volumes = [
        "-v", f"{project_dir.resolve()}:{workdir}:rw",
        "-v", f"{data_dir.resolve()}:{workdir}/data:ro",
    ]

    cmd = [
        "docker", "run", "--rm",
        *env_args,
        *volumes,
        image,
        "bash", "-lc",
        f"set -e; cd {workdir}; rm -rf build; chmod +x scripts/*.sh || true; {inner_script}"
    ]
    sh(cmd)


def main():
    parser = argparse.ArgumentParser(description="Iceberg Docker build & evaluation runner")
    parser.add_argument("algorithm", type=str, help="Algorithm: " + ", ".join(sorted(SUPPORTED_ALGOS.keys())))
    parser.add_argument("dataset", type=str, help="Dataset name (must match a dataset_* function suffix in scripts/config_dataset.sh, e.g., imagenet1k_dinov2)")
    parser.add_argument("--mode", default="search", choices=["build", "search"], help="Run mode: build or search (default: search)")
    parser.add_argument("--image", default="iceberg:latest", help="Docker image tag (default: iceberg:latest)")
    parser.add_argument("--no-build", action="store_true", help="Skip docker build even if image is missing (not recommended)")
    parser.add_argument("--project-root", default=None, help="Host project root (defaults to this script's repo root)")
    parser.add_argument("--data", default=None, help="Host data root to mount as /workspace/data. If omitted or invalid, defaults to <repo_root>/data.")

    args = parser.parse_args()

    repo_root = Path(args.project_root or Path(__file__).resolve().parent)
    docker_context = repo_root

    # Resolve data directory (mounted to /workspace/data inside the container)
    data_dir = resolve_data_root(repo_root, args.dataset, args.data)
    if not data_dir.exists():
        print(f"[WARN] Data directory does not exist: {data_dir}. The container will not be able to read dataset files.")

    # Build image if needed
    if args.no_build:
        print(f"[INFO] --no-build specified. Skipping image build: {args.image}")
    else:
        if image_exists(args.image):
            print(f"[INFO] Docker image already exists. Skipping build: {args.image}")
        else:
            build_image(args.image, docker_context)

    run_container(
        image=args.image,
        project_dir=repo_root,
        data_dir=data_dir,
        algo=args.algorithm,
        dataset=args.dataset,
        mode=args.mode,
        extra_env=None,
    )


if __name__ == "__main__":
    main()
