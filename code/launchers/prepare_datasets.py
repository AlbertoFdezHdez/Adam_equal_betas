import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from utils.paths import LOGS_ROOT, RESULTS_ROOT, ensure_dir
from utils.datasets.utils import configure_cache_env, get_data_dir
from utils.datasets.slimpajama import inspect_local_slimpajama_subset
from utils.runtime import get_nanogpt_root, nanogpt_repo_status


SLIMPAJAMA_REPO_ID = "cerebras/SlimPajama-627B"


def _write_status(output_path: Path, status: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


def _ok(name: str, **extra) -> dict:
    return {"dataset": name, "ok": True, **extra}


def _fail(name: str, message: str, **extra) -> dict:
    return {"dataset": name, "ok": False, "error": message, **extra}


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _hf_command() -> List[str]:
    exe = shutil.which("hf")
    if exe:
        return [exe]
    return [sys.executable, "-m", "huggingface_hub.commands.hf_cli"]


def _run_command(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    display_cmd: Optional[str] = None,
    check: bool = True,
) -> Dict[str, object]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
    )
    result = {
        "cmd": display_cmd or " ".join(cmd),
        "cwd": str(cwd) if cwd else None,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }
    if check and proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or f"Command failed with exit code {proc.returncode}"
        raise RuntimeError(f"{result['cmd']} failed: {message}")
    return result


def _install_python_packages() -> List[Dict[str, object]]:
    commands = []
    shared_packages = [
        "huggingface_hub[cli]",
        "datasets",
        "pyarrow",
        "zstandard",
        "fsspec",
        "aiohttp",
        "tqdm",
        "numpy",
        "tiktoken",
    ]
    nanogpt_packages = [
        "torch",
        "numpy",
        "transformers",
        "datasets",
        "tiktoken",
        "wandb",
        "tqdm",
    ]
    commands.append(_run_command([sys.executable, "-m", "pip", "install", "--upgrade", *shared_packages]))
    commands.append(_run_command([sys.executable, "-m", "pip", "install", "--upgrade", *nanogpt_packages]))
    return commands


def _hf_login(token: str) -> List[Dict[str, object]]:
    hf = _hf_command()
    env = os.environ.copy()
    env["HF_TOKEN"] = token
    env["HUGGINGFACE_HUB_TOKEN"] = token
    commands = []
    commands.append(
        _run_command(
            [*hf, "auth", "login", "--token", token],
            env=env,
            display_cmd=f"{' '.join(hf)} auth login --token ***",
        )
    )
    commands.append(_run_command([*hf, "auth", "whoami"], env=env))
    return commands


def _download_slimpajama_subset(local_dir: Path) -> List[Dict[str, object]]:
    hf = _hf_command()
    local_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    commands.append(
        _run_command(
            [
                *hf,
                "download",
                SLIMPAJAMA_REPO_ID,
                "--repo-type",
                "dataset",
                "--dry-run",
                "--include",
                "validation/chunk1/*",
            ]
        )
    )
    commands.append(
        _run_command(
            [
                *hf,
                "download",
                SLIMPAJAMA_REPO_ID,
                "--repo-type",
                "dataset",
                "--include",
                "train/chunk1/*",
                "--include",
                "validation/chunk1/*",
                "--include",
                "test/chunk1/*",
                "--local-dir",
                str(local_dir),
            ]
        )
    )
    return commands


def _ensure_nanogpt_repo(nanogpt_root: Path) -> List[Dict[str, object]]:
    commands = []
    if nanogpt_root.exists():
        return commands
    nanogpt_root.parent.mkdir(parents=True, exist_ok=True)
    commands.append(_run_command(["git", "clone", "https://github.com/karpathy/nanoGPT.git", str(nanogpt_root)]))
    return commands


def _deploy_nanogpt_prepare_script(nanogpt_root: Path) -> Path:
    template = CODE_ROOT / "runtime_assets" / "nanogpt_slimpajama_prepare.py"
    target = nanogpt_root / "data" / "slimpajama" / "prepare.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template, target)
    return target


def _smoke_test_nanogpt_sample(nanogpt_root: Path) -> Dict[str, object]:
    result = _run_command(
        [
            sys.executable,
            "sample.py",
            "--init_from=gpt2",
            '--start=Hello',
            "--max_new_tokens=32",
        ],
        cwd=nanogpt_root,
    )
    return {
        "ok": True,
        "cmd": result["cmd"],
        "stdout_tail": "\n".join(result["stdout"].splitlines()[-10:]),
    }


def _smoke_test_nanogpt_prepare(nanogpt_root: Path, slimpajama_dir: Path) -> Dict[str, object]:
    env = os.environ.copy()
    env["SLIMPAJAMA_DIR"] = str(slimpajama_dir)
    env["LIMIT_TRAIN_DOCS"] = "32"
    env["LIMIT_VAL_DOCS"] = "8"
    prepare_path = nanogpt_root / "data" / "slimpajama" / "prepare.py"
    result = _run_command([sys.executable, str(prepare_path)], cwd=prepare_path.parent, env=env)
    return {
        "ok": True,
        "cmd": result["cmd"],
        "stdout_tail": "\n".join(result["stdout"].splitlines()[-10:]),
        "train_bin": str(prepare_path.parent / "train.bin"),
        "val_bin": str(prepare_path.parent / "val.bin"),
    }


def _build_preflight(cache_layout: dict, slimpajama_dir: Path, nanogpt_root: Path) -> dict:
    repo_status = nanogpt_repo_status(nanogpt_root)
    slimpajama_status = {
        "local_dir": str(slimpajama_dir),
        "ok": False,
    }
    try:
        details = inspect_local_slimpajama_subset(str(slimpajama_dir))
        slimpajama_status.update({"ok": True, **details})
    except Exception as exc:
        slimpajama_status["error"] = str(exc)

    return {
        "python_executable": sys.executable,
        "cwd": str(Path.cwd()),
        "cache_layout": cache_layout,
        "env": {
            "TRAINING_DATA_DIR": os.environ.get("TRAINING_DATA_DIR"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
            "TORCH_HOME": os.environ.get("TORCH_HOME"),
            "KAGGLE_CONFIG_DIR": os.environ.get("KAGGLE_CONFIG_DIR"),
            "HF_TOKEN": "set" if os.environ.get("HF_TOKEN") else "missing",
            "HUGGINGFACE_HUB_TOKEN": "set" if os.environ.get("HUGGINGFACE_HUB_TOKEN") else "missing",
            "SLIMPAJAMA_DIR": str(slimpajama_dir),
            "NANOGPT_ROOT": str(nanogpt_root),
        },
        "python_modules": {
            "torch": _module_available("torch"),
            "torchvision": _module_available("torchvision"),
            "datasets": _module_available("datasets"),
            "transformers": _module_available("transformers"),
            "huggingface_hub": _module_available("huggingface_hub"),
            "kaggle": _module_available("kaggle"),
            "tiktoken": _module_available("tiktoken"),
            "pyarrow": _module_available("pyarrow"),
            "zstandard": _module_available("zstandard"),
            "fsspec": _module_available("fsspec"),
            "aiohttp": _module_available("aiohttp"),
        },
        "slimpajama_local_subset": slimpajama_status,
        "nanogpt_repo": repo_status,
    }


def _expected_dataset_manifest(cache_layout: dict, datasets_status: list, slimpajama_dir: Path, nanogpt_root: Path) -> dict:
    data_root = Path(cache_layout["data_root"])
    datasets_map = {entry["dataset"]: entry for entry in datasets_status}
    return {
        "data_root": cache_layout["data_root"],
        "training_env": cache_layout,
        "datasets": {
            "cifar100": {
                "used_by": ["resnet18_cifar100", "vit_cifar100"],
                "storage": "torchvision",
                "root": str(data_root),
                "expected_paths": [str(data_root / "cifar-100-batches-py")],
                "status": datasets_map.get("cifar100"),
            },
            "tinyimagenet": {
                "used_by": ["efficientnet_tinyimagenet"],
                "storage": "kaggle",
                "root": str(data_root),
                "expected_paths": [
                    str(data_root / "tiny-imagenet-200"),
                    str(data_root / "tiny-imagenet-200" / "train"),
                    str(data_root / "tiny-imagenet-200" / "val"),
                ],
                "status": datasets_map.get("tinyimagenet"),
            },
            "wikitext2": {
                "used_by": ["nanogpt_wikitext"],
                "storage": "huggingface_cache",
                "root": cache_layout["hf_datasets_cache"],
                "expected_paths": [
                    cache_layout["hf_home"],
                    cache_layout["hf_datasets_cache"],
                    cache_layout["hf_hub_cache"],
                ],
                "dataset_id": "wikitext/wikitext-2-raw-v1",
                "tokenizer": "gpt2",
                "status": datasets_map.get("wikitext2"),
            },
            "squad": {
                "used_by": ["t5_squad"],
                "storage": "huggingface_cache",
                "root": cache_layout["hf_datasets_cache"],
                "expected_paths": [
                    cache_layout["hf_home"],
                    cache_layout["hf_datasets_cache"],
                    cache_layout["hf_hub_cache"],
                ],
                "dataset_id": "squad",
                "model_artifact": "t5-small",
                "status": datasets_map.get("squad"),
            },
            "slimpajama": {
                "used_by": ["nanogpt_slimpajama"],
                "storage": "local_subset_via_hf_cli",
                "root": str(slimpajama_dir),
                "expected_paths": [
                    str(slimpajama_dir / "train" / "chunk1"),
                    str(slimpajama_dir / "validation" / "chunk1"),
                    str(slimpajama_dir / "test" / "chunk1"),
                ],
                "dataset_id": SLIMPAJAMA_REPO_ID,
                "tokenizer": "gpt2",
                "nanogpt_root": str(nanogpt_root),
                "status": datasets_map.get("slimpajama"),
            },
        },
    }


def prepare_cifar() -> dict:
    from utils.datasets import load_cifar100, load_cifar100_vit

    load_cifar100(batch_size=2)
    load_cifar100_vit(batch_size=2)
    data_root = Path(get_data_dir())
    return _ok("cifar100", data_root=str(data_root), expected_paths=[str(data_root / "cifar-100-batches-py")])


def prepare_tinyimagenet() -> dict:
    try:
        from utils.datasets import load_tinyimagenet

        load_tinyimagenet(batch_size=2, n_samples=2)
        data_root = Path(get_data_dir())
        return _ok(
            "tinyimagenet",
            data_root=str(data_root),
            expected_paths=[
                str(data_root / "tiny-imagenet-200"),
                str(data_root / "tiny-imagenet-200" / "train"),
                str(data_root / "tiny-imagenet-200" / "val"),
            ],
        )
    except Exception as exc:
        return _fail("tinyimagenet", str(exc), data_root=get_data_dir())


def prepare_wikitext() -> dict:
    from utils.datasets.wikitext import load_wikitext2

    train_loader, val_loader, vocab_size = load_wikitext2(
        batch_size=2,
        block_size=64,
        max_train_samples=8,
        max_val_samples=8,
        num_workers=0,
        nanogpt_pairs=True,
    )
    return _ok(
        "wikitext2",
        train_batches=len(train_loader),
        val_batches=len(val_loader),
        vocab_size=vocab_size,
        hf_home=os.environ.get("HF_HOME"),
        hf_datasets_cache=os.environ.get("HF_DATASETS_CACHE"),
        hf_hub_cache=os.environ.get("HF_HUB_CACHE"),
    )


def prepare_squad() -> dict:
    from utils.models import load_t5
    from utils.datasets import load_squad

    model = load_t5(input_shape=None, num_classes=None)
    train_loader, val_loader = load_squad(model, batch_size=2, max_train_samples=8, max_val_samples=8)
    return _ok(
        "squad",
        train_batches=len(train_loader),
        val_batches=len(val_loader),
        hf_home=os.environ.get("HF_HOME"),
        hf_datasets_cache=os.environ.get("HF_DATASETS_CACHE"),
        hf_hub_cache=os.environ.get("HF_HUB_CACHE"),
    )


def prepare_slimpajama(local_dir: Path) -> dict:
    try:
        smoke = inspect_local_slimpajama_subset(str(local_dir))
        return _ok("slimpajama", **smoke)
    except Exception as exc:
        return _fail(
            "slimpajama",
            f"Local SlimPajama subset check failed: {exc}",
            local_dir=str(local_dir),
        )


def install_kaggle_credentials(kaggle_json: Path) -> Path:
    if not kaggle_json.is_file():
        raise FileNotFoundError(f"kaggle.json not found: {kaggle_json}")

    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")).expanduser()
    config_dir.mkdir(parents=True, exist_ok=True)
    target = config_dir / "kaggle.json"
    shutil.copy2(kaggle_json, target)
    try:
        os.chmod(target, 0o600)
    except PermissionError:
        pass
    os.environ["KAGGLE_CONFIG_DIR"] = str(config_dir)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("TRAINING_DATA_DIR", "/scratch1/hernanal/datasets"))
    parser.add_argument("--kaggle-json", default=str(CODE_ROOT / "runtime_assets" / "kaggle.json"))
    parser.add_argument("--hf-token-file", default=str(CODE_ROOT / "runtime_assets" / "hf_token.txt"))
    parser.add_argument("--slimpajama-dir", default=os.environ.get("SLIMPAJAMA_DIR", "/data/slimpajama-mini"))
    parser.add_argument("--nanogpt-root", default=os.environ.get("NANOGPT_ROOT", "/opt/nanoGPT"))
    parser.add_argument("--status-file", default=str(RESULTS_ROOT / "dataset_prepare_status.json"))
    parser.add_argument("--manifest-file", default=str(RESULTS_ROOT / "dataset_manifest.json"))
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN") and Path(args.hf_token_file).is_file():
        token = Path(args.hf_token_file).read_text(encoding="utf-8").strip()
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required. Put it in the environment or in code/runtime_assets/hf_token.txt.")

    os.environ["TRAINING_DATA_DIR"] = args.data_root
    os.environ["SLIMPAJAMA_DIR"] = args.slimpajama_dir
    os.environ["NANOGPT_ROOT"] = args.nanogpt_root

    ensure_dir(Path(args.data_root))
    ensure_dir(LOGS_ROOT)
    ensure_dir(RESULTS_ROOT)
    cache_layout = configure_cache_env(args.data_root)

    slimpajama_dir = Path(args.slimpajama_dir).expanduser()
    nanogpt_root = Path(args.nanogpt_root).expanduser()

    status = {
        "data_root": str(Path(args.data_root).resolve()),
        "kaggle_json": args.kaggle_json,
        "slimpajama_dir": str(slimpajama_dir),
        "nanogpt_root": str(nanogpt_root),
        "commands": [],
        "datasets": [],
        "smoke_tests": {},
    }

    try:
        status["commands"].extend(_install_python_packages())
    except Exception as exc:
        status["install_error"] = str(exc)

    try:
        status["commands"].extend(_hf_login(os.environ["HF_TOKEN"]))
    except Exception as exc:
        status["hf_auth_error"] = str(exc)

    try:
        status["commands"].extend(_download_slimpajama_subset(slimpajama_dir))
    except Exception as exc:
        status["slimpajama_download_error"] = str(exc)

    try:
        status["commands"].extend(_ensure_nanogpt_repo(nanogpt_root))
    except Exception as exc:
        status["nanogpt_clone_error"] = str(exc)

    try:
        prepare_script = _deploy_nanogpt_prepare_script(nanogpt_root)
        status["nanogpt_prepare_script"] = str(prepare_script)
    except Exception as exc:
        status["nanogpt_prepare_script_error"] = str(exc)

    try:
        status["smoke_tests"]["slimpajama_local_subset"] = {"ok": True, **inspect_local_slimpajama_subset(str(slimpajama_dir))}
    except Exception as exc:
        status["smoke_tests"]["slimpajama_local_subset"] = {"ok": False, "error": str(exc)}

    try:
        status["smoke_tests"]["nanogpt_repo_sample"] = _smoke_test_nanogpt_sample(nanogpt_root)
    except Exception as exc:
        status["smoke_tests"]["nanogpt_repo_sample"] = {"ok": False, "error": str(exc)}

    try:
        status["smoke_tests"]["nanogpt_prepare_slimpajama"] = _smoke_test_nanogpt_prepare(nanogpt_root, slimpajama_dir)
    except Exception as exc:
        status["smoke_tests"]["nanogpt_prepare_slimpajama"] = {"ok": False, "error": str(exc)}

    try:
        status["kaggle_config"] = str(install_kaggle_credentials(Path(args.kaggle_json)))
    except Exception as exc:
        status["kaggle_config_error"] = str(exc)

    checks = [
        prepare_cifar,
        prepare_tinyimagenet,
        prepare_wikitext,
        prepare_squad,
        lambda: prepare_slimpajama(slimpajama_dir),
    ]
    for check in checks:
        try:
            status["datasets"].append(check())
        except Exception as exc:
            name = getattr(check, "__name__", "dataset_check")
            status["datasets"].append(_fail(name, str(exc)))

    status["preflight"] = _build_preflight(cache_layout, slimpajama_dir, nanogpt_root)

    _write_status(Path(args.status_file), status)
    _write_status(Path(args.manifest_file), _expected_dataset_manifest(cache_layout, status["datasets"], slimpajama_dir, nanogpt_root))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
