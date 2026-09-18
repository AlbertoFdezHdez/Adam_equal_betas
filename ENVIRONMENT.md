# Environment provenance

The exact frozen environment for every original server run was not preserved. For that reason, `requirements.txt` and `requirements-training.txt` list required packages without inventing exact pins.

Available historical setup records mention Python 3.9 and, at one point in environment preparation, the following versions:

- PyTorch 2.6.0
- NumPy 2.0.2
- Transformers 4.57.6
- Datasets 4.5.0
- tiktoken 0.12.0
- tqdm 4.67.3
- pandas 2.3.3

This evidence is not sufficient to assert that every recorded run used exactly that combination. In particular, the environment log included attempted package changes. These versions are therefore provenance notes, not guaranteed pins.

Original full runs were performed on x86-64 Linux with NVIDIA A100-SXM4 80 GB GPUs. The training scripts require CUDA-capable PyTorch for practical runtimes, although lightweight analysis and the Digits experiment can run on CPU. Five full-training entry points explicitly request highest float32 matrix-multiplication precision and disable cuDNN TF32; the ResNet18 entry point does not make those calls. Mixed precision is disabled in all six documented experiments.

For a new reproducibility run, record at least:

```bash
python --version
python -m pip freeze > outputs/environment-pip-freeze.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Also record the Git revision of the external nanoGPT checkout. The original revision is unavailable.
