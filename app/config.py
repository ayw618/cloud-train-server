"""全局配置 + 硬件档位预设"""
import os
from pathlib import Path

# ---------- 路径 ----------
BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE = Path(os.getenv("WORKSPACE_DIR", BASE_DIR / "workspace")).resolve()
LF_DIR = Path(os.getenv("LLAMAFACTORY_DIR", BASE_DIR / "LLaMA-Factory")).resolve()

UPLOAD_DIR = WORKSPACE / "uploads"
DATASET_DIR = WORKSPACE / "datasets"
MODEL_DIR = WORKSPACE / "models"
JOB_DIR = WORKSPACE / "jobs"
REGISTRY = DATASET_DIR / "registry.json"

for _d in (UPLOAD_DIR, DATASET_DIR, MODEL_DIR, JOB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------- 限制 ----------
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024
ALLOWED_SUFFIX = {".json", ".jsonl", ".csv"}

# ---------- 硬件档位预设 ----------
# 前端只需传一个档位名，后端自动填这一整套显存安全参数，
# 用户不需要知道 gradient_checkpointing 是什么。
HARDWARE_PRESETS: dict[str, dict] = {
    "gpu_4g": {  # RTX 3050 笔记本版等 4GB 卡
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "cutoff_len": 512,
        "gradient_checkpointing": True,
        "precision": "bf16",
    },
    "gpu_6g": {  # GTX 1660 / RTX 2060（Turing 架构，不支持 bf16）
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "cutoff_len": 512,
        "gradient_checkpointing": True,
        "precision": "fp16",
    },
    "gpu_8g": {  # RTX 3050 桌面版 / 4060
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "cutoff_len": 768,
        "gradient_checkpointing": True,
        "precision": "bf16",
    },
    "gpu_12g_plus": {  # RTX 3060 12G 及以上
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "cutoff_len": 1024,
        "gradient_checkpointing": False,
        "precision": "bf16",
    },
    "cpu": {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "cutoff_len": 384,
        "gradient_checkpointing": False,
        "precision": "fp32",  # CPU 不支持 fp16 训练
    },
}
