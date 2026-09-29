"""基座模型：列表 + 异步拉取"""
import asyncio
import json
import shutil
import uuid
from pathlib import Path

from .config import MODEL_DIR

RECOMMENDED = [
    {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "size_gb": 1.0, "template": "qwen",
     "note": "最小可用，4GB 显存也能训"},
    {"repo": "Qwen/Qwen2.5-1.5B-Instruct", "size_gb": 3.1, "template": "qwen",
     "note": "效果更好，建议 8GB 以上显存"},
    {"repo": "Qwen/Qwen3-0.6B", "size_gb": 1.2, "template": "qwen3",
     "note": "支持思考模式，注意 cutoff_len 要加大"},
]

_pull_tasks: dict[str, dict] = {}  # 内存态，重启丢失，可接受


def _is_model_dir(p: Path) -> bool:
    """有 config.json 且有权重文件才算有效模型"""
    if not (p / "config.json").is_file():
        return False
    return any(p.glob("*.safetensors")) or any(p.glob("*.bin"))


def list_models() -> list[dict]:
    out = []
    for p in sorted(MODEL_DIR.iterdir()):
        if not p.is_dir() or not _is_model_dir(p):
            continue
        try:
            cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cfg = {}
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        out.append({
            "name": p.name,
            "size_mb": round(size / 1024 / 1024, 1),
            "architecture": (cfg.get("architectures") or ["unknown"])[0],
            "hidden_size": cfg.get("hidden_size"),
            "num_layers": cfg.get("num_hidden_layers"),
        })
    return out


async def pull_model(repo: str, source: str = "modelscope") -> str:
    """异步拉取模型，返回 task_id 供前端轮询"""
    task_id = uuid.uuid4().hex[:12]
    local_dir = MODEL_DIR / repo.split("/")[-1]
    _pull_tasks[task_id] = {
        "task_id": task_id, "repo": repo, "status": "RUNNING",
        "message": "", "log_tail": "",
    }

    if source == "modelscope":
        cmd = ["modelscope", "download", "--model", repo, "--local_dir", str(local_dir)]
    else:
        cmd = ["huggingface-cli", "download", repo, "--local-dir", str(local_dir)]

    async def _run():
        task = _pull_tasks[task_id]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )
            tail: list[str] = []
            assert proc.stdout
            async for raw in proc.stdout:
                tail.append(raw.decode("utf-8", "ignore").rstrip())
                del tail[:-20]  # 只留最后 20 行
                task["log_tail"] = "\n".join(tail)
            rc = await proc.wait()
            if rc == 0 and _is_model_dir(local_dir):
                task.update(status="SUCCEEDED", message="下载完成")
            else:
                task.update(
                    status="FAILED",
                    message=f"下载失败（退出码 {rc}）。检查网络，"
                            "或设置 HF_ENDPOINT=https://hf-mirror.com",
                )
        except FileNotFoundError:
            task.update(status="FAILED",
                        message="找不到下载命令。请先 pip install modelscope 或 huggingface_hub")
        except Exception as e:  # noqa: BLE001 - 后台任务不能让异常逃逸
            task.update(status="FAILED", message=str(e))

    asyncio.create_task(_run())
    return task_id


def get_pull_task(task_id: str) -> dict | None:
    return _pull_tasks.get(task_id)


def delete_model(name: str) -> bool:
    p = MODEL_DIR / name
    if not p.is_dir():
        return False
    shutil.rmtree(p, ignore_errors=True)
    return True
