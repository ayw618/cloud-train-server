"""训练引擎：配置生成、队列调度、子进程管理、进度解析、状态持久化"""
import asyncio
import json
import os
import shutil
import sys
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

import yaml

from .config import HARDWARE_PRESETS, JOB_DIR, LF_DIR, MODEL_DIR
from .errors import diagnose
from .schemas import JobStatus, TrainRequest

# ---------- 全局状态 ----------
_jobs: dict[str, dict] = {}
_procs: dict[str, asyncio.subprocess.Process] = {}
_queue: asyncio.Queue[str] = asyncio.Queue()
_worker_started = False

ADAPTER_FILES = [
    "adapter_config.json", "adapter_model.safetensors",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt",
    "train_results.json", "trainer_state.json", "training_loss.png",
]


# ==================== 配置生成 ====================

def build_config(req: TrainRequest, job_path: Path) -> dict:
    """把前端的简单请求翻译成完整的 LLaMA-Factory YAML 配置"""
    preset = HARDWARE_PRESETS[req.hardware_preset].copy()
    precision = preset.pop("precision")

    cfg = {
        # 模型
        "model_name_or_path": str((MODEL_DIR / req.model_name).resolve()),
        "trust_remote_code": True,
        # 方法
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": req.lora_rank,
        "lora_alpha": req.lora_alpha,
        "lora_dropout": 0.05,
        "lora_target": "all",
        # 数据
        "dataset": req.dataset_name,
        "dataset_dir": str((LF_DIR / "data").resolve()),
        "template": req.template,
        "max_samples": req.max_samples,
        "overwrite_cache": True,
        "preprocessing_num_workers": 4,
        # 输出
        "output_dir": str(job_path.resolve()),
        "logging_steps": 5,        # 密一点，前端曲线更平滑
        "save_steps": 500,
        "save_total_limit": 2,     # 只留 2 个 checkpoint，省磁盘
        "plot_loss": True,
        "overwrite_output_dir": True,
        "report_to": "none",       # 关掉 wandb，避免卡在登录
        # 超参
        "learning_rate": req.learning_rate,
        "num_train_epochs": req.num_train_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        **preset,                  # batch / accum / cutoff_len / grad_ckpt
    }

    # 精度三者互斥，只开一个；fp32 什么都不加
    if precision == "fp16":
        cfg["fp16"] = True
    elif precision == "bf16":
        cfg["bf16"] = True

    if req.hardware_preset == "cpu":
        cfg["use_cpu"] = True

    return cfg


# ==================== 任务创建与持久化 ====================

def create_job(req: TrainRequest) -> dict:
    ts = datetime.now()
    job_id = f"job_{ts:%Y%m%d_%H%M%S}_{os.urandom(2).hex()}"
    job_path = JOB_DIR / job_id
    job_path.mkdir(parents=True, exist_ok=True)

    cfg = build_config(req, job_path)
    (job_path / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    job = {
        "job_id": job_id,
        "job_name": req.job_name or f"{req.model_name} × {req.dataset_name}",
        "status": JobStatus.PENDING.value,
        "model_name": req.model_name,
        "dataset_name": req.dataset_name,
        "hardware_preset": req.hardware_preset,
        "created_at": ts.isoformat(timespec="seconds"),
        "started_at": None,
        "finished_at": None,
        "queue_position": None,
        "progress": _empty_progress(),
        "error_message": None,
        "error_hint": None,
        "error_raw": None,
        "has_adapter": False,
        "_path": str(job_path),
    }
    _jobs[job_id] = job
    _persist(job)
    _queue.put_nowait(job_id)
    return job


def _empty_progress() -> dict:
    return {
        "current_steps": 0, "total_steps": 0, "percentage": 0.0, "epoch": 0.0,
        "loss": None, "learning_rate": None, "elapsed_time": "", "remaining_time": "",
    }


def _persist(job: dict) -> None:
    """状态落盘，服务重启后可恢复"""
    try:
        (Path(job["_path"]) / "meta.json").write_text(
            json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass  # 落盘失败不影响内存态运行


def load_jobs_from_disk() -> None:
    """启动时恢复历史任务；中断的 RUNNING/PENDING 标为 FAILED"""
    for d in sorted(JOB_DIR.iterdir()):
        meta = d / "meta.json"
        if not meta.is_file():
            continue
        try:
            job = json.loads(meta.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if job.get("status") in (JobStatus.RUNNING.value, JobStatus.PENDING.value):
            job["status"] = JobStatus.FAILED.value
            job["error_message"] = "服务重启导致训练中断"
            job["error_hint"] = "请重新提交该任务。"
            _persist(job)
        job["_path"] = str(d)  # 目录可能被移动过，以实际路径为准
        job["has_adapter"] = (d / "adapter_model.safetensors").is_file()
        _jobs[job["job_id"]] = job


# ==================== 队列 Worker ====================

async def start_worker() -> None:
    """后台常驻协程，串行取任务执行。在 FastAPI startup 里调一次"""
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    asyncio.create_task(_worker_loop())


async def _worker_loop() -> None:
    while True:
        job_id = await _queue.get()
        try:
            job = _jobs.get(job_id)
            if job and job["status"] == JobStatus.PENDING.value:
                await _run_training(job)
        except Exception as e:  # noqa: BLE001 - worker 必须活下去
            # 完整堆栈必须打到服务端控制台并落盘：前端只提示“请检查服务端日志”，
            # 若这里只存 str(e)，遇到无消息异常（如 Windows SelectorEventLoop 不支持
            # 子进程时抛的 NotImplementedError），用户在任何日志里都查不到原因。
            tb = traceback.format_exc()
            print(f"[trainer] {job_id} 训练启动失败：\n{tb}", flush=True)
            if job := _jobs.get(job_id):
                reason = str(e) or type(e).__name__  # str(e) 可能为空，用类型名兜底
                job.update(
                    status=JobStatus.FAILED.value,
                    error_message=f"引擎内部错误：{reason}",
                    error_hint="请检查服务端日志。",
                    error_raw=tb[-2000:],
                    finished_at=datetime.now().isoformat(timespec="seconds"),
                )
                _persist(job)
        finally:
            _queue.task_done()


# ==================== 核心：跑训练 + 解析进度 ====================

def _train_cmd(cfg_path: Path) -> list[str]:
    """优先用已注册的 llamafactory-cli，回退到 python -m"""
    if exe := shutil.which("llamafactory-cli"):
        return [exe, "train", str(cfg_path)]
    return [sys.executable, "-m", "llamafactory.cli", "train", str(cfg_path)]


async def _run_training(job: dict) -> None:
    job_path = Path(job["_path"])
    cfg_path = job_path / "config.yaml"
    log_path = job_path / "stdout.log"
    jsonl_path = job_path / "trainer_log.jsonl"

    job.update(status=JobStatus.RUNNING.value,
               started_at=datetime.now().isoformat(timespec="seconds"))
    _persist(job)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # 关键：不缓冲，日志实时可读
    env["WANDB_DISABLED"] = "true"
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    poller = None
    with log_path.open("w", encoding="utf-8") as logf:
        proc = await asyncio.create_subprocess_exec(
            *_train_cmd(cfg_path),
            cwd=str(LF_DIR),
            stdout=logf,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        _procs[job["job_id"]] = proc
        poller = asyncio.create_task(_poll_loop(job, jsonl_path))
        rc = await proc.wait()

    poller.cancel()
    _procs.pop(job["job_id"], None)
    _poll_once(job, jsonl_path)  # 收尾再读一次，拿到最终进度

    job["finished_at"] = datetime.now().isoformat(timespec="seconds")

    if job["status"] == JobStatus.CANCELLED.value:
        pass  # 已被 cancel 标记，不覆盖
    elif rc == 0:
        job["status"] = JobStatus.SUCCEEDED.value
        job["progress"]["percentage"] = 100.0
        job["has_adapter"] = (job_path / "adapter_model.safetensors").is_file()
    else:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        msg, hint, raw = diagnose(text)
        job.update(status=JobStatus.FAILED.value,
                   error_message=msg, error_hint=hint, error_raw=raw)
    _persist(job)


async def _poll_loop(job: dict, jsonl_path: Path) -> None:
    while True:
        _poll_once(job, jsonl_path)
        await asyncio.sleep(2)


def _poll_once(job: dict, jsonl_path: Path) -> None:
    """
    读 trainer_log.jsonl 最后一行更新进度。
    文件不存在（训练刚启动）或读到写入一半的行都是正常的 —— 静默跳过，
    下个周期再读。若让异常逃逸，轮询协程会死掉，前端进度永远卡 0%。
    """
    if not jsonl_path.is_file():
        return
    try:
        lines = [x for x in jsonl_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not lines:
            return
        last = json.loads(lines[-1])
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return

    job["progress"].update({
        "current_steps": last.get("current_steps", 0),
        "total_steps": last.get("total_steps", 0),
        "percentage": round(float(last.get("percentage", 0.0)), 2),
        "epoch": last.get("epoch", 0.0),
        "loss": last.get("loss"),
        "learning_rate": last.get("learning_rate"),
        "elapsed_time": last.get("elapsed_time", ""),
        "remaining_time": last.get("remaining_time", ""),
    })
    _persist(job)


# ==================== 查询 / 曲线 / 日志 / 取消 ====================

def list_jobs(status: str | None = None) -> list[dict]:
    jobs = sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)
    # 给排队中的任务算队列位置
    pending = sorted(
        (j for j in jobs if j["status"] == JobStatus.PENDING.value),
        key=lambda j: j["created_at"],
    )
    for i, j in enumerate(pending, 1):
        j["queue_position"] = i
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    return jobs


def get_job(job_id: str) -> dict | None:
    return _jobs.get(job_id)


def get_curve(job_id: str) -> dict:
    """读全量 trainer_log.jsonl，返回曲线点数组"""
    job = _jobs.get(job_id)
    points, total = [], 0
    if job:
        p = Path(job["_path"]) / "trainer_log.jsonl"
        if p.is_file():
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "loss" not in d:  # 末尾汇总行没有 loss
                    continue
                total = d.get("total_steps", total)
                points.append({
                    "step": d.get("current_steps", 0),
                    "epoch": round(d.get("epoch", 0.0), 4),
                    "loss": d["loss"],
                    "learning_rate": d.get("learning_rate", 0.0),
                })
    return {"job_id": job_id, "total_steps": total, "points": points}


def get_logs(job_id: str, tail: int = 200) -> str:
    job = _jobs.get(job_id)
    if not job:
        return ""
    p = Path(job["_path"]) / "stdout.log"
    if not p.is_file():
        return "(暂无日志，训练可能还在启动中)"
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-tail:])


async def cancel_job(job_id: str) -> bool:
    job = _jobs.get(job_id)
    if not job:
        return False

    if job["status"] == JobStatus.PENDING.value:
        job.update(status=JobStatus.CANCELLED.value,
                   finished_at=datetime.now().isoformat(timespec="seconds"))
        _persist(job)
        return True

    if job["status"] != JobStatus.RUNNING.value:
        return False

    job["status"] = JobStatus.CANCELLED.value  # 先标记，_run_training 收尾时不覆盖
    _persist(job)

    proc = _procs.get(job_id)
    if proc and proc.returncode is None:
        try:
            proc.terminate()  # 先温柔地要求退出
            try:
                await asyncio.wait_for(proc.wait(), timeout=15)
            except asyncio.TimeoutError:
                proc.kill()   # 15 秒不退就强杀
        except ProcessLookupError:
            pass
    return True


def delete_job(job_id: str) -> bool:
    job = _jobs.pop(job_id, None)
    if not job:
        return False
    shutil.rmtree(job["_path"], ignore_errors=True)
    return True


# ==================== 产物 ====================

def list_artifacts(job_id: str) -> list[dict]:
    job = _jobs.get(job_id)
    if not job:
        return []
    root = Path(job["_path"])
    return [
        {"name": name, "size_mb": round((root / name).stat().st_size / 1024 / 1024, 3)}
        for name in ADAPTER_FILES
        if (root / name).is_file()
    ]


def pack_adapter(job_id: str) -> Path | None:
    """把 LoRA 适配器打成 zip（约 17MB）"""
    job = _jobs.get(job_id)
    if not job:
        return None
    root = Path(job["_path"])
    if not (root / "adapter_model.safetensors").is_file():
        return None

    zip_path = root / f"{job_id}_adapter.zip"
    if zip_path.is_file():
        return zip_path  # 已打包过，直接复用

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ADAPTER_FILES:
            if (f := root / name).is_file():
                zf.write(f, arcname=name)
        if (cfg := root / "config.yaml").is_file():
            zf.write(cfg, arcname="train_config.yaml")  # 附上配置，方便复现
    return zip_path
