"""FastAPI 入口：所有 HTTP 接口"""
import asyncio
import contextlib
import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse

from . import datasets_mgr, models_mgr, trainer
from .config import ALLOWED_SUFFIX, MAX_UPLOAD_BYTES, UPLOAD_DIR
from .schemas import CurveResponse, JobInfo, TrainRequest

app = FastAPI(
    title="云端 LoRA 训练服务",
    description="上传数据集 → 选模型 → 一键训练 → 看进度和曲线 → 下载模型",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 上线时改成具体域名
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    trainer.load_jobs_from_disk()
    await trainer.start_worker()


def _public(job: dict) -> dict:
    """去掉内部字段，不把服务器路径暴露给前端"""
    return {k: v for k, v in job.items() if not k.startswith("_")}


# ==================== 系统 ====================

@app.get("/api/system/health", tags=["系统"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/system/gpu", tags=["系统"])
def gpu_info() -> dict:
    """前端可用 recommended_preset 自动选中合适的硬件档位"""
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 - 没装/没显卡都走 CPU 分支
        return {"available": False, "devices": [], "recommended_preset": "cpu"}

    devices = []
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            util = None
            with contextlib.suppress(Exception):
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            devices.append({
                "index": i, "name": name,
                "total_mb": mem.total // 1024 // 1024,
                "used_mb": mem.used // 1024 // 1024,
                "free_mb": mem.free // 1024 // 1024,
                "util_percent": util,
            })
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()

    return {
        "available": bool(devices),
        "devices": devices,
        "recommended_preset": _recommend_preset(devices),
    }


def _recommend_preset(devices: list[dict]) -> str:
    """
    按显存 + 架构推荐档位。
    关键：GTX 16xx / RTX 20xx 是 Turing，不支持 bf16 —— 即使有 11GB 显存
    也只能推荐 gpu_6g（fp16 档位），否则训练会直接报错退出。
    """
    if not devices:
        return "cpu"
    d = devices[0]
    total_gb = d["total_mb"] / 1024
    name = d["name"].upper()
    is_turing = any(k in name for k in ("GTX 16", "RTX 20", "TITAN RTX", "QUADRO RTX"))

    if is_turing:
        return "gpu_6g"
    if total_gb >= 11.5:
        return "gpu_12g_plus"
    if total_gb >= 7.5:
        return "gpu_8g"
    if total_gb >= 5.5:
        return "gpu_6g"
    return "gpu_4g"


# ==================== 数据集 ====================

@app.get("/api/datasets", tags=["数据集"])
def api_list_datasets() -> list[dict]:
    return datasets_mgr.list_datasets()


@app.post("/api/datasets/upload", tags=["数据集"])
async def api_upload_dataset(
    file: UploadFile = File(..., description="json / jsonl / csv"),
    display_name: str | None = Form(None),
    clean_calc_markers: bool = Form(True),
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIX:
        raise HTTPException(400, f"只支持 {sorted(ALLOWED_SUFFIX)}，收到 '{suffix}'")

    stem = Path(file.filename or "dataset").stem
    dest = UPLOAD_DIR / f"{stem}_{id(file)}{suffix}"
    size = 0
    try:
        with dest.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413, f"文件超过 {MAX_UPLOAD_BYTES // 1024 // 1024} MB 上限"
                    )
                f.write(chunk)
        return datasets_mgr.convert_and_register(
            dest, display_name or stem, clean_calc_markers
        )
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except ValueError as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"数据集解析失败：{e}") from e


@app.post("/api/datasets/builtin/gsm8k", tags=["数据集"])
def api_install_gsm8k() -> dict:
    """一键安装内置 GSM8K，方便不上传就试跑"""
    try:
        return datasets_mgr.install_builtin_gsm8k()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            500, f"下载 GSM8K 失败：{e}（检查网络，或设置 HF_ENDPOINT=https://hf-mirror.com）"
        ) from e


@app.get("/api/datasets/{name}/preview", tags=["数据集"])
def api_preview_dataset(name: str, n: int = Query(5, ge=1, le=50)) -> dict:
    info = datasets_mgr.get_dataset(name)
    if not info:
        raise HTTPException(404, "数据集不存在")
    records = json.loads(Path(info["path"]).read_text(encoding="utf-8"))
    return {"name": name, "count": len(records), "samples": records[:n]}


@app.delete("/api/datasets/{name}", tags=["数据集"])
def api_delete_dataset(name: str) -> dict:
    if not datasets_mgr.delete_dataset(name):
        raise HTTPException(404, "数据集不存在")
    return {"deleted": name}


# ==================== 模型 ====================

@app.get("/api/models", tags=["模型"])
def api_list_models() -> dict:
    return {"local": models_mgr.list_models(), "recommended": models_mgr.RECOMMENDED}


@app.post("/api/models/pull", tags=["模型"])
async def api_pull_model(
    repo: str = Form(..., description="如 Qwen/Qwen2.5-0.5B-Instruct"),
    source: str = Form("modelscope", description="modelscope 或 huggingface"),
) -> dict:
    task_id = await models_mgr.pull_model(repo, source)
    return {"task_id": task_id, "message": "开始下载，请轮询 /api/models/pull/{task_id}"}


@app.get("/api/models/pull/{task_id}", tags=["模型"])
def api_pull_status(task_id: str) -> dict:
    if not (t := models_mgr.get_pull_task(task_id)):
        raise HTTPException(404, "任务不存在")
    return t


# ==================== 训练 ====================

@app.post("/api/jobs", response_model=JobInfo, tags=["训练"])
def api_create_job(req: TrainRequest) -> dict:
    """一键启动训练。最少只需传 model_name + dataset_name"""
    if not datasets_mgr.get_dataset(req.dataset_name):
        raise HTTPException(400, f"数据集 '{req.dataset_name}' 不存在")
    if req.model_name not in {m["name"] for m in models_mgr.list_models()}:
        raise HTTPException(400, f"模型 '{req.model_name}' 不存在，请先拉取")
    return _public(trainer.create_job(req))


@app.get("/api/jobs", tags=["训练"])
def api_list_jobs(status: str | None = None) -> list[dict]:
    return [_public(j) for j in trainer.list_jobs(status)]


@app.get("/api/jobs/{job_id}", response_model=JobInfo, tags=["训练"])
def api_get_job(job_id: str) -> dict:
    if not (job := trainer.get_job(job_id)):
        raise HTTPException(404, "任务不存在")
    return _public(job)


@app.get("/api/jobs/{job_id}/stream", tags=["训练"])
async def api_stream(job_id: str) -> StreamingResponse:
    """SSE 实时推送进度。前端用 EventSource 订阅"""
    if not trainer.get_job(job_id):
        raise HTTPException(404, "任务不存在")

    async def gen():
        last_step = -1
        while True:
            job = trainer.get_job(job_id)
            if not job:
                yield 'event: error\ndata: {"message": "任务已删除"}\n\n'
                return

            payload = {
                "status": job["status"],
                "progress": job["progress"],
                "error_message": job.get("error_message"),
                "error_hint": job.get("error_hint"),
            }
            yield f"event: progress\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

            # 有新 loss 点就一并推过去，前端直接 append 到曲线
            step = job["progress"]["current_steps"]
            if step > last_step and job["progress"].get("loss") is not None:
                last_step = step
                point = {
                    "step": step,
                    "epoch": job["progress"]["epoch"],
                    "loss": job["progress"]["loss"],
                    "learning_rate": job["progress"]["learning_rate"],
                }
                yield f"event: point\ndata: {json.dumps(point)}\n\n"

            if job["status"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
                done = {"status": job["status"], "has_adapter": job.get("has_adapter", False)}
                yield f"event: done\ndata: {json.dumps(done)}\n\n"
                return
            await asyncio.sleep(2)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        # X-Accel-Buffering: no 是给 nginx 的，否则 nginx 缓冲 SSE，
        # 前端表现为进度一直 0%、训练结束才一次性刷出全部事件
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/jobs/{job_id}/curve", response_model=CurveResponse, tags=["训练"])
def api_curve(job_id: str) -> dict:
    """全量曲线数据。前端刷新页面后用这个恢复已有曲线"""
    if not trainer.get_job(job_id):
        raise HTTPException(404, "任务不存在")
    return trainer.get_curve(job_id)


@app.get("/api/jobs/{job_id}/logs", response_class=PlainTextResponse, tags=["训练"])
def api_logs(job_id: str, tail: int = Query(200, ge=1, le=5000)) -> str:
    if not trainer.get_job(job_id):
        raise HTTPException(404, "任务不存在")
    return trainer.get_logs(job_id, tail)


@app.post("/api/jobs/{job_id}/cancel", tags=["训练"])
async def api_cancel(job_id: str) -> dict:
    if not await trainer.cancel_job(job_id):
        raise HTTPException(400, "任务不存在或当前状态不可取消")
    return {"job_id": job_id, "status": "CANCELLED"}


@app.delete("/api/jobs/{job_id}", tags=["训练"])
def api_delete_job(job_id: str) -> dict:
    if not (job := trainer.get_job(job_id)):
        raise HTTPException(404, "任务不存在")
    if job["status"] == "RUNNING":
        raise HTTPException(400, "任务正在运行，请先取消")
    trainer.delete_job(job_id)
    return {"deleted": job_id}


# ==================== 产物 ====================

@app.get("/api/jobs/{job_id}/artifacts", tags=["产物"])
def api_artifacts(job_id: str) -> dict:
    if not trainer.get_job(job_id):
        raise HTTPException(404, "任务不存在")
    return {"job_id": job_id, "files": trainer.list_artifacts(job_id)}


@app.get("/api/jobs/{job_id}/loss_plot", tags=["产物"])
def api_loss_plot(job_id: str) -> FileResponse:
    """LLaMA-Factory 自动生成的 loss 曲线图，可直接 <img src>"""
    if not (job := trainer.get_job(job_id)):
        raise HTTPException(404, "任务不存在")
    p = Path(job["_path"]) / "training_loss.png"
    if not p.is_file():
        raise HTTPException(404, "曲线图尚未生成（训练结束后才会生成）")
    return FileResponse(p, media_type="image/png")


@app.get("/api/jobs/{job_id}/download", tags=["产物"])
def api_download(job_id: str) -> FileResponse:
    """下载训练好的 LoRA 适配器（zip，约 17MB）"""
    if not trainer.get_job(job_id):
        raise HTTPException(404, "任务不存在")
    zp = trainer.pack_adapter(job_id)
    if not zp:
        raise HTTPException(404, "没有 LoRA 产物（训练可能未成功完成）")
    return FileResponse(zp, media_type="application/zip", filename=zp.name)
