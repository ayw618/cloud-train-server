"""Pydantic 模型 —— 同时也是给前端的接口契约"""
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TrainRequest(BaseModel):
    """一键启动训练的请求体：最少只需 model_name + dataset_name"""

    model_name: str = Field(..., description="基座模型目录名，来自 GET /api/models")
    dataset_name: str = Field(..., description="数据集名，来自 GET /api/datasets")
    hardware_preset: Literal["gpu_4g", "gpu_6g", "gpu_8g", "gpu_12g_plus", "cpu"] = "gpu_8g"

    lora_rank: int = Field(8, ge=1, le=256)
    lora_alpha: int = Field(16, ge=1, le=512)
    learning_rate: float = Field(1e-4, gt=0, le=1e-2)
    num_train_epochs: float = Field(3.0, gt=0, le=100)
    template: str = Field("qwen", description="对话模板：Qwen2.5→qwen，Qwen3→qwen3")
    max_samples: int = Field(100000, ge=1, description="最多用多少条样本，CPU 建议 1000")
    job_name: str | None = None


class JobProgress(BaseModel):
    current_steps: int = 0
    total_steps: int = 0
    percentage: float = 0.0
    epoch: float = 0.0
    loss: float | None = None
    learning_rate: float | None = None
    elapsed_time: str = ""
    remaining_time: str = ""


class JobInfo(BaseModel):
    job_id: str
    job_name: str
    status: JobStatus
    model_name: str
    dataset_name: str
    hardware_preset: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    queue_position: int | None = None
    progress: JobProgress = JobProgress()
    error_message: str | None = None  # 友好化中文提示
    error_hint: str | None = None     # 修复建议
    error_raw: str | None = None      # 原始 traceback 尾部
    has_adapter: bool = False


class CurvePoint(BaseModel):
    step: int
    epoch: float
    loss: float
    learning_rate: float


class CurveResponse(BaseModel):
    job_id: str
    total_steps: int
    points: list[CurvePoint]
