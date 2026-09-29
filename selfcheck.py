#!/usr/bin/env python3
"""
自检脚本 —— 不需要 GPU、不需要真的 LLaMA-Factory，也不需要下载模型。

它用一个假的训练脚本冒充 llamafactory-cli，把整条链路跑通：
  上传数据集 → 注册 → 建任务 → 生成 YAML → 起子进程 → 解析进度 → 打包下载

用法：
    python selfcheck.py

全部通过会打印 "ALL CHECKS PASSED"。
"""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# ---------- 把工作区指向临时目录，不污染真实 workspace ----------
TMP = Path(tempfile.mkdtemp(prefix="cts_selfcheck_"))
FAKE_LF = TMP / "LLaMA-Factory"
(FAKE_LF / "data").mkdir(parents=True)
(FAKE_LF / "data" / "dataset_info.json").write_text("{}", encoding="utf-8")

os.environ["WORKSPACE_DIR"] = str(TMP / "workspace")
os.environ["LLAMAFACTORY_DIR"] = str(FAKE_LF)

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  — {detail}" if detail and not cond else ""))


# ================= 1. 导入 =================
print("\n[1] 模块导入")
try:
    from fastapi.testclient import TestClient

    from app import datasets_mgr, models_mgr, trainer
    from app.config import HARDWARE_PRESETS, MODEL_DIR
    from app.errors import diagnose
    from app.main import app, _recommend_preset
    from app.schemas import TrainRequest
    check("导入 app.* 全部模块", True)
except Exception as e:  # noqa: BLE001
    print(f"  ✗ 导入失败：{e}")
    print("\n请先安装依赖： pip install -r requirements.txt")
    sys.exit(1)


# ================= 2. 纯函数 =================
print("\n[2] 纯函数逻辑")

# 2.1 硬件档位互斥：精度只能开一个
for preset_name in HARDWARE_PRESETS:
    req = TrainRequest(model_name="m", dataset_name="d", hardware_preset=preset_name)
    cfg = trainer.build_config(req, TMP / "fakejob")
    both = cfg.get("fp16") and cfg.get("bf16")
    check(f"档位 {preset_name} 精度不冲突", not both, f"fp16={cfg.get('fp16')} bf16={cfg.get('bf16')}")

# 2.2 CPU 档位必须 use_cpu 且不开半精度
cpu_cfg = trainer.build_config(
    TrainRequest(model_name="m", dataset_name="d", hardware_preset="cpu"), TMP / "fakejob"
)
check("cpu 档位 use_cpu=True", cpu_cfg.get("use_cpu") is True)
check("cpu 档位不开 fp16/bf16", not cpu_cfg.get("fp16") and not cpu_cfg.get("bf16"))

# 2.3 Turing 卡即使显存大也只推荐 fp16 档位
check("GTX 1660 → gpu_6g",
      _recommend_preset([{"name": "NVIDIA GeForce GTX 1660", "total_mb": 6144}]) == "gpu_6g")
check("RTX 2080Ti(11G) → gpu_6g 而非 12g_plus",
      _recommend_preset([{"name": "NVIDIA GeForce RTX 2080 Ti", "total_mb": 11264}]) == "gpu_6g")
check("RTX 3050(8G) → gpu_8g",
      _recommend_preset([{"name": "NVIDIA GeForce RTX 3050", "total_mb": 8192}]) == "gpu_8g")
check("RTX 3050(4G) → gpu_4g",
      _recommend_preset([{"name": "NVIDIA GeForce RTX 3050 Laptop GPU", "total_mb": 4096}]) == "gpu_4g")
check("无显卡 → cpu", _recommend_preset([]) == "cpu")

# 2.4 错误诊断
msg, hint, _ = diagnose("torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 512MB")
check("OOM 被识别", "显存不足" in msg)
msg2, _, _ = diagnose("RuntimeError: Your setup doesn't support bf16")
check("bf16 不支持被识别", "bf16" in msg2)
msg3, _, raw3 = diagnose("Traceback (most recent call last):\n  ...\nValueError: weird thing")
check("未知错误回退到 traceback", "ValueError" in msg3 and "Traceback" in raw3)

# 2.5 GSM8K 计算器标记清理
cleaned = datasets_mgr._clean_markers("48/2 = <<48/2=24>>24 clips\n#### 72")
check("清理 <<>> 保留 ####", "<<" not in cleaned and "#### 72" in cleaned, cleaned)


# ================= 3. 数据集上传（三种格式 + 字段别名） =================
print("\n[3] 数据集解析")

# 必须用 with 进入上下文，否则 FastAPI 的 startup 事件不会触发，
# 队列 worker 不启动，任务会永远停在 PENDING。
_ctx = TestClient(app)
client = _ctx.__enter__()

cases = {
    "json 数组 + instruction/output": (
        "a.json",
        json.dumps([{"instruction": "1+1?", "output": "2\n#### 2"}]).encode(),
    ),
    "json 包一层 {data:[...]}": (
        "b.json",
        json.dumps({"data": [{"question": "2+2?", "answer": "4\n#### 4"}]}).encode(),
    ),
    "jsonl + prompt/response 别名": (
        "c.jsonl",
        b'{"prompt": "3+3?", "response": "6\\n#### 6"}\n{"prompt": "4+4?", "response": "8"}\n',
    ),
    "csv + 中文字段名": (
        "d.csv",
        "问题,答案\n5+5?,\"10\n#### 10\"\n".encode(),
    ),
    "jsonl 含 <<>> 标记": (
        "e.jsonl",
        b'{"question": "6+6?", "answer": "6+6 = <<6+6=12>>12\\n#### 12"}\n',
    ),
}
uploaded = []
for name, (fn, content) in cases.items():
    r = client.post("/api/datasets/upload", files={"file": (fn, content)})
    ok = r.status_code == 200 and r.json()["count"] > 0
    check(f"上传 {name}", ok, f"HTTP {r.status_code} {r.text[:120]}")
    if ok:
        uploaded.append(r.json())

if uploaded:
    last = uploaded[-1]
    out = last["preview"][0]["output"]
    check("上传后 <<>> 已清理", "<<" not in out, out)

# 坏输入必须给可读的中文错误
bad = client.post("/api/datasets/upload", files={"file": ("x.json", b'{"foo":"bar"}')})
check("坏 JSON 返回 400 + 中文提示",
      bad.status_code == 400 and "解析失败" in bad.json().get("detail", ""),
      f"HTTP {bad.status_code} {bad.text[:120]}")

bad2 = client.post("/api/datasets/upload", files={"file": ("x.txt", b"hello")})
check("不支持的后缀返回 400", bad2.status_code == 400)

# 注册是否真的写进了 LLaMA-Factory
lf_info = json.loads((FAKE_LF / "data" / "dataset_info.json").read_text(encoding="utf-8"))
check("已注册到 dataset_info.json", len(lf_info) == len(uploaded), f"{len(lf_info)} 条")
if uploaded:
    entry = lf_info[uploaded[0]["name"]]
    check("注册项含 columns 映射",
          entry["columns"]["prompt"] == "instruction" and Path(entry["file_name"]).is_absolute())


# ================= 4. 系统 / 列表接口 =================
print("\n[4] 基础接口")
check("GET /api/system/health", client.get("/api/system/health").json()["status"] == "ok")
gpu = client.get("/api/system/gpu").json()
check("GET /api/system/gpu 返回 recommended_preset",
      gpu["recommended_preset"] in HARDWARE_PRESETS)
check("GET /api/datasets", len(client.get("/api/datasets").json()) == len(uploaded))
check("GET /api/models 含 recommended", "recommended" in client.get("/api/models").json())
check("预览不存在的数据集 → 404", client.get("/api/datasets/nope/preview").status_code == 404)
check("查不存在的任务 → 404", client.get("/api/jobs/nope").status_code == 404)


# ================= 5. 参数校验 =================
print("\n[5] 请求校验")
r = client.post("/api/jobs", json={"model_name": "x", "dataset_name": "y"})
check("模型/数据集不存在 → 400", r.status_code == 400, f"HTTP {r.status_code}")
r = client.post("/api/jobs", json={
    "model_name": "x", "dataset_name": "y", "hardware_preset": "gpu_99g"})
check("非法档位 → 422", r.status_code == 422)
r = client.post("/api/jobs", json={
    "model_name": "x", "dataset_name": "y", "learning_rate": -1})
check("负学习率 → 422", r.status_code == 422)


# ================= 6. 端到端：假训练进程跑通全链路 =================
print("\n[6] 端到端训练链路（用假 llamafactory 冒充）")

# 6.1 造一个假模型目录
fake_model = MODEL_DIR / "fake-0.5B"
fake_model.mkdir(parents=True, exist_ok=True)
(fake_model / "config.json").write_text(
    json.dumps({"architectures": ["Qwen2ForCausalLM"], "hidden_size": 896,
                "num_hidden_layers": 24}), encoding="utf-8")
(fake_model / "model.safetensors").write_bytes(b"\0" * 2048)
check("假模型被识别", "fake-0.5B" in [m["name"] for m in models_mgr.list_models()])

# 6.2 造一个假训练脚本：读 YAML，往 output_dir 写 trainer_log.jsonl 和产物
fake_cli = TMP / "fake_train.py"
fake_cli.write_text('''
import json, sys, time, yaml
from pathlib import Path

cfg = yaml.safe_load(Path(sys.argv[-1]).read_text(encoding="utf-8"))
out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
print("[fake] loading model...", flush=True)

total = 20
with (out / "trainer_log.jsonl").open("w", encoding="utf-8") as f:
    for step in range(1, total + 1):
        rec = {"current_steps": step, "total_steps": total,
               "loss": round(2.0 - step * 0.05, 4), "learning_rate": 1e-4,
               "epoch": round(step / total * 3, 4),
               "percentage": round(step / total * 100, 2),
               "elapsed_time": "0:00:0%d" % step, "remaining_time": "0:00:0%d" % (total - step)}
        f.write(json.dumps(rec) + "\\n"); f.flush()
        print("[fake] step %d/%d loss=%s" % (step, total, rec["loss"]), flush=True)
        time.sleep(0.05)

(out / "adapter_model.safetensors").write_bytes(b"\\0" * 4096)
(out / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
(out / "training_loss.png").write_bytes(b"\\x89PNG\\r\\n\\x1a\\n")
print("[fake] done", flush=True)
''', encoding="utf-8")

_orig_cmd = trainer._train_cmd
trainer._train_cmd = lambda cfg_path: [sys.executable, str(fake_cli), str(cfg_path)]

ds_name = uploaded[0]["name"]
r = client.post("/api/jobs", json={
    "model_name": "fake-0.5B", "dataset_name": ds_name,
    "hardware_preset": "cpu", "job_name": "selfcheck",
})
check("POST /api/jobs 创建成功", r.status_code == 200, f"HTTP {r.status_code} {r.text[:200]}")

if r.status_code == 200:
    job_id = r.json()["job_id"]
    check("初始状态 PENDING", r.json()["status"] == "PENDING")

    # 生成的 YAML 是否合法
    import yaml as _yaml
    cfg_file = Path(os.environ["WORKSPACE_DIR"]) / "jobs" / job_id / "config.yaml"
    check("config.yaml 已生成", cfg_file.is_file())
    if cfg_file.is_file():
        cfg = _yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
        check("YAML 含必要字段",
              all(k in cfg for k in ("model_name_or_path", "dataset", "output_dir",
                                     "finetuning_type", "lora_rank")))
        check("YAML dataset 名正确", cfg["dataset"] == ds_name)

    # 等训练结束（假脚本约 1-2 秒）
    final = None
    for _ in range(120):
        final = client.get(f"/api/jobs/{job_id}").json()
        if final["status"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(0.25)

    check("任务最终 SUCCEEDED", final["status"] == "SUCCEEDED",
          f"status={final['status']} err={final.get('error_message')}")
    check("进度到 100%", final["progress"]["percentage"] == 100.0)
    check("步数被解析", final["progress"]["total_steps"] == 20,
          str(final["progress"]))
    check("has_adapter=True", final["has_adapter"] is True)
    check("响应不含内部路径字段", "_path" not in final)

    # 曲线
    curve = client.get(f"/api/jobs/{job_id}/curve").json()
    check("曲线点数 == 20", len(curve["points"]) == 20, str(len(curve["points"])))
    if curve["points"]:
        check("曲线 loss 递减",
              curve["points"][0]["loss"] > curve["points"][-1]["loss"])

    # 日志
    logs = client.get(f"/api/jobs/{job_id}/logs?tail=50").text
    check("日志可读", "[fake]" in logs)

    # 产物 + 下载
    arts = client.get(f"/api/jobs/{job_id}/artifacts").json()
    check("产物列表非空", len(arts["files"]) > 0)
    dl = client.get(f"/api/jobs/{job_id}/download")
    check("下载 zip 成功", dl.status_code == 200 and dl.content[:2] == b"PK",
          f"HTTP {dl.status_code}")
    plot = client.get(f"/api/jobs/{job_id}/loss_plot")
    check("loss_plot 返回 PNG", plot.status_code == 200)

    # 幂等：重复下载复用已打好的 zip
    check("重复下载幂等", client.get(f"/api/jobs/{job_id}/download").status_code == 200)

    # SSE 流
    with client.stream("GET", f"/api/jobs/{job_id}/stream") as s:
        body = ""
        for chunk in s.iter_text():
            body += chunk
            if "event: done" in body:
                break
    check("SSE 推送 progress+done", "event: progress" in body and "event: done" in body)

    # 状态持久化 & 重启恢复
    meta = Path(os.environ["WORKSPACE_DIR"]) / "jobs" / job_id / "meta.json"
    check("meta.json 已落盘", meta.is_file())
    trainer._jobs.clear()
    trainer.load_jobs_from_disk()
    check("重启后能恢复任务", job_id in trainer._jobs)

    # 已完成任务不可取消，可删除
    check("完成的任务取消返回 400",
          client.post(f"/api/jobs/{job_id}/cancel").status_code == 400)
    check("删除任务成功", client.delete(f"/api/jobs/{job_id}").status_code == 200)
    check("删除后 404", client.get(f"/api/jobs/{job_id}").status_code == 404)


# ================= 7. 失败路径：假脚本崩溃 → 报错友好化 =================
print("\n[7] 失败路径")
crash_cli = TMP / "crash.py"
crash_cli.write_text(
    'import sys\n'
    'sys.stderr.write("Traceback (most recent call last):\\n")\n'
    'sys.stderr.write("torch.cuda.OutOfMemoryError: CUDA out of memory.\\n")\n'
    'sys.exit(1)\n',
    encoding="utf-8",
)
trainer._train_cmd = lambda cfg_path: [sys.executable, str(crash_cli), str(cfg_path)]

r = client.post("/api/jobs", json={
    "model_name": "fake-0.5B", "dataset_name": ds_name, "hardware_preset": "cpu"})
if r.status_code == 200:
    jid = r.json()["job_id"]
    fin = None
    for _ in range(80):
        fin = client.get(f"/api/jobs/{jid}").json()
        if fin["status"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(0.25)
    check("崩溃 → FAILED", fin["status"] == "FAILED", f"status={fin['status']}")
    check("报错友好化为「显存不足」", "显存不足" in (fin.get("error_message") or ""),
          str(fin.get("error_message")))
    check("含修复建议", bool(fin.get("error_hint")))
    check("含原始 traceback", "OutOfMemoryError" in (fin.get("error_raw") or ""))
    check("失败任务无产物下载", client.get(f"/api/jobs/{jid}/download").status_code == 404)

trainer._train_cmd = _orig_cmd


# ================= 汇总 =================
_ctx.__exit__(None, None, None)
shutil.rmtree(TMP, ignore_errors=True)

print("\n" + "=" * 60)
print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
if FAIL:
    print("\n失败项：")
    for f in FAIL:
        print(f"  ✗ {f}")
    print("=" * 60)
    sys.exit(1)
print("ALL CHECKS PASSED")
print("=" * 60)
