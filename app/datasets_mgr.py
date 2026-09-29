"""数据集：上传 → 校验 → 转 alpaca 格式 → 注册到 LLaMA-Factory"""
import csv
import json
import re
import uuid
from pathlib import Path

from .config import DATASET_DIR, LF_DIR, REGISTRY

# 用户的字段名千奇百怪，这里做别名兼容
FIELD_ALIASES = {
    "instruction": ["instruction", "question", "prompt", "query", "input_text", "问题", "指令"],
    "output": ["output", "answer", "response", "completion", "target", "答案", "回答"],
    "input": ["input", "context", "上下文"],
    "system": ["system", "system_prompt", "系统提示"],
}


def _pick(row: dict, canonical: str) -> str | None:
    for alias in FIELD_ALIASES[canonical]:
        val = row.get(alias)
        if val not in (None, ""):
            return str(val)
    return None


def _clean_markers(text: str) -> str:
    """删掉 GSM8K 的计算器标记 <<...>>，保留 #### 答案分隔符"""
    return re.sub(r"<<[^>]*>>", "", text).strip()


def _load_raw(path: Path) -> list[dict]:
    suffix = path.suffix.lower()

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):  # 兼容 {"data": [...]} 包一层
            for key in ("data", "records", "items", "train"):
                if isinstance(data.get(key), list):
                    return data[key]
            raise ValueError("JSON 顶层是对象但找不到数组字段（data/records/items/train）")
        if not isinstance(data, list):
            raise ValueError("JSON 顶层必须是数组")
        return data

    if suffix == ".jsonl":
        rows = []
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {i} 行 JSON 解析失败：{e}") from e
        return rows

    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    raise ValueError(f"不支持的文件类型：{suffix}")


def _read_registry() -> dict:
    if not REGISTRY.exists():
        return {}
    try:
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_entry(info: dict) -> None:
    reg = _read_registry()
    reg[info["name"]] = info
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


def _register_to_lf(ds_name: str, json_path: Path, has_system: bool) -> None:
    """写进 LLaMA-Factory 的 dataset_info.json（用绝对路径，无需拷文件）"""
    info_path = LF_DIR / "data" / "dataset_info.json"
    if not info_path.parent.exists():
        raise ValueError(
            f"找不到 LLaMA-Factory 的 data 目录：{info_path.parent}\n"
            "请确认已 git clone LLaMA-Factory，或设置环境变量 LLAMAFACTORY_DIR"
        )
    info = {}
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            info = {}

    columns = {"prompt": "instruction", "query": "input", "response": "output"}
    if has_system:
        columns["system"] = "system"
    info[ds_name] = {"file_name": str(json_path.resolve()), "columns": columns}
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def convert_and_register(upload_path: Path, display_name: str,
                         clean_calc_markers: bool = True) -> dict:
    """核心函数：原始文件 → 校验 → alpaca 格式 → 落盘 → 注册"""
    raw = _load_raw(upload_path)
    if not raw:
        raise ValueError("数据集为空")

    records, skipped = [], 0
    for row in raw:
        if not isinstance(row, dict):
            skipped += 1
            continue
        instruction = _pick(row, "instruction")
        output = _pick(row, "output")
        if not instruction or not output:
            skipped += 1
            continue
        if clean_calc_markers:
            output = _clean_markers(output)
        rec = {"instruction": instruction.strip(), "input": "", "output": output}
        if extra := _pick(row, "input"):
            rec["input"] = extra.strip()
        if sys_p := _pick(row, "system"):
            rec["system"] = sys_p.strip()
        records.append(rec)

    if not records:
        raise ValueError(
            "没有解析出任何有效样本。请确认文件包含「问题字段」"
            "(instruction/question/prompt) 和「答案字段」(output/answer/response)"
        )

    slug = re.sub(r"[^0-9a-zA-Z_\-]", "_", display_name)[:40] or "dataset"
    ds_name = f"ds_{uuid.uuid4().hex[:6]}_{slug}"
    out_path = DATASET_DIR / f"{ds_name}.json"
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    _register_to_lf(ds_name, out_path, has_system=any("system" in r for r in records))

    info = {
        "name": ds_name,
        "display_name": display_name,
        "count": len(records),
        "skipped": skipped,
        "path": str(out_path),
        "source": "upload",
        "preview": records[:3],
    }
    _save_entry(info)
    return info


def install_builtin_gsm8k() -> dict:
    """一键装内置 GSM8K，让用户不上传也能直接试跑"""
    if existing := get_dataset("builtin_gsm8k"):
        return existing

    from datasets import load_dataset  # 延迟导入，没装 datasets 也能启动服务

    ds = load_dataset("openai/gsm8k", "main", split="train")
    system_prompt = (
        "You are a helpful assistant that solves grade school math problems. "
        "Think step by step, then give the final numeric answer after '#### '."
    )
    records = [
        {
            "instruction": it["question"].strip(),
            "input": "",
            "output": _clean_markers(it["answer"]),
            "system": system_prompt,
        }
        for it in ds
    ]

    out_path = DATASET_DIR / "builtin_gsm8k.json"
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    _register_to_lf("builtin_gsm8k", out_path, has_system=True)

    info = {
        "name": "builtin_gsm8k",
        "display_name": "GSM8K (内置)",
        "count": len(records),
        "skipped": 0,
        "path": str(out_path),
        "source": "builtin",
        "preview": records[:3],
    }
    _save_entry(info)
    return info


def list_datasets() -> list[dict]:
    return sorted(_read_registry().values(), key=lambda x: x["name"])


def get_dataset(name: str) -> dict | None:
    return _read_registry().get(name)


def delete_dataset(name: str) -> bool:
    reg = _read_registry()
    if name not in reg:
        return False
    Path(reg[name]["path"]).unlink(missing_ok=True)
    del reg[name]
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    return True
