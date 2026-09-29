"""把训练进程的原始报错翻译成人能看懂的提示 + 修复建议"""
import re

# (正则, 友好提示, 修复建议)
ERROR_PATTERNS: list[tuple[str, str, str]] = [
    (r"CUDA out of memory|OutOfMemoryError",
     "显存不足（OOM）",
     "请改用更低的硬件档位（如 gpu_4g），或减小 cutoff_len。"),

    (r"addmm_impl_cpu_.*not implemented for 'Half'",
     "CPU 环境下开启了 fp16，不被支持",
     "硬件档位请选择 cpu（该档位自动使用 fp32）。"),

    (r"fp16 and bf16 cannot be set at the same time|Cannot use both",
     "fp16 与 bf16 同时开启，冲突",
     "这是配置生成错误，请提交 issue。"),

    (r"bf16.*not supported|doesn't support bf16|BF16 is not supported",
     "当前显卡不支持 bf16",
     "GTX 16xx / RTX 20xx 是 Turing 架构，不支持 bf16。请选择 gpu_6g 档位（自动用 fp16）。"),

    (r"KeyError: '[^']+'|Cannot find dataset|dataset.*not found",
     "找不到指定的数据集",
     "数据集可能已被删除。请重新上传，或从列表中选择已存在的数据集。"),

    (r"(FileNotFoundError|OSError).*config\.json|is not a local folder",
     "找不到基座模型",
     "模型目录不存在或不完整。请到模型管理页重新拉取该模型。"),

    (r"Unknown template|template.*not (found|exist)",
     "对话模板名称错误",
     "Qwen2.5 系列用 qwen，Qwen3 用 qwen3，LLaMA3 用 llama3。"),

    (r"loss.*nan|Detected inf/nan|NaN detected",
     "训练损失变为 NaN（数值溢出）",
     "fp16 下容易溢出。请把学习率降到 5e-5，或换用支持 bf16 的显卡档位。"),

    (r"No module named ['\"]?llamafactory|No such file.*llamafactory",
     "找不到 llamafactory 模块",
     'LLaMA-Factory 未正确安装。请执行 pip install -e "./LLaMA-Factory[torch,metrics]"',),

    (r"CUDA driver version is insufficient|no kernel image is available",
     "CUDA 驱动与 PyTorch 版本不匹配",
     "请重装匹配的 PyTorch。用 nvidia-smi 查看右上角 CUDA Version。"),
]


def diagnose(log_text: str) -> tuple[str, str, str]:
    """返回 (友好提示, 修复建议, 原始报错尾部)"""
    tail_lines = log_text.splitlines()[-400:]  # 真正的报错通常在末尾
    tail = "\n".join(tail_lines)

    for pattern, msg, hint in ERROR_PATTERNS:
        if re.search(pattern, tail, re.IGNORECASE):
            return msg, hint, "\n".join(tail_lines[-40:])

    if "Traceback" in tail:
        raw = tail[tail.rindex("Traceback"):][:2000]
        non_empty = [line for line in raw.splitlines() if line.strip()]
        last = non_empty[-1][:200] if non_empty else "未知错误"
        return f"训练进程异常退出：{last}", "请查看完整日志定位问题。", raw

    return "训练进程非正常退出", "请查看完整日志定位问题。", "\n".join(tail_lines[-40:])
