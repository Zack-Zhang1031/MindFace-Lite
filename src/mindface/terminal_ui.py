from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ActionItem:
    id: str
    title: str
    description: str
    argv: tuple[str, ...]
    environment: str
    installer: bool = False

    @property
    def supports_custom_config(self) -> bool:
        return "--config" in self.argv


@dataclass(frozen=True, slots=True)
class MenuGroup:
    id: str
    title: str
    description: str
    items: tuple[ActionItem, ...]


def _action(
    item_id: str,
    title: str,
    description: str,
    argv: Sequence[str],
    environment: str = "Windows / WSL 基础环境",
    *,
    installer: bool = False,
) -> ActionItem:
    return ActionItem(item_id, title, description, tuple(argv), environment, installer)


def menu_groups() -> tuple[MenuGroup, ...]:
    windows_gpu = "Windows mindface-lite；训练可使用 NVIDIA CUDA GPU"
    windows_optional = "Windows 可选功能环境；需要 requirements-optional.txt"
    wsl_rknn = "WSL mindface-rknn；需要 Rockchip RKNN-Toolkit2"
    linux_bsp = "Linux / RK3588 BSP 环境；需要 dtc 和板级工具"
    return (
        MenuGroup(
            "environment",
            "环境与安装",
            "检查、创建或修复 Windows 训练环境和 Ubuntu RKNN 环境。",
            (
                _action(
                    "environment-status",
                    "查看环境状态",
                    "快速检查 Conda、mindface-lite、WSL Ubuntu 和 RKNN venv 是否存在。",
                    ("env", "status", "--distro", "Ubuntu"),
                    "Windows PowerShell；可同时检查 WSL2 Ubuntu",
                ),
                _action(
                    "environment-check",
                    "完整检查两个环境",
                    "检查 Python 包、CUDA/GPU、RKNN、pip、CMake、dtc 和交叉编译器。",
                    ("env", "check", "--distro", "Ubuntu"),
                    "Windows PowerShell；WSL 发行版名称为 Ubuntu",
                ),
                _action(
                    "install-windows",
                    "安装或修复 Windows 环境",
                    "复用 mindface-lite；缺失时创建 Python 3.10，并安装 core、optional、dev。",
                    ("env", "install-windows", "--source", "official", "--yes"),
                    "Windows + Conda；现有环境不会被删除",
                    installer=True,
                ),
                _action(
                    "install-wsl",
                    "安装或修复 WSL RKNN 环境",
                    "管理 Ubuntu 的 ~/.venvs/mindface-rknn，并安装 apt、RKNN 和交叉编译工具。",
                    ("env", "install-wsl", "--distro", "Ubuntu", "--source", "official", "--yes"),
                    "Windows + WSL2 Ubuntu；sudo 阶段需要输入 Linux 密码",
                    installer=True,
                ),
            ),
        ),
        MenuGroup(
            "quick-start",
            "快速体验",
            "从规则嘴型逐步观察更完整的 2D 数字人效果。",
            (
                _action("generate-audio", "生成测试音频", "生成能量变化明确的测试 WAV。", ("demo", "generate-audio")),
                _action("rule-demo", "Stage 1 RMS 规则嘴型", "学习 RMS 到 mouth_open 的映射。", ("demo", "rule", "--config", "configs/demos/rule-demo.yaml")),
                _action("better-visual", "Stage 1.5 改进渲染", "使用更自然的 2D 嘴部绘制。", ("demo", "better-visual", "--config", "configs/demos/better-visual-demo.yaml")),
                _action("expressive-avatar", "Stage 1.6 表情头像", "运行静态人脸、嘴部形变、眨眼和轻微头动。", ("demo", "expressive-avatar", "--config", "configs/demos/expressive-avatar-demo.yaml")),
                _action("basic-pipeline", "基础流水线", "按顺序运行数据、训练、推理、部署与 benchmark。", ("pipeline", "basic"), windows_gpu),
            ),
        ),
        MenuGroup(
            "data",
            "数据处理",
            "生成合成数据，并准备 GRID 音频和视频 landmark 标签。",
            (
                _action("synthetic-data", "生成合成训练集", "生成便于调试训练闭环的 NPZ 与 manifest。", ("data", "synthetic", "--config", "configs/datasets/synthetic-dataset.yaml")),
                _action("grid-audio-debug", "GRID 音频快速预处理", "只处理 8 个样本验证路径。", ("data", "prepare-grid", "--config", "configs/datasets/prepare-grid.yaml", "--max-samples", "8")),
                _action("grid-audio-all", "GRID 音频完整预处理", "扫描并处理全部可用 GRID WAV。", ("data", "prepare-grid", "--config", "configs/datasets/prepare-grid.yaml")),
                _action("landmark-deps", "检查 MediaPipe", "仅检查 landmark 可选依赖。", ("data", "extract-landmarks", "--config", "configs/datasets/grid-video-landmarks.yaml", "--check-deps"), windows_optional),
                _action("landmark-debug", "提取 8 个视频 landmark", "小规模验证视频标签提取。", ("data", "extract-landmarks", "--config", "configs/datasets/grid-video-landmarks.yaml", "--max-videos", "8"), windows_optional),
                _action("landmark-all", "提取全部视频 landmark", "运行完整 GRID 视频标签提取。", ("data", "extract-landmarks", "--config", "configs/datasets/grid-video-landmarks.yaml"), windows_optional),
                _action("landmark-quality", "生成 landmark 质量报告", "统计检测率和口型参数分布。", ("data", "extract-landmarks", "--config", "configs/datasets/grid-video-landmarks.yaml", "--quality-only"), windows_optional),
                _action("align-landmarks", "对齐音频与 landmark", "生成真实视频口型监督训练 manifest。", ("data", "align-landmarks", "--config", "configs/datasets/prepare-grid-landmark.yaml")),
            ),
        ),
        MenuGroup(
            "training",
            "模型训练",
            "比较 MLP、LSTM、TCN、Transformer 和不同数据来源。",
            (
                _action("train-mlp", "训练 MLP", "最简单的逐帧基线模型。", ("train", "--config", "configs/training/train-mlp.yaml"), windows_gpu),
                _action("train-lstm", "训练 LSTM", "学习时序记忆与序列建模。", ("train", "--config", "configs/training/train-lstm.yaml"), windows_gpu),
                _action("train-tcn", "训练 TCN", "学习因果卷积时序模型。", ("train", "--config", "configs/training/train-tcn.yaml"), windows_gpu),
                _action("train-transformer", "训练 Transformer", "学习自注意力序列建模。", ("train", "--config", "configs/training/train-transformer.yaml"), windows_gpu),
                _action("train-grid-debug", "GRID 快速训练", "使用少量 GRID 预处理数据检查训练闭环。", ("train", "--config", "configs/training/train-grid-debug-mlp.yaml"), windows_gpu),
                _action("train-grid-rms", "GRID RMS 标签训练", "使用完整 GRID 音频规则标签。", ("train", "--config", "configs/training/train-grid-mlp.yaml"), windows_gpu),
                _action("train-grid-landmark", "GRID landmark 标签训练", "使用视频 landmark 真实口型监督。", ("train", "--config", "configs/training/train-grid-landmark-mlp.yaml"), windows_gpu),
            ),
        ),
        MenuGroup(
            "inference-deployment",
            "推理、优化与部署",
            "完成 PyTorch、ONNX、量化、剪枝和后端一致性闭环。",
            (
                _action("infer-pytorch", "PyTorch 推理", "从 WAV 预测口型并生成视频。", ("infer", "pytorch", "--config", "configs/inference/infer-pytorch.yaml"), windows_gpu),
                _action("export-onnx", "导出 ONNX", "将 ModelBundle 导出为部署模型。", ("export", "onnx", "--config", "configs/deployment/export-onnx.yaml"), windows_gpu),
                _action("infer-onnx", "ONNXRuntime 推理", "验证 ONNX 输出视频。", ("infer", "onnx", "--config", "configs/inference/infer-onnx.yaml")),
                _action("benchmark-runtime", "运行时 benchmark", "比较 PyTorch 与 ONNXRuntime 延迟。", ("benchmark", "runtime", "--config", "configs/benchmarks/benchmark.yaml")),
                _action("compare-backends", "后端一致性", "比较 PyTorch、ONNXRuntime 和可选 RKNN。", ("benchmark", "backends", "--config", "configs/benchmarks/backend-consistency.yaml")),
                _action("quantize", "ONNX 动态量化", "生成 INT8 dynamic ONNX。", ("optimize", "quantize", "--config", "configs/optimization/quantize-onnx.yaml")),
                _action("benchmark-quantized", "量化前后对比", "比较模型大小、误差和延迟。", ("optimize", "benchmark-quantized", "--config", "configs/benchmarks/benchmark-quantized-onnx.yaml")),
                _action("prune", "剪枝与微调", "对模型剪枝并恢复精度。", ("optimize", "prune", "--config", "configs/optimization/prune-finetune.yaml"), windows_gpu),
                _action("benchmark-pruned", "剪枝前后对比", "比较稀疏率、误差和延迟。", ("optimize", "benchmark-pruned", "--config", "configs/benchmarks/benchmark-pruned.yaml")),
            ),
        ),
        MenuGroup(
            "realtime-tts",
            "实时输入与 TTS",
            "学习队列、麦克风输入和语音合成驱动。",
            (
                _action("realtime-queue", "实时队列模拟", "统计 FPS、延迟、丢帧与队列状态。", ("realtime", "queue", "--config", "configs/realtime/realtime-rule.yaml")),
                _action("mic-deps", "检查麦克风依赖", "检查 PortAudio 与 sounddevice。", ("realtime", "microphone", "--config", "configs/realtime/mic-stream.yaml", "--check-deps"), windows_optional),
                _action("mic-devices", "列出麦克风设备", "显示当前系统可见的录音设备。", ("realtime", "microphone", "--config", "configs/realtime/mic-stream.yaml", "--list-devices"), windows_optional),
                _action("mic-demo", "麦克风实时嘴型", "录音 10 秒并显示 OpenCV 窗口。", ("realtime", "microphone", "--config", "configs/realtime/mic-stream.yaml", "--duration-sec", "10", "--show"), windows_optional),
                _action("pseudo-tts", "模拟 TTS 音频", "离线生成确定性的 TTS-like WAV。", ("tts", "pseudo-generate", "--config", "configs/realtime/tts-demo.yaml")),
                _action("pseudo-tts-demo", "模拟 TTS 嘴型", "生成音频并驱动规则嘴型。", ("tts", "pseudo-demo", "--config", "configs/realtime/tts-demo.yaml")),
                _action("real-tts-deps", "检查真实 TTS", "检查 pyttsx3 或 edge-tts。", ("tts", "generate", "--config", "configs/realtime/real-tts.yaml", "--check-deps"), windows_optional),
                _action("real-tts", "生成真实 TTS", "调用配置中的真实 TTS 后端。", ("tts", "generate", "--config", "configs/realtime/real-tts.yaml"), windows_optional),
                _action("real-tts-demo", "真实 TTS 嘴型", "真实 TTS 音频驱动嘴部视频。", ("tts", "demo", "--config", "configs/realtime/real-tts.yaml"), windows_optional),
            ),
        ),
        MenuGroup(
            "cpp-edge",
            "C++ 与边缘部署",
            "构建实时控制程序并进入 RKNN、Device Tree 与 U-Boot 路径。",
            (
                _action("cpp-configure", "配置 C++ 工程", "按当前系统选择 CMake preset。", ("cpp", "configure"), "Windows MSVC 或 Linux Ninja + CMake"),
                _action("cpp-build", "构建 C++ 工程", "编译队列、UDP、串口和测试程序。", ("cpp", "build"), "Windows MSVC 或 Linux Ninja + CMake"),
                _action("cpp-test", "运行 C++ 测试", "通过 CTest 验证有界队列。", ("cpp", "test"), "Windows MSVC 或 Linux Ninja + CMake"),
                _action("cpp-queue", "运行 C++ 队列 demo", "运行 producer-consumer 队列程序。", ("cpp", "run", "queue-demo"), "已完成 C++ build"),
                _action("cpp-udp", "运行 UDP sender", "发送嘴型参数到 UDP 接收端。", ("cpp", "run", "udp-sender"), "已完成 C++ build"),
                _action("cpp-serial", "运行串口 sender", "向机器人嘴部执行器输出控制值。", ("cpp", "run", "serial-sender"), "已完成 C++ build；需要串口设备"),
                _action("cpp-onnx", "运行 C++ ONNXRuntime demo", "使用 ONNXRuntime C++ SDK 执行模型。", ("cpp", "run", "onnxruntime-cpp-demo"), "需要启用 BUILD_ONNXRUNTIME_DEMO 并完成 C++ build"),
                _action("rknn-deps", "检查 RKNN 环境", "检查 RKNN-Toolkit2。", ("deploy", "rknn", "--config", "configs/deployment/rknn-deploy.yaml", "--check-deps"), wsl_rknn),
                _action("rknn-dry-run", "生成 RKNN 部署计划", "不转换模型，只生成部署报告。", ("deploy", "rknn", "--config", "configs/deployment/rknn-deploy.yaml", "--dry-run"), wsl_rknn),
                _action("rknn-convert", "转换 RKNN 模型", "将 ONNX 转换为 RKNN。", ("deploy", "rknn", "--config", "configs/deployment/rknn-deploy.yaml"), wsl_rknn),
                _action("rknn-quantize", "RKNN INT8 转换", "使用校准数据生成量化 RKNN。", ("deploy", "rknn", "--config", "configs/deployment/rknn-deploy.yaml", "--quantize"), wsl_rknn),
                _action("rknn-infer", "RKNN 推理验证", "转换后调用 RKNN Runtime 推理。", ("deploy", "rknn", "--config", "configs/deployment/rknn-deploy.yaml", "--run-inference"), "RKNN 开发环境或连接 RK3588 设备"),
                _action("device-tree-deps", "检查 Device Tree 工具", "检查 dtc。", ("deploy", "device-tree", "--config", "configs/deployment/device-tree-uboot.yaml", "--check-deps"), linux_bsp),
                _action("device-tree", "编译 Device Tree overlay", "编译 DTBO 并输出 U-Boot 操作步骤。", ("deploy", "device-tree", "--config", "configs/deployment/device-tree-uboot.yaml"), linux_bsp),
            ),
        ),
        MenuGroup(
            "project-tools",
            "项目检查",
            "验证环境、配置、输出、Python 和 C++ 工程状态。",
            (
                _action("health", "健康检查", "检查依赖、数据、模型和外部工具。", ("health",)),
                _action("config-list", "列出全部 YAML", "按领域列出当前配置文件。", ("config", "list")),
                _action("config-validate", "校验全部 YAML", "运行统一配置 schema 校验。", ("config", "validate", "--all")),
                _action("verify-outputs", "检查基础输出", "确认基础流水线产物完整。", ("verify",)),
                _action("pytest", "运行 Python 测试", "运行完整 pytest 测试集。", ("project", "test")),
                _action("compileall", "Python 语法检查", "编译检查 src 与 scripts。", ("project", "compile")),
                _action("pipeline-plan", "查看流水线计划", "只展示将执行的步骤。", ("pipeline", "basic", "--dry-run")),
            ),
        ),
    )


def action_items() -> tuple[ActionItem, ...]:
    return tuple(item for group in menu_groups() for item in group.items)


def command_with_custom_config(item: ActionItem, path: str) -> tuple[str, ...]:
    if not item.supports_custom_config:
        raise ValueError(f"Action '{item.id}' does not accept a YAML config")
    command = list(item.argv)
    index = command.index("--config")
    command[index + 1] = path
    return tuple(command)


def command_with_source(item: ActionItem, source: str) -> tuple[str, ...]:
    if "--source" not in item.argv:
        raise ValueError(f"Action '{item.id}' does not accept a download source")
    command = list(item.argv)
    index = command.index("--source")
    command[index + 1] = source
    return tuple(command)


def _read_key() -> str:
    if os.name == "nt":
        import msvcrt

        key = msvcrt.getwch()
        if key in {"\x00", "\xe0"}:
            return {"H": "up", "P": "down"}.get(msvcrt.getwch(), "other")
        return {"\r": "enter", "\x1b": "escape"}.get(key, key.lower())
    import select
    import termios
    import tty

    file_descriptor = sys.stdin.fileno()
    old_settings = termios.tcgetattr(file_descriptor)
    try:
        tty.setraw(file_descriptor)
        key = os.read(file_descriptor, 1).decode(errors="ignore")
        if key == "\x1b":
            ready, _, _ = select.select([file_descriptor], [], [], 0.03)
            tail = os.read(file_descriptor, 2).decode(errors="ignore") if ready else ""
            return {"[A": "up", "[B": "down"}.get(tail, "escape")
        return {"\r": "enter", "\n": "enter"}.get(key, key.lower())
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, old_settings)


def _select(title: str, labels: Sequence[str], read_key: Callable[[], str] = _read_key) -> int | None:
    selected = 0
    while True:
        sys.stdout.write("\x1b[2J\x1b[H")
        print("MindFace-Lite 教学工作台")
        print(f"\n{title}\n")
        for index, label in enumerate(labels):
            marker = ">" if index == selected else " "
            print(f" {marker} {label}")
        print("\n↑/↓ 选择，Enter 确认，Esc/Q 返回")
        key = read_key()
        if key == "up":
            selected = (selected - 1) % len(labels)
        elif key == "down":
            selected = (selected + 1) % len(labels)
        elif key == "enter":
            return selected
        elif key in {"escape", "q"}:
            return None


def _run_action(item: ActionItem, execute: Callable[[list[str]], int]) -> None:
    labels = ["开始安装（确认后自动执行）", "仅查看安装计划"] if item.installer else ["运行默认预设"]
    if item.supports_custom_config and not item.installer:
        labels.append("指定自定义 YAML")
    labels.append("返回")
    selection = _select(
        f"{item.title}\n\n用途：{item.description}\n环境：{item.environment}\n命令：mindface {' '.join(item.argv)}",
        labels,
    )
    if selection is None or labels[selection] == "返回":
        return
    command = item.argv
    if item.installer:
        source_index = _select("选择本次 Python 下载源", ["官方源", "清华源", "阿里云源"])
        if source_index is None:
            return
        command = command_with_source(item, ("official", "tsinghua", "aliyun")[source_index])
        if labels[selection] == "仅查看安装计划":
            command = (*command, "--dry-run")
    elif labels[selection] == "指定自定义 YAML":
        path = input("\n请输入 YAML 路径：").strip()
        if not path:
            return
        command = command_with_custom_config(item, path)
    print(f"\n执行：mindface {' '.join(command)}\n")
    try:
        return_code = execute(list(command))
        print(f"\n命令结束，退出码：{return_code}")
    except Exception as exc:
        print(f"\n运行失败：{exc}")
    input("按 Enter 返回菜单...")


def launch(execute: Callable[[list[str]], int]) -> int:
    groups = menu_groups()
    while True:
        group_index = _select("选择学习模块", [f"{group.title} - {group.description}" for group in groups])
        if group_index is None:
            print("已退出 MindFace-Lite 教学工作台。")
            return 0
        group = groups[group_index]
        item_index = _select(group.title, [f"{item.title} [{item.environment}]" for item in group.items])
        if item_index is not None:
            _run_action(group.items[item_index], execute)
