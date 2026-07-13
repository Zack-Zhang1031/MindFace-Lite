# 统一 CLI 与教学工作台

## overview

MindFace-Lite 使用 `mindface` 作为唯一主入口，并提供 `mindface ui` 方向键终端菜单。菜单面向逐阶段学习，完整展示 demo、数据、训练、推理、优化、实时输入、TTS、C++ 和 RKNN 功能；暂时不能在当前环境运行的功能仍会显示，并标明所需环境。

编号脚本继续存在，但只负责把旧命令转发给 CLI，不再保存业务逻辑。核心执行代码位于 `src/mindface/commands/` 和对应领域模块。

## design decisions

- 教学优先：菜单按照学习阶段组织，而不是按文件编号组织。
- 单一执行路径：CLI 直接调用内部 command handler，pipeline 也调用 CLI，不再反向启动编号脚本。
- 零额外 TUI 依赖：Windows 使用 `msvcrt`，Linux/WSL 使用 `termios`，避免破坏 RKNN 依赖环境。
- 全功能可见：菜单项始终展示环境要求，例如 Windows CUDA、Windows 可选功能环境、WSL RKNN 或 Linux BSP。
- 预设优先：菜单默认使用仓库中的 YAML；配置型功能还可以选择“指定自定义 YAML”。
- 兼容保留：`scripts/*.py` 和早期平铺命令仍可运行，但只作为旧资料和旧命令的兼容入口。
- 延迟导入：打开菜单不会加载 PyTorch、MediaPipe、sounddevice 或 RKNN，只有执行对应功能时才加载。

## implementation notes

安装项目入口：

```powershell
cd C:\Users\Administrator\Desktop\MindFace-Lite
conda activate mindface-lite
python -m pip install -e .
```

启动方向键菜单：

```powershell
mindface ui
```

未注册短命令时也可以运行：

```powershell
python -m mindface ui
```

菜单操作：

```text
↑ / ↓        移动选择
Enter        进入或执行
Esc / Q      返回或退出
```

常用非交互命令：

```powershell
mindface health
mindface config validate --all
mindface pipeline basic --dry-run
mindface demo expressive-avatar
mindface data extract-landmarks --max-videos 8
mindface train --config configs/training/train-mlp.yaml
mindface benchmark backends
mindface deploy rknn --dry-run
mindface cpp configure
mindface cpp build
mindface cpp test
mindface project test
```

CLI 主要实现文件：

```text
src/mindface/cli.py
src/mindface/terminal_ui.py
src/mindface/commands/
scripts/_compat.py
```

路由测试会递归枚举所有 argparse 叶子命令，TUI 测试会验证所有菜单动作都能被 CLI 解析，兼容测试会限制编号脚本只能包含跳转代码。
