# 环境安装与检查

## overview

MindFace-Lite 的方向键工作台提供“环境与安装”模块，可从 Windows PowerShell 同时管理 Windows 训练环境和 WSL2 Ubuntu RKNN 环境。安装器支持状态检查、完整检查、安装计划预览、创建缺失环境和修复已有环境。

Windows 使用 Conda 环境 `mindface-lite` 与 Python 3.10，安装 core、optional、dev 和 editable package。WSL 使用 Ubuntu 中的 `~/.venvs/mindface-rknn`，只安装 `requirements-rknn.txt`、RKNN-Toolkit2 和嵌入式系统工具。

## design decisions

- Windows 作为统一管理入口，通过 `wsl.exe -d Ubuntu` 管理 WSL 环境。
- 已有环境只复用和修复，安装计划不包含 `conda remove`、`rm -rf` 或其他自动删除操作。
- Windows 与 RKNN 依赖严格隔离，WSL 安装计划不会读取 `requirements-optional.txt`。
- TUI 中选择“开始安装”即完成一次确认，随后自动执行全部步骤；命令行未传 `--yes` 时会再询问一次。
- 每次安装都选择 Python 下载源：官方、清华或阿里云。该选择只影响 pip，不永久修改 Conda 或 pip 全局配置。
- 安装任一步失败后立即停止，保留已有结果，并把命令与退出码写入 `outputs/logs/environment-install-*.log`。
- `--dry-run` 只打印计划，不执行 Conda、pip、apt 或 sudo 命令。
- WSL 系统工具安装允许 `sudo apt` 在当前终端请求 Linux 密码。

## implementation notes

方向键入口：

```powershell
conda activate mindface-lite
mindface ui
```

首次安装尚未注册 CLI 时，环境入口可以直接从源码启动，且不要求预先安装 PyYAML、PyTorch 或 OpenCV：

```powershell
$env:PYTHONPATH = "src"
python -m mindface ui
```

非交互检查：

```powershell
mindface env status --distro Ubuntu
mindface env check --distro Ubuntu
```

先查看安装计划：

```powershell
mindface env install-windows --source official --dry-run
mindface env install-wsl --distro Ubuntu --source official --dry-run
```

执行安装时，省略 `--yes` 会在打印完整计划后确认一次：

```powershell
mindface env install-windows --source tsinghua
mindface env install-wsl --distro Ubuntu --source aliyun
```

自动化环境可显式跳过询问：

```powershell
mindface env install-windows --source official --yes
mindface env install-wsl --distro Ubuntu --source official --yes
```

报告和日志：

```text
outputs/reports/environment-matrix.json
outputs/logs/environment-install-windows.log
outputs/logs/environment-install-wsl.log
```

核心实现：

```text
src/mindface/environment/manager.py
src/mindface/commands/environment.py
src/mindface/terminal_ui.py
```
