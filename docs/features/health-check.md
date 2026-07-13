# 一键健康检查

## overview

健康检查用于快速确认当前环境、依赖、关键数据和输出文件是否满足继续开发或演示的条件。入口是 `scripts/99_health_check.py`，输出 JSON 报告到 `outputs/reports/health_check.json`。

## design decisions

- 使用 `packaging` 解析 requirements 版本约束，避免手写版本字符串比较。
- 检查结果分为 `pass`、`warn`、`fail`，方便区分阻塞项和可选项。
- 同时检查 Python 包导入、主环境 requirements 版本、CUDA、关键路径、GRID manifest 和外部工具。

## implementation notes

运行命令：

```powershell
python scripts/99_health_check.py
python -m mindface health
```

报告结构：

```text
summary
checks[]
  name
  status
  message
  details
```

如果在 WSL RKNN 环境运行，MediaPipe、麦克风等 Windows 侧功能可能显示 `warn`，这不代表 RKNN 转换不可用。

`requirement:*` 检查为 `fail` 时，表示包虽然可能可以导入，但实际版本不满足 `requirements.txt`。使用当前环境的 Python 重新安装依赖后再运行健康检查。
