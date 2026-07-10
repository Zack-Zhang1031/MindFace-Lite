# 一键健康检查

## overview

健康检查用于快速确认当前环境、依赖、关键数据和输出文件是否满足继续开发或演示的条件。入口是 `scripts/99_health_check.py`，输出 JSON 报告到 `outputs/reports/health_check.json`。

## design decisions

- 使用轻量标准库实现，不引入新的诊断依赖。
- 检查结果分为 `pass`、`warn`、`fail`，方便区分阻塞项和可选项。
- 同时检查 Python 包、CUDA、关键路径、GRID manifest 和外部工具。

## implementation notes

运行命令：

```powershell
python scripts/99_health_check.py
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

