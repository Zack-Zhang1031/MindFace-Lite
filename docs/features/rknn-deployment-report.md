# RKNN 部署报告

## overview

RKNN 转换脚本会保存部署报告，记录转换环境、ONNX 输入、RKNN 输出、量化数据集和后续板端验证步骤。入口是 `scripts/24_rknn_convert_and_infer.py`。

## design decisions

- 报告默认写到 `outputs/reports/rknn_deploy_report.json`。
- `--dry-run` 也会生成报告，方便在没有完整 RKNN 转换条件时先检查配置。
- 报告强调开发机转换和 RK3588 板端推理是两个阶段。

## implementation notes

WSL 环境中运行：

```bash
python scripts/24_rknn_convert_and_infer.py --config configs/rknn_deploy.yaml --dry-run
python scripts/24_rknn_convert_and_infer.py --config configs/rknn_deploy.yaml
```

报告包含 Python、NumPy、ONNX、ONNXRuntime、RKNN-Toolkit2 版本，ONNX/RKNN 文件大小，输入 shape，量化数据集条目数，以及板端验证建议。

