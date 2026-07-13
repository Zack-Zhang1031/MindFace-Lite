# Engineering Contracts And Runtime Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add validated configuration, stable feature/model artifacts, common inference backends, controllable pipelines, robust realtime queues, speaker-safe dataset splits, thin scripts, and a reusable C++ runtime library.

**Architecture:** YAML files are validated at the repository boundary and converted into explicit contracts. Training saves a versioned model bundle containing its feature contract and resumable optimizer state; inference backends consume the same feature contract. Python and C++ realtime paths use bounded queues with explicit overflow, stop, and error behavior.

**Tech Stack:** Python 3.10, PyTorch, NumPy, PyYAML, ONNXRuntime, pytest, C++17, CMake, CTest.

## Global Constraints

- Preserve all existing numbered script entry points and legacy config path aliases.
- Keep Windows `mindface-lite` and WSL `mindface-rknn` dependencies separate.
- Keep generated datasets, checkpoints, ONNX, RKNN, logs, and videos out of Git.
- Python modules use snake_case because they must be importable; docs and configuration files use kebab-case.
- Every new behavior starts with a failing test and ends with a focused passing test.

---

### Task 1: Configuration Schema Validation

**Files:** `src/mindface/configuration/schema.py`, `src/mindface/cli.py`, `tests/test_config_schema.py`.

**Interfaces:** `validate_config(path) -> ValidationReport`; `validate_all_configs(root="configs") -> list[ValidationReport]`.

- [ ] Test all YAML files plus malformed training fields and invalid split ratios; confirm missing API failures.
- [ ] Implement directory-based schema dispatch, required nested fields, numeric ranges, and ratio checks.
- [ ] Add `mindface config list|show|validate`; run `python -m pytest tests/test_config_schema.py -q`.

### Task 2: Feature And Model Artifact Contracts

**Files:** `src/mindface/audio/spec.py`, `src/mindface/artifacts/model_bundle.py`, `src/mindface/training/trainer.py`, `src/mindface/inference.py`, `src/mindface/deploy/onnx_tools.py`, `tests/test_model_bundle.py`.

**Interfaces:** `FeatureSpec(fps, frame_ms, feature_dim)`; versioned `ModelBundle`; `save_model_bundle` and `load_model_bundle` with legacy migration.

- [ ] Test feature extraction, bundle round-trip, and legacy migration; confirm imports fail first.
- [ ] Implement contracts and migrate trainer, inference, and ONNX export.
- [ ] Run model bundle and backend consistency tests.

### Task 3: Resumable Mini Training

**Files:** `src/mindface/training/trainer.py`, `scripts/03_train_model.py`, `tests/test_training_resume.py`.

**Interfaces:** optional `train.resume_from`; best and last checkpoints include optimizer, epoch, feature spec, and metrics.

- [ ] Build a tiny manifest and test train, save, load, resume, optimizer restoration, and epoch advancement.
- [ ] Confirm resume test fails, implement minimal resume behavior, then run it on CPU until green.

### Task 4: Common Inference Backend Interface

**Files:** `src/mindface/backends/base.py`, `pytorch_backend.py`, `onnx_backend.py`, `rknn_backend.py`, `src/mindface/deploy/consistency.py`, `tests/test_backends.py`.

**Interfaces:** `MouthPredictor.predict(features) -> np.ndarray`; `metadata() -> dict`; PyTorch, ONNXRuntime, and optional RKNN adapters.

- [ ] Test protocol shape and errors with a real tiny PyTorch model; confirm missing modules fail.
- [ ] Implement adapters, route consistency comparison through them, and run focused tests.

### Task 5: Controllable Pipeline And Thin Scripts

**Files:** `src/mindface/pipelines/basic.py`, `src/mindface/data/synthetic_generation.py`, `src/mindface/deploy/benchmark.py`, `src/mindface/deploy/rknn_pipeline.py`, four matching scripts, `src/mindface/cli.py`, `tests/test_pipeline_control.py`.

**Interfaces:** `PipelineStep`; `select_steps(from_step, to_step)`; `run_pipeline(..., dry_run, force)`; CLI `--dry-run`, `--from-step`, `--to-step`, `--force`.

- [ ] Test selection, invalid names, dry-run, and skip/force behavior; confirm failures.
- [ ] Move reusable logic into package modules, retain thin wrappers, and run CLI dry-run tests.

### Task 6: Speaker-Disjoint Dataset Splits

**Files:** `src/mindface/data/splitting.py`, GRID preparation modules/configs, `tests/test_dataset_splitting.py`.

**Interfaces:** `speaker_disjoint_split(speakers, ratios, seed) -> dict[str, str]`; manifests add `schema_version`.

- [ ] Test determinism, speaker disjointness, ratios, and small speaker counts; confirm missing API failure.
- [ ] Implement speaker assignment and integrate GRID audio and landmark manifests.

### Task 7: Robust Python Realtime Queue

**Files:** `src/mindface/realtime/bounded_queue.py`, `src/mindface/realtime/pipeline.py`, `tests/test_realtime_queue.py`.

**Interfaces:** `BoundedDropQueue.put/get/stop/fail`; policies `block`, `drop_oldest`, `drop_newest`; accepted/dropped counters.

- [ ] Test full queues, drop policies, stopped waiters, and exception propagation; confirm missing API failure.
- [ ] Implement with condition variables, integrate the WAV pipeline, and run repeatedly for deadlock detection.

### Task 8: Reusable C++ Runtime And CTest

**Files:** `cpp/include/mindface/bounded_queue.hpp`, `mouth_params.hpp`, `cpp/src/runtime.cpp`, `cpp/apps/*.cpp`, `cpp/tests/test_bounded_queue.cpp`, `cpp/CMakePresets.json`, `cpp/CMakeLists.txt`.

**Interfaces:** `mindface_runtime` C++17 library; `BoundedQueue<T>` with close and drop-oldest; CTest target `mindface_queue_tests`.

- [ ] Add tests for overflow, FIFO, close, and waiting consumer release; confirm build failure before headers exist.
- [ ] Implement library/apps/presets, then configure, build, and run CTest.

### Task 9: Complete CLI, YAML, Documentation, And Verification

**Files:** `tests/test_cli.py`, `README.md`, `RUN_ORDER.md`, `PROJECT_DEMO.md`, `docs/features/engineering-contracts.md`, `docs/features/README.md`, `docs/progress.md`.

**Interfaces:** CLI route matrix enumerates every leaf parser and maps each to an existing script.

- [ ] Expand CLI route coverage and all-YAML tests.
- [ ] Update commands, contracts, resume behavior, Pipeline controls, C++ layout, and debugging notes.
- [ ] Run full pytest, compileall, docs guard, CMake/CTest, health check, pipeline dry-run, and `git diff --check`.
