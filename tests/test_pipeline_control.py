from __future__ import annotations

from pathlib import Path

import pytest

from mindface.pipelines.basic import PipelineStep, run_pipeline, select_steps


def _steps(tmp_path: Path) -> list[PipelineStep]:
    return [
        PipelineStep("audio", ("scripts/audio.py",), outputs=(tmp_path / "audio.wav",)),
        PipelineStep("train", ("scripts/train.py",), outputs=(tmp_path / "model.pt",)),
        PipelineStep("benchmark", ("scripts/bench.py",), outputs=(tmp_path / "report.json",)),
    ]


def test_select_steps_supports_inclusive_from_and_to(tmp_path) -> None:
    selected = select_steps(_steps(tmp_path), from_step="train", to_step="benchmark")

    assert [step.name for step in selected] == ["train", "benchmark"]


def test_select_steps_rejects_unknown_or_reversed_range(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown pipeline step"):
        select_steps(_steps(tmp_path), from_step="missing")
    with pytest.raises(ValueError, match="comes after"):
        select_steps(_steps(tmp_path), from_step="benchmark", to_step="audio")


def test_pipeline_dry_run_and_existing_output_do_not_execute(tmp_path) -> None:
    calls: list[list[str]] = []

    def runner(command: list[str]) -> int:
        calls.append(command)
        return 0

    dry_results = run_pipeline(_steps(tmp_path), dry_run=True, runner=runner)
    (tmp_path / "audio.wav").write_bytes(b"ready")
    skip_results = run_pipeline([_steps(tmp_path)[0]], runner=runner)

    assert calls == []
    assert all(result.status == "planned" for result in dry_results)
    assert skip_results[0].status == "skipped"


def test_pipeline_force_executes_even_when_output_exists(tmp_path) -> None:
    output = tmp_path / "audio.wav"
    output.write_bytes(b"ready")
    calls: list[list[str]] = []

    results = run_pipeline(
        [PipelineStep("audio", ("scripts/audio.py",), outputs=(output,))],
        force=True,
        runner=lambda command: calls.append(command) or 0,
        python_executable="python-test",
    )

    assert calls == [["python-test", "scripts/audio.py"]]
    assert results[0].status == "completed"

