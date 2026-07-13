from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import mindface.cli as cli


CLI_LEAF_CASES: list[tuple[list[str], str | None]] = [
    (["ui"], None),
    (["health"], "health"),
    (["config", "list"], None),
    (["config", "show", "configs/training/train-mlp.yaml"], None),
    (["config", "validate", "--all"], None),
    (["pipeline", "basic"], "pipeline.basic"),
    (["verify"], "verify"),
    (["demo", "generate-audio"], "demo.generate-audio"),
    (["demo", "rule"], "demo.rule"),
    (["demo", "better-visual"], "demo.better-visual"),
    (["demo", "expressive-avatar"], "demo.expressive-avatar"),
    (["data", "synthetic"], "data.synthetic"),
    (["data", "prepare-grid"], "data.prepare-grid"),
    (["data", "extract-landmarks"], "data.extract-landmarks"),
    (["data", "align-landmarks"], "data.align-landmarks"),
    (["train"], "train"),
    (["infer", "pytorch"], "infer.pytorch"),
    (["infer", "onnx"], "infer.onnx"),
    (["export", "onnx"], "export.onnx"),
    (["optimize", "quantize"], "optimize.quantize"),
    (["optimize", "benchmark-quantized"], "optimize.benchmark-quantized"),
    (["optimize", "prune"], "optimize.prune"),
    (["optimize", "benchmark-pruned"], "optimize.benchmark-pruned"),
    (["benchmark", "runtime"], "benchmark.runtime"),
    (["benchmark", "backends"], "benchmark.backends"),
    (["realtime", "queue"], "realtime.queue"),
    (["realtime", "microphone"], "realtime.microphone"),
    (["tts", "pseudo-generate"], "tts.pseudo-generate"),
    (["tts", "pseudo-demo"], "tts.pseudo-demo"),
    (["tts", "generate"], "tts.generate"),
    (["tts", "demo"], "tts.demo"),
    (["deploy", "rknn"], "deploy.rknn"),
    (["deploy", "device-tree"], "deploy.device-tree"),
    (["cpp", "configure"], "cpp.configure"),
    (["cpp", "build"], "cpp.build"),
    (["cpp", "test"], "cpp.test"),
    (["cpp", "run", "queue-demo"], "cpp.run"),
    (["project", "test"], "project.test"),
    (["project", "compile"], "project.compile"),
    (["rule-demo"], "demo.rule"),
    (["better-visual"], "demo.better-visual"),
    (["expressive-avatar"], "demo.expressive-avatar"),
    (["compare-backends"], "benchmark.backends"),
    (["export-onnx"], "export.onnx"),
    (["prepare-grid-landmark"], "data.align-landmarks"),
]


def _discover_leaf_commands(parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    leaves: set[tuple[str, ...]] = set()
    subparser_actions = [action for action in parser._actions if isinstance(action, argparse._SubParsersAction)]
    if not subparser_actions:
        return {prefix}
    for action in subparser_actions:
        for name, subparser in action.choices.items():
            leaves.update(_discover_leaf_commands(subparser, (*prefix, name)))
    return leaves


@pytest.mark.parametrize(("argv", "expected_command_id"), CLI_LEAF_CASES)
def test_every_cli_leaf_has_a_valid_route(argv: list[str], expected_command_id: str | None) -> None:
    args = cli.build_parser().parse_args(argv)

    if expected_command_id is None:
        assert callable(args.handler)
    else:
        assert args.command_id == expected_command_id
        assert not hasattr(args, "script_name")


def test_cli_leaf_matrix_covers_every_parser_leaf() -> None:
    discovered = _discover_leaf_commands(cli.build_parser())
    grouped = {
        "config",
        "pipeline",
        "demo",
        "data",
        "infer",
        "export",
        "optimize",
        "benchmark",
        "realtime",
        "tts",
        "deploy",
        "cpp",
        "project",
    }
    documented = {
        tuple(argv[:2] if len(argv) > 1 and argv[0] in grouped else argv[:1])
        for argv, _ in CLI_LEAF_CASES
    }

    assert discovered == documented


@pytest.mark.parametrize(
    ("argv", "expected_command_id", "expected_values"),
    [
        (
            ["pipeline", "basic", "--skip-quantization"],
            "pipeline.basic",
            {"skip_quantization": True},
        ),
        (
            ["data", "extract-landmarks", "--max-videos", "8", "--delegate", "gpu"],
            "data.extract-landmarks",
            {
                "config": "configs/datasets/grid-video-landmarks.yaml",
                "max_videos": 8,
                "delegate": "gpu",
            },
        ),
        (
            ["realtime", "microphone", "--duration-sec", "10", "--show"],
            "realtime.microphone",
            {"duration_sec": 10.0, "show": True},
        ),
        (
            ["deploy", "rknn", "--dry-run"],
            "deploy.rknn",
            {"config": "configs/deployment/rknn-deploy.yaml", "dry_run": True},
        ),
    ],
)
def test_cli_dispatches_to_internal_command_handler(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected_command_id: str,
    expected_values: dict[str, object],
) -> None:
    calls: list[tuple[str, argparse.Namespace]] = []

    def fake_run_command(command_id: str, args: argparse.Namespace) -> int:
        calls.append((command_id, args))
        return 0

    monkeypatch.setattr(cli, "run_command", fake_run_command)

    assert cli.run(argv) == 0
    assert calls[0][0] == expected_command_id
    for name, value in expected_values.items():
        assert getattr(calls[0][1], name) == value


def test_config_validate_cli_checks_all_configs(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.run(["config", "validate", "--all"]) == 0
    assert "32 valid, 0 invalid" in capsys.readouterr().out


def test_numbered_scripts_are_only_compatibility_forwarders() -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    for path in scripts_dir.glob("*.py"):
        if path.name == "_compat.py":
            continue
        source = path.read_text(encoding="utf-8")
        assert "run_compat_command" in source, path.name
        assert "argparse" not in source, path.name
        assert len(source.splitlines()) <= 6, path.name
