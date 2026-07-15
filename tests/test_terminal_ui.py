from __future__ import annotations

import pytest

from mindface.cli import build_parser
from mindface.terminal_ui import (
    _select,
    action_items,
    command_with_custom_config,
    command_with_source,
    menu_groups,
)


def test_terminal_ui_catalog_exposes_every_learning_area() -> None:
    group_ids = {group.id for group in menu_groups()}

    assert group_ids == {
        "environment",
        "quick-start",
        "data",
        "training",
        "inference-deployment",
        "realtime-tts",
        "cpp-edge",
        "project-tools",
    }


def test_every_terminal_ui_action_has_environment_and_valid_cli_route() -> None:
    parser = build_parser()

    for item in action_items():
        assert item.environment
        assert item.description
        parser.parse_args(list(item.argv))


def test_custom_yaml_replaces_the_default_preset() -> None:
    item = next(item for item in action_items() if item.id == "train-mlp")

    command = command_with_custom_config(item, "configs/training/train-grid-mlp.yaml")

    assert command == ("train", "--config", "configs/training/train-grid-mlp.yaml")


def test_direction_keys_move_selection(capsys: pytest.CaptureFixture[str]) -> None:
    keys = iter(("down", "down", "up", "enter"))

    selected = _select("test", ("first", "second", "third"), read_key=lambda: next(keys))

    assert selected == 1
    assert "↑/↓" in capsys.readouterr().out


def test_environment_install_actions_are_confirmed_and_choose_source() -> None:
    item = next(item for item in action_items() if item.id == "install-windows")

    assert item.installer is True
    assert "--yes" in item.argv
    assert command_with_source(item, "tsinghua") == (
        "env",
        "install-windows",
        "--source",
        "tsinghua",
        "--yes",
    )
