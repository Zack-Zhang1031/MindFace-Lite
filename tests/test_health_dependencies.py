from __future__ import annotations

from mindface.diagnostics import health


def test_requirements_policy_reports_incompatible_installed_version(tmp_path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "--extra-index-url https://download.pytorch.org/whl/cu128\n"
        "numpy>=1.26.4,<2.0\n"
        "onnxruntime>=1.23.2,<1.27\n",
        encoding="utf-8",
    )
    versions = {"numpy": "1.26.4", "onnxruntime": "1.22.0"}

    results = health.check_requirement_versions(requirements, versions.get)
    by_name = {result.name: result for result in results}

    assert by_name["requirement:numpy"].status == "pass"
    assert by_name["requirement:onnxruntime"].status == "fail"
    assert "does not satisfy" in by_name["requirement:onnxruntime"].message

