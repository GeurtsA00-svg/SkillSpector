# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repeated boolean targets must use their final assigned value."""

from __future__ import annotations

import pytest

from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm


def _scan(statement: str) -> list:
    source = (
        "import subprocess\n"
        "enabled = True\n" + statement + "subprocess.run('echo ok', shell=enabled)\n"
    )
    result = tm.node({"components": ["runner.py"], "file_cache": {"runner.py": source}})
    return [finding for finding in result["findings"] if finding.rule_id == "TM1"]


@pytest.mark.parametrize("false_value", ["False", "0", "''", "None", "()"])
@pytest.mark.parametrize(
    "assignment",
    [
        "enabled, enabled = True, {value}\n",
        "[enabled, enabled] = [True, {value}]\n",
        "enabled, (enabled, enabled) = False, (True, {value})\n",
        "enabled = enabled, enabled = True, {value}\n",
    ],
)
def test_last_false_store_drops_earlier_truthy_value(assignment: str, false_value: str) -> None:
    assert not _scan(assignment.format(value=false_value))


@pytest.mark.parametrize(
    "assignment",
    [
        "enabled, enabled = False, True\n",
        "[enabled, enabled] = [False, True]\n",
        "enabled, (enabled, enabled) = False, (False, True)\n",
        "enabled = enabled, enabled = False, True\n",
    ],
)
def test_last_true_store_keeps_detection(assignment: str) -> None:
    findings = _scan(assignment)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


def test_direct_boolean_controls() -> None:
    assert len(_scan("")) == 1
    assert not _scan("enabled = False\n")
