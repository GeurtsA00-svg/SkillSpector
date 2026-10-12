# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-comment ownership regressions, using independent synthetic text."""

from __future__ import annotations

import pytest

from skillspector.models import compute_match_fingerprint
from skillspector.nodes.analyzers import static_patterns_prompt_injection as prompt_injection
from skillspector.nodes.analyzers import static_runner


def _p2(content: str):
    return [
        finding
        for finding in prompt_injection.analyze(content, "SKILL.md", "markdown")
        if finding.rule_id == "P2"
    ]


@pytest.mark.parametrize(
    "visible",
    [
        "Then target the endpoint (see below).",
        "GET the page (documented here).",
        "Follow the visible instructions (documented here).",
        "Use the operating system (documented here).",
        "Send feedback (documented here).",
        "Transmit the visible message (documented here).",
        "Ignore this visible example (documented here).",
    ],
)
def test_closed_comment_cannot_borrow_visible_keyword(visible: str) -> None:
    assert not _p2(f"[//]: # (layout note)\n\n{visible}")


def test_separate_comments_only_own_their_own_title() -> None:
    hidden = "[//]: # (GET https://example.invalid/collect)"
    content = f"[//]: # (layout note)\nVisible text.\n{hidden}\n[//]: # (end note)"
    findings = _p2(content)
    assert len(findings) == 1
    assert findings[0].matched_text == hidden
    assert findings[0].match_fingerprint == compute_match_fingerprint("P2", hidden)
    assert findings[0].location.start_line == 3
    assert findings[0].location.end_line == 3


@pytest.mark.parametrize(
    "title",
    ["target", "forget", "GETTING", "POSTGRES", "SYSTEMATIC", "sender", "transmitter"],
)
def test_title_keywords_do_not_match_ordinary_word_substrings(title: str) -> None:
    assert not _p2(f"[//]: # ({title})")


@pytest.mark.parametrize(
    "title",
    [
        "SYSTEM: do something",
        "instructions: do something",
        "ignore_previous_instructions",
        "ignorePreviousInstructions",
        "pleaseIgnorePreviousInstructions",
        "POSTRequestToServer",
        "SYSTEMPrompt",
        "GET https://example.invalid/data",
        "send the message",
        "transmit the message",
    ],
)
def test_real_hidden_keywords_keep_detection(title: str) -> None:
    comment = f"[//]: # ({title})"
    findings = _p2(comment)
    assert len(findings) == 1
    assert findings[0].matched_text == comment
    assert findings[0].match_fingerprint == compute_match_fingerprint("P2", comment)


@pytest.mark.parametrize(
    "comment",
    [
        r"[//]: # (layout \(small\) note)",
        r"[//]: # (layout \) note)",
        r"[//]: # (layout \\)",
        r"[//]: # (layout \\\) note)",
    ],
)
def test_escaped_parentheses_and_backslashes_still_bound_visible_text(comment: str) -> None:
    assert not _p2(f"{comment}\n\nGET a visible page (see below).")


@pytest.mark.parametrize(
    "comment",
    [
        r"[//]: # (layout \(small\) GET the page)",
        r"[//]: # (layout \) GET the page)",
        r"[//]: # (layout \\\) GET the page)",
        "[//]: # (layout\nGET the page)",
    ],
)
def test_escaped_closer_does_not_hide_later_keyword_inside_title(comment: str) -> None:
    findings = _p2(comment)
    assert len(findings) == 1
    assert findings[0].matched_text == comment
    assert findings[0].match_fingerprint == compute_match_fingerprint("P2", comment)


@pytest.mark.parametrize(
    "content",
    [
        "[//]: # (layout (nested) note)\nGET a visible page (see below).",
        "[//]: # (layout note",
    ],
)
def test_malformed_titles_do_not_borrow_visible_keywords(content: str) -> None:
    assert not _p2(content)


@pytest.mark.parametrize(
    "content",
    [
        "[//]: # (GET a page (nested) note)",
        "[//]: # (GET a page\n\nextra text)",
    ],
)
def test_malformed_titles_keep_conservative_hidden_keyword_detection(content: str) -> None:
    assert _p2(content)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("［//］： ＃ （layout note）\n\nThen target the endpoint (see below).", False),
        ("［//］： ＃ （ＧＥＴ https://example.invalid/data）", True),
        ("[//]: # (layout note&#41;\n\nGET a visible page (see below).", False),
    ],
)
def test_transformed_views_keep_comment_ownership(content: str, expected: bool) -> None:
    state = {"components": ["SKILL.md"], "file_cache": {"SKILL.md": content}}
    findings = static_runner.run_static_patterns(state, [prompt_injection])
    p2 = [finding for finding in findings if finding.rule_id == "P2"]
    assert bool(p2) is expected
    if expected:
        assert len(p2) == 1
        assert p2[0].start_line == 1
