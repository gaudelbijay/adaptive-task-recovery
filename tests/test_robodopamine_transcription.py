"""The Robo-Dopamine audit only means anything if it reproduces their metric.

These tests pin the three pieces transcribed from their released evaluation
code (eval/evaluation_grm.py): the score parser, the progress accumulator, and
the frame sampler. Each is checked against a literal restatement of their
implementation rather than against my expectation of what it does.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "rd_audit", Path(__file__).resolve().parents[1] / "scripts" / "audit_robodopamine_history.py")
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)


def their_parse(pred: str) -> float:
    """Verbatim from evaluation_grm.py:386."""
    return min(100, max(-100, float(
        pred.split("<score>")[-1].split("</score>")[0].replace('%', "").strip()))) / 100.0


def their_progress(preds, inverse):
    """Verbatim from evaluation_grm.py:382-410, including the in-place write."""
    data = [{"pred": p} for p in preds]
    if inverse:
        pre_prog = 1
        for item in data[::-1]:
            pred = their_parse(item["pred"])
            item["progress"] = (pre_prog + (1 - pre_prog) * pred if pred >= 0
                                else pre_prog + pre_prog * pred)
            pre_prog = item["progress"]
    else:
        pre_prog = 0
        for item in data:
            pred = their_parse(item["pred"])
            item["progress"] = (pre_prog + (1 - pre_prog) * pred if pred >= 0
                                else pre_prog + pre_prog * pred)
            pre_prog = item["progress"]
    return [d["progress"] for d in data]


def their_indices(num_frames, m):
    """Verbatim from evaluation_grm.py:53-62 plus the endpoint fixups at 222-226."""
    if num_frames < 1:
        return []
    if m <= 1:
        return [0]
    idx = [round(i * (num_frames - 1) / (m - 1)) for i in range(m)]
    if 0 not in idx:
        idx[0] = 0
    if (num_frames - 1) not in idx:
        idx[-1] = num_frames - 1
    return idx


@pytest.mark.parametrize("text,expected", [
    ("<score>+31.6%</score>", 0.316),
    ("<score>-100.0%</score>", -1.0),
    ("<score>+27.8%</score>", 0.278),
    ("<score>0%</score>", 0.0),
    ("<score>-45%</score>", -0.45),
    ("<score>+150%</score>", 1.0),      # their clip
])
def test_parse_matches_theirs(text, expected):
    assert rd.parse_score(text) == pytest.approx(expected)
    assert rd.parse_score(text) == pytest.approx(their_parse(text))


def test_parse_regression_decimal_scores():
    """The bug this replaced: a regex that returned the fractional digit.

    "-100.0%" came back as +0.0 and "+31.6%" as +0.06, so every score was small
    and positive and VOC read +1.0 regardless of direction.
    """
    assert rd.parse_score("<score>-100.0%</score>") == -1.0
    assert rd.parse_score("<score>+31.6%</score>") == pytest.approx(0.316)


@pytest.mark.parametrize("inverse", [False, True])
def test_progress_curve_matches_theirs(inverse):
    rng = np.random.default_rng(0)
    for _ in range(200):
        preds = [round(float(x), 1) for x in rng.uniform(-100, 100, rng.integers(2, 12))]
        texts = [f"<score>{p:+.1f}%</score>" for p in preds]
        mine = rd.progress_curve([rd.parse_score(t) for t in texts], inverse)
        assert mine == pytest.approx(their_progress(texts, inverse))


def test_progress_stays_bounded():
    """Their accumulator saturates in [0, 1]; a cumulative sum would not."""
    for inverse in (False, True):
        curve = rd.progress_curve([1.0] * 10 + [-1.0] * 10, inverse)
        assert min(curve) >= -1e-12 and max(curve) <= 1 + 1e-12


def test_inverse_curve_reads_ascending():
    """Both directions score near +1: inverse walks back from 1 but writes in place."""
    assert rd.voc(rd.progress_curve([-0.5] * 8, inverse=True)) == pytest.approx(1.0)
    assert rd.voc(rd.progress_curve([0.5] * 8, inverse=False)) == pytest.approx(1.0)


@pytest.mark.parametrize("n,interval", [(300, 30), (301, 30), (97, 30), (60, 30), (1000, 30)])
def test_sample_indices_match_theirs(n, interval):
    m = n // interval
    assert rd.make_sample_indices(n, m) == their_indices(n, m)


def test_sample_indices_reach_the_final_frame():
    """The stride version this replaced stopped short of the last frame."""
    idx = rd.make_sample_indices(300, 300 // 30)
    assert idx[0] == 0 and idx[-1] == 299
