#!/usr/bin/env python3
"""Render the shortcut-ladder figure used in the README.

Two panels, both reading only committed result artifacts:

  A. Held-out-mechanism accuracy by control rung, for both benchmarks. A lower
     rung matching the top rung means the held-out mechanism is a shortcut.
  B. The permanent/temporary confusion pair closed-loop, where the two
     non-recurrent arms fail in opposite directions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Reference categorical palette, slots 1 and 2, in fixed order.
SERIES_1 = "#2a78d6"   # blue
SERIES_2 = "#eb6834"   # orange
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dedcd5"

RUNGS = [
    ("instantaneous", "1  current frame only"),
    ("one_past_frame", "2  one past frame"),
    ("hand_written", "3  hand-written rule *"),
    ("recurrent_factorized", "4  recurrent (factorized)"),
]


def load_ladder(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text())
    return {
        key: entry["heldout_option_accuracy_mean"]
        for key, entry in report["rungs"].items()
        if entry["heldout_option_accuracy_mean"] is not None
    }


def panel_a(ax, v4: dict[str, float], peg: dict[str, float]) -> None:
    labels = [label for _, label in RUNGS]
    y = range(len(RUNGS))
    height = 0.36
    gap = 0.02  # 2px-equivalent surface gap between adjacent bars
    v4_values = [v4.get(key, 0.0) for key, _ in RUNGS]
    peg_values = [peg.get(key, 0.0) for key, _ in RUNGS]

    ax.barh([i + (height + gap) / 2 for i in y], v4_values, height=height,
            color=SERIES_1, label="LearnedRecovery-v4", zorder=3)
    ax.barh([i - (height + gap) / 2 for i in y], peg_values, height=height,
            color=SERIES_2, label="PegInsertionSide-v1", zorder=3)

    for i, (a, b) in enumerate(zip(v4_values, peg_values)):
        ax.text(a + 0.015, i + (height + gap) / 2, f"{a:.2f}", va="center",
                ha="left", fontsize=8.5, color=INK)
        ax.text(b + 0.015, i - (height + gap) / 2, f"{b:.2f}", va="center",
                ha="left", fontsize=8.5, color=INK)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlim(0, 1.14)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlabel("held-out mechanism accuracy", fontsize=9, color=INK_MUTED)
    ax.set_title(
        "A.  A lower rung matching rung 4 means the held-out\n"
        "     mechanism is a shortcut, not a composition result",
        fontsize=10, color=INK, loc="left", pad=34,
    )


def panel_b(ax, arms: list[tuple[str, float, float]]) -> None:
    labels = [name for name, _, _ in arms]
    x = range(len(arms))
    width = 0.36
    gap = 0.02
    permanent = [p for _, p, _ in arms]
    temporary = [t for _, _, t in arms]

    ax.bar([i - (width + gap) / 2 for i in x], permanent, width=width,
           color=SERIES_1, label="permanent obstruction", zorder=3)
    ax.bar([i + (width + gap) / 2 for i in x], temporary, width=width,
           color=SERIES_2, label="temporary obstruction", zorder=3)

    for i, (p, t) in enumerate(zip(permanent, temporary)):
        ax.text(i - (width + gap) / 2, p + 0.02, f"{p:.2f}", ha="center",
                va="bottom", fontsize=8.5, color=INK)
        ax.text(i + (width + gap) / 2, t + 0.02, f"{t:.2f}", ha="center",
                va="bottom", fontsize=8.5, color=INK)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_ylim(0, 1.16)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_ylabel("closed-loop safe recovery", fontsize=9, color=INK_MUTED)
    ax.set_title(
        "B.  Both non-recurrent arms fail the confusion pair\n"
        "     in opposite directions; both recurrent arms solve it",
        fontsize=10, color=INK, loc="left", pad=34,
    )


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder-dir", type=Path, default=Path("results/router/ladder"))
    parser.add_argument(
        "--comparison", type=Path,
        default=Path("results/router/matched_router_comparison_347M.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("media/results/shortcut-ladder"))
    args = parser.parse_args()

    v4 = load_ladder(args.ladder_dir / "learned_recovery_v4.json")
    peg = load_ladder(args.ladder_dir / "peg_insertion.json")
    comparison = json.loads(args.comparison.read_text())["arms"]

    def cell(arm: str, condition: str) -> float:
        return comparison[arm]["conditions"][condition]["safe_success_rate"]

    arms = [
        ("factorized\nGRU", cell("causal_gru_matched", "permanent_block"),
         cell("causal_gru_matched", "temporary_block")),
        ("unstructured\nGRU", cell("unstructured_gru", "permanent_block"),
         cell("unstructured_gru", "temporary_block")),
        ("hand-written\nrule", cell("heuristic_v28", "permanent_block"),
         cell("heuristic_v28", "temporary_block")),
        ("one past frame\n(no memory)", cell("static_offset", "permanent_block"),
         cell("static_offset", "temporary_block")),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5), facecolor=SURFACE)
    fig.subplots_adjust(wspace=0.30, top=0.80, bottom=0.20)
    panel_a(axes[0], v4, peg)
    panel_b(axes[1], arms)
    for ax in axes:
        style(ax)
    axes[0].xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    axes[1].yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    axes[0].set_axisbelow(True)
    axes[1].set_axisbelow(True)
    for ax in axes:
        ax.legend(
            frameon=False, fontsize=8.5, labelcolor=INK_MUTED, ncol=2,
            loc="lower left", bbox_to_anchor=(0.0, 1.005), borderaxespad=0.0,
            handlelength=1.2, columnspacing=1.6,
        )

    fig.text(
        0.085, -0.02,
        "* the hand-written rule never emits \u201cdefer\u201d, so its offline score is not comparable to the "
        "learned rungs; closed-loop it reaches 0.97 on v4.",
        fontsize=7.5, color=INK_MUTED, va="top", ha="left",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(f"{args.output}.{suffix}", dpi=200, bbox_inches="tight",
                    facecolor=SURFACE)
    print(f"wrote {args.output}.png and {args.output}.pdf")


if __name__ == "__main__":
    main()
