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
    ("instantaneous", "1  current frame"),
    ("one_past_frame", "2  one past frame"),
    ("moment_summary", "\u2013  prefix summary"),
    ("hand_written", "3  hand-written rule *"),
    ("recurrent_factorized", "4  recurrent"),
]


def load_ladder(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text())
    return {
        key: entry["heldout_option_accuracy_mean"]
        for key, entry in report["rungs"].items()
        if entry["heldout_option_accuracy_mean"] is not None
    }


def panel_a(axes, v4, peg, reboot):
    """Three small multiples: the same question asked of three benchmarks.

    The simulated benchmarks are scored as held-out-option accuracy (floor 0);
    REBOOT is macro-AUROC (chance 0.5). They get separate axes rather than a
    shared one, because the two metrics have different floors.
    """
    panels = [
        ("LearnedRecovery-v4", v4, "held-out accuracy", None, SERIES_1),
        ("PegInsertionSide-v1", peg, "held-out accuracy", None, SERIES_1),
        ("REBOOT (real robot)", reboot, "macro-AUROC", 0.5, SERIES_2),
    ]
    for ax, (title, values, xlabel, chance, colour) in zip(axes, panels):
        rows = [(lab, values.get(key)) for key, lab in RUNGS if key in values]
        y = range(len(rows))
        ax.barh(list(y), [v for _, v in rows], height=0.55, color=colour, zorder=3)
        for i, (_, v) in enumerate(rows):
            ax.text(v + 0.02, i, f"{v:.2f}", va="center", ha="left",
                    fontsize=8.5, color=INK)
        if chance is not None:
            ax.axvline(chance, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=4)
            ax.set_ylim(-0.6, len(rows) - 0.15)
            ax.text(chance + 0.012, len(rows) - 0.42, "chance", fontsize=7.5,
                    color=INK_MUTED, va="center", ha="left")
        ax.set_yticks(list(y))
        ax.set_yticklabels([lab for lab, _ in rows], fontsize=8.5, color=INK)
        ax.set_xlim(0, 1.18)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_xlabel(xlabel, fontsize=8.5, color=INK_MUTED)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=8)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)


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
        "B.  What memory is for: both non-recurrent arms fail\n"
        "     the confusion pair, in opposite directions",
        fontsize=10.5, color=INK, loc="left", pad=46,
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
        ("ours\n(factorized)", cell("causal_gru_matched", "permanent_block"),
         cell("causal_gru_matched", "temporary_block")),
        ("unstructured\nGRU", cell("unstructured_gru", "permanent_block"),
         cell("unstructured_gru", "temporary_block")),
        ("motion\nrule", cell("heuristic_v28", "permanent_block"),
         cell("heuristic_v28", "temporary_block")),
        ("one\nframe", cell("static_offset", "permanent_block"),
         cell("static_offset", "temporary_block")),
    ]

    reboot_path = args.ladder_dir.parent.parent / "a_plus_audit" / "reboot_ladder_v4_aggregate.json"
    reboot = {}
    if reboot_path.exists():
        agg = json.loads(reboot_path.read_text())["aggregate"]
        reboot = {
            "instantaneous": agg["static_mlp"]["macro_auroc_mean"],
            "one_past_frame": agg["endpoint_pair_mlp"]["macro_auroc_mean"],
            "moment_summary": agg["moment_mlp"]["macro_auroc_mean"],
            "recurrent_factorized": agg["causal_dynamics_gru"]["macro_auroc_mean"],
        }

    fig = plt.figure(figsize=(16.0, 4.8), facecolor=SURFACE)
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.3], wspace=0.75)
    ladder_axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    confusion_ax = fig.add_subplot(grid[0, 3])
    panel_a(ladder_axes, v4, peg, reboot)
    panel_b(confusion_ax, arms)
    for ax in ladder_axes + [confusion_ax]:
        style(ax)
    confusion_ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    confusion_ax.set_axisbelow(True)
    confusion_ax.legend(
        frameon=False, fontsize=8.5, labelcolor=INK_MUTED, loc="lower left", bbox_to_anchor=(0.0, 1.005), borderaxespad=0.0,
        handlelength=1.2, ncol=2, columnspacing=1.4,
    )
    fig.suptitle(
        "A.  Is the held-out mechanism a shortcut?  A lower rung matching rung 4 says yes.",
        fontsize=10.5, color=INK, x=0.055, ha="left", y=1.045,
    )
    fig.text(
        0.055, -0.06,
        "* the hand-written rule never emits \u201cdefer\u201d, so its offline score is not comparable to the learned rungs; "
        "closed-loop it reaches 0.97 on LearnedRecovery-v4.",
        fontsize=7.5, color=INK_MUTED, va="top", ha="left",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(f"{args.output}.{suffix}", dpi=200, bbox_inches="tight",
                    facecolor=SURFACE)
    print(f"wrote {args.output}.png and {args.output}.pdf")


if __name__ == "__main__":
    main()
