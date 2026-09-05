#!/usr/bin/env python3
"""Audit a published claim: does Robo-Dopamine's history conditioning help?

Robo-Dopamine 2.0 argues that "existing learned visual reward models often rely
on static before-after observations, causing temporal ambiguity and weak
discrimination", and its remedy is a history-conditioned reward that adds a
reference panel: a REFERENCE START and REFERENCE END frame alongside the queried
BEFORE/AFTER sets.

That is a capability claim of exactly the shape the ladder audits, and it is
testable because the model, the benchmark and the evaluation code are released.
The matched control is the baseline they describe: the same released model, the
same queried endpoints, the same scoring instructions, with the reference panel
removed. If the panel is doing the work claimed, removing it should cost
accuracy on their own metric.

Fairness notes, since a sloppy ablation would prove nothing:
  * the prompt instructs the model to calibrate against the references, so the
    ablated condition removes those instructions too rather than leaving the
    model told to use anchors it cannot see;
  * both conditions score the identical episodes, frame pairs, and directions;
  * the metric is their own VOC, computed by their own function.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

# The reference panel is images[0:2]; the queried endpoints are images[2:8].
REFERENCE_SLICE = slice(0, 2)
QUERY_SLICE = slice(2, 8)

FULL_HEAD = (
    "\nYou are a rigorous, impartial vision evaluator for robot task progress. "
    "Your job is to judge whether the AFTER image set moves closer to the task "
    "objective than the BEFORE image set, using the provided reference examples "
    "only as anchors.\n\n<Task>\n`{task}`\n\nREFERENCE EXAMPLES (for visual "
    "anchoring only; not necessarily this run's actual START/END):\n"
    "- REFERENCE START — Robot Front Image (task just starting): "
)
ABLATED_HEAD = (
    "\nYou are a rigorous, impartial vision evaluator for robot task progress. "
    "Your job is to judge whether the AFTER image set moves closer to the task "
    "objective than the BEFORE image set.\n\n<Task>\n`{task}`\n</Task>\n\n"
)


def ablate_goal_text(goal: str) -> str:
    """Strip reference-panel instructions from the scoring block.

    Everything about how to score -- direction, normalization, the criteria, the
    output format -- is preserved verbatim. Only the calibration step that names
    the absent anchors is removed.
    """
    goal = goal.replace(
        ", using the REFERENCE START/END images as conceptual anchors", "")
    goal = re.sub(
        r"1\) Calibrate using the references:.*?2\) Direction:",
        "1) Direction:", goal, flags=re.S)
    goal = goal.replace("2) Direction:", "1) Direction:")
    goal = re.sub(r"\n\s*-\s*REFERENCE (START|END)[^\n]*\n", "\n", goal)
    goal = goal.replace(
        "   - For improvements, scale the improvement relative to what remained from BEFORE to END.",
        "   - For improvements, scale by how much closer AFTER is to the objective.")
    goal = goal.replace(
        "   - For regressions, scale the deterioration relative to how far BEFORE had progressed from START.",
        "   - For regressions, scale by how much further AFTER is from the objective.")
    return goal


def build_messages(task: str, goal_text: str, ablate: bool):
    """Reproduce their message layout, optionally without the reference panel."""
    labels = ["BEFORE Robot Front Image: ", "\nBEFORE Robot Left Wrist Image: ",
              "\nBEFORE Robot Right Wrist Image: ", "\n\nAFTER Robot Front Image: ",
              "\nAFTER Robot Left Wrist Image: ", "\nAFTER Robot Right Wrist Image: "]
    content = []
    if ablate:
        content.append({"type": "text", "text": ABLATED_HEAD.format(task=task)})
    else:
        content.append({"type": "text", "text": FULL_HEAD.format(task=task)})
        content.append({"type": "image"})
        content.append({"type": "text",
                        "text": "\n- REFERENCE END — Robot Front Image (task fully completed): "})
        content.append({"type": "image"})
        content.append({"type": "text", "text": "\n</Task>\n\n"})
    for i, label in enumerate(labels):
        content.append({"type": "text", "text": label if i or ablate else label})
        content.append({"type": "image"})
    content.append({"type": "text", "text": goal_text})
    return [{"role": "user", "content": content}]


def make_sample_indices(num_frames: int, m: int):
    """Their frame sampler: m evenly spaced indices, inclusive endpoints."""
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


def parse_score(text: str) -> float:
    """Their parser: split on the tags, float the remainder, clip to +/-100%.

    Transcribed rather than re-expressed as a regex. An earlier regex here
    required digits flush against the percent sign, so it fell through to a
    loose fallback that matched the *fractional* digit: "-100.0%" parsed as
    +0.0 and "+31.6%" as +0.06. Every score became small and positive, which
    made VOC read +1.0 in both directions and looked like a model result.
    """
    raw = text.split("<score>")[-1].split("</score>")[0].replace("%", "").strip()
    return min(100.0, max(-100.0, float(raw))) / 100.0


def progress_curve(preds, inverse: bool):
    """Their saturating accumulator, written back in original order.

    Forward starts at 0 and walks forward. Inverse starts at 1 and walks
    backward, but assigns in place, so the stored curve is still read back
    ascending -- a well-behaved model scores VOC near +1 in *both* directions.
    This is not a cumulative sum; values stay bounded in [0, 1].
    """
    prog = [0.0] * len(preds)
    if inverse:
        pre = 1.0
        for i in range(len(preds) - 1, -1, -1):
            p = preds[i]
            pre = pre + (1.0 - pre) * p if p >= 0 else pre + pre * p
            prog[i] = pre
    else:
        pre = 0.0
        for i in range(len(preds)):
            p = preds[i]
            pre = pre + (1.0 - pre) * p if p >= 0 else pre + pre * p
            prog[i] = pre
    return prog


def voc(progress) -> float:
    """Their metric: Spearman correlation between time index and progress."""
    values = np.asarray(progress, dtype=float)
    if len(values) < 2:
        return float("nan")
    corr, _ = spearmanr(np.arange(len(values)), values)
    return float(corr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--bench", required=True, help="Robo-Dopamine-Bench root.")
    parser.add_argument("--output", default="results/robodopamine/history_audit.json")
    parser.add_argument("--domains", nargs="+", default=[
        "agibot", "droid_oxe", "galaxea_r1lite", "human_egodex", "libero", "robocasa"])
    parser.add_argument("--episodes", type=int, default=25,
                        help="Episodes per domain; the full set is 100.")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--gpu-fraction", type=float, default=0.85)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    llm = LLM(model=args.model, limit_mm_per_prompt={"image": args.max_images},
              gpu_memory_utilization=args.gpu_fraction, trust_remote_code=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=64)

    # The scoring block is identical across conditions apart from the calibration
    # step, so it is lifted from their source rather than paraphrased.
    source = Path(args.bench).parent / "Robo-Dopamine" / "eval" / "evaluation_grm.py"
    text = source.read_text()
    start = text.index("\\n\\nGoal\\nCompare the BEFORE and AFTER")
    end = text.index('"},', start)
    goal_full = text[start:end].encode().decode("unicode_escape")
    goal_ablated = ablate_goal_text(goal_full)

    root = Path(args.bench)
    dir_for = {"agibot": "agibotworld", "droid_oxe": "droid_oxe",
               "galaxea_r1lite": "galaxea_r1lite", "human_egodex": "human_egodex",
               "libero": "libero_data", "robocasa": "robocasa_data"}

    records, skipped = [], []
    for domain in args.domains:
        meta = json.loads((root / "jsons" / f"{domain}_test_100.json").read_text())
        pairs = list(zip(meta["sample_path_list"], meta["sample_path_task"]))[: args.episodes]
        for episode, task in pairs:
            base = root / "images" / episode.split("/", 1)[0] / episode.split("/", 1)[1] \
                if "/" in episode else root / "images" / episode
            high = base / "cam_high"
            if not high.is_dir():
                continue
            frames = sorted(high.glob("frame_*.jpg")) or sorted(high.glob("frame_*.png"))
            if len(frames) < 2 * args.interval:
                continue
            wrist_l = base / ("cam_high" if "egodex" in episode else "cam_left_wrist")
            wrist_r = base / ("cam_high" if "egodex" in episode else "cam_right_wrist")
            # Their process_videos asserts the three streams have equal frame
            # counts. Four agibot episodes in the official test list ship with
            # cam_high only -- verified absent in the upstream archive, not a
            # partial download -- so their own code cannot evaluate them either.
            # Skip and account for them rather than padding with repeated frames.
            counts = [len(list(d.glob("frame_*"))) for d in (high, wrist_l, wrist_r)]
            if len(set(counts)) != 1:
                skipped.append({"domain": domain, "episode": episode,
                                "stream_frame_counts": counts})
                continue
            picks = make_sample_indices(len(frames), len(frames) // args.interval)
            if len(picks) < 2:
                continue
            last = len(frames) - 1

            def frame(directory: Path, i: int):
                cands = sorted(directory.glob("frame_*"))
                return Image.open(cands[min(i, len(cands) - 1)]).convert("RGB")

            for inverse in (False, True):
                for ablate in (False, True):
                    # Steps are independent -- progress is accumulated afterwards --
                    # so the whole episode is one batched generate call.
                    reqs = []
                    for k in range(len(picks) - 1):
                        bf, af = picks[k], picks[k + 1]
                        if inverse:
                            bf, af = af, bf
                        images = [frame(high, 0), frame(high, last),
                                  frame(high, bf), frame(wrist_l, bf), frame(wrist_r, bf),
                                  frame(high, af), frame(wrist_l, af), frame(wrist_r, af)]
                        used = images[QUERY_SLICE] if ablate else images
                        messages = build_messages(task, goal_ablated if ablate else goal_full,
                                                  ablate)
                        prompt = processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True)
                        reqs.append({"prompt": prompt, "multi_modal_data": {"image": used}})
                    outs = llm.generate(reqs, sampling, use_tqdm=False)
                    scores, raw, failures = [], [], 0
                    for o in outs:
                        text = o.outputs[0].text
                        try:
                            scores.append(parse_score(text))
                        except Exception:
                            scores.append(0.0)
                            failures += 1
                        raw.append(text.strip()[:60])
                    records.append({
                        "domain": domain, "episode": episode, "inverse": inverse,
                        "condition": "ablated" if ablate else "full",
                        "voc": voc(progress_curve(scores, inverse)),
                        "steps": len(scores), "parse_failures": failures,
                        "scores": scores,
                        "progress": progress_curve(scores, inverse),
                        "raw": raw,
                    })
            print(f"{domain:<16} {episode:<28} done", flush=True)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema_version": 1,
        "claim_under_test": (
            "Robo-Dopamine 2.0 attributes its gain to history conditioning via a "
            "reference panel, against static before-after baselines."
        ),
        "control": "same released model and queried endpoints, reference panel removed",
        "model": args.model, "episodes_per_domain": args.episodes,
        "skipped_incomplete_streams": skipped,
        "records": records,
    }, indent=2) + "\n")
    print(f"wrote {out}  ({len(records)} records)")


if __name__ == "__main__":
    main()
