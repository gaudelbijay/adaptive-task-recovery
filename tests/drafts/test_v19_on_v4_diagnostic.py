"""Static checks for the V19-on-V4 bottleneck diagnostic."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "scripts/evaluate_v19_on_v4.py").read_text(encoding="utf-8")


def test_diagnostic_crosses_all_seeds_conditions_and_progress_sources():
    assert "SEEDS = (9351, 4796, 1788)" in SCRIPT
    assert '"reverse_ejection",' in SCRIPT
    assert 'PROGRESS_SOURCES = ("normal", "oracle", "oracle_defer")' in SCRIPT


def test_oracle_changes_only_progress_interface():
    assert 'progress = predicted if source == "normal" else resolved.float()' in SCRIPT
    assert "agent.actor(torch.cat((latent, proprio, progress), dim=1))" in SCRIPT
    assert 'if progress_source == "oracle_defer"' in SCRIPT
