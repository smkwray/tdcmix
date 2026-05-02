from pathlib import Path

from tdcmix.contracts import load_source_contracts
from tdcmix.panel import build_holder_panel
from tdcmix.validation import (
    build_residual_gate_report,
    build_residual_gate_review,
    build_panel_validation_summary,
    build_exact_residual_overlap_diagnostics,
    build_residual_diagnostics,
    build_scale_reconciliation,
    build_share_sum_diagnostics,
    build_wam_no_lookahead_diagnostics,
)
from tests.test_panel import _mix_config


def _panel():
    return build_holder_panel(load_source_contracts(Path("config/source_contracts.yml")), _mix_config())


def test_validation_summary_passes_core_checks() -> None:
    summary = build_panel_validation_summary(_panel())

    checks = dict(zip(summary["check"], summary["passes"], strict=True))
    assert checks["share_sums_pass"]
    assert checks["residual_available_rows"]


def test_share_sum_diagnostics_are_bounded() -> None:
    diagnostics = build_share_sum_diagnostics(_panel())
    defined = diagnostics["share_sum"].notna()

    assert defined.any()
    assert diagnostics.loc[defined, "passes"].all()


def test_residual_diagnostics_include_positive_residual_share() -> None:
    diagnostics = build_residual_diagnostics(_panel())

    assert "positive_residual_share" in diagnostics.columns
    assert diagnostics["residual_available"].any()


def test_scale_reconciliation_documents_tdcest_overlap() -> None:
    reconciliation = build_scale_reconciliation(_panel())

    assert set(reconciliation["bucket"]) == {"fed", "banks", "row"}
    assert (reconciliation["overlap_rows"] > 0).all()


def test_wam_no_lookahead_diagnostics_pass() -> None:
    diagnostics = build_wam_no_lookahead_diagnostics(_panel())

    assert diagnostics["passes"].all()


def test_exact_residual_overlap_diagnostics_available() -> None:
    diagnostics = build_exact_residual_overlap_diagnostics(_panel())

    assert diagnostics["exact_residual"].notna().any()
    assert "residual_proxy_error" in diagnostics.columns


def test_residual_gate_review_blocks_headline_use() -> None:
    panel = _panel()
    residual = build_residual_diagnostics(panel)
    exact = build_exact_residual_overlap_diagnostics(panel)

    gate = build_residual_gate_review(residual, exact)
    report = build_residual_gate_report(gate)

    row = gate.iloc[0]
    assert row["status"] == "blocked"
    assert row["headline_use"] == "blocked"
    assert row["exact_overlap_rows"] < row["min_exact_overlap_rows"]
    assert "Residual-sensitive contrasts must remain blocked" in report
