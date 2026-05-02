from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from tdcmix.audits import (
    write_sector_map_audit,
    write_source_overlap_audit,
    write_source_upgrade_recommendations,
)
from tdcmix.contracts import load_source_contracts
from tdcmix.decision import write_project_closeout_outputs, write_project_decision_outputs
from tdcmix.design import write_design_outputs
from tdcmix.downstream import write_downstream_contract_bundle_outputs
from tdcmix.estimates import write_estimate_outputs
from tdcmix.exact_primary_analysis import write_exact_primary_analysis_outputs
from tdcmix.figures import (
    write_lp_diagnostic_profile_figure,
    write_stacked_holder_absorption_figure,
)
from tdcmix.manifest import write_artifact_manifest_outputs
from tdcmix.panel import (
    write_exact_fu_primary_panel_outputs,
    write_holder_panel,
    write_holder_panel_skeleton,
)
from tdcmix.priors import write_holder_absorption_priors
from tdcmix.schema import load_sector_map
from tdcmix.validation import write_validation_outputs
from tdcmix.z1_exact_analysis import write_z1_exact_analysis_outputs
from tdcmix.z1_exact import write_z1_exact_outputs


DEFAULT_CONTRACTS = Path("config/source_contracts.yml")
DEFAULT_SECTOR_MAP = Path("config/holder_sector_map.yml")
DEFAULT_MIX_VARIABLES = Path("config/holder_mix_variables.yml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tdcmix")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-contracts")
    validate.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)

    audits = subparsers.add_parser("write-audits")
    audits.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    audits.add_argument("--sector-map", type=Path, default=DEFAULT_SECTOR_MAP)
    audits.add_argument("--project-root", type=Path, default=Path("."))
    audits.add_argument("--z1-candidate-output", type=Path, default=Path("data/processed/z1_candidate_audit.csv"))
    audits.add_argument(
        "--source-upgrade-output",
        type=Path,
        default=Path("data/processed/source_upgrade_recommendations.csv"),
    )
    audits.add_argument(
        "--source-upgrade-report",
        type=Path,
        default=Path("data/processed/source_upgrade_recommendations.md"),
    )

    skeleton = subparsers.add_parser("write-panel-skeleton")
    skeleton.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    skeleton.add_argument("--mix-variables", type=Path, default=DEFAULT_MIX_VARIABLES)
    skeleton.add_argument("--quarter", action="append", default=[])

    panel = subparsers.add_parser("write-panel")
    panel.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    panel.add_argument("--mix-variables", type=Path, default=DEFAULT_MIX_VARIABLES)
    panel.add_argument("--project-root", type=Path, default=Path("."))

    exact_primary_panel = subparsers.add_parser("write-exact-primary-panel")
    exact_primary_panel.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    exact_primary_panel.add_argument("--mix-variables", type=Path, default=DEFAULT_MIX_VARIABLES)
    exact_primary_panel.add_argument(
        "--legacy-panel",
        type=Path,
        default=Path("data/processed/holder_absorption_panel.csv"),
    )
    exact_primary_panel.add_argument(
        "--exact-sidecar-panel",
        type=Path,
        default=Path("data/processed/z1_exact_holder_absorption_panel.csv"),
    )
    exact_primary_panel.add_argument(
        "--migration-gate",
        type=Path,
        default=Path("data/processed/z1_exact_primary_migration_gate.csv"),
    )
    exact_primary_panel.add_argument("--output-dir", type=Path, default=Path("data/processed"))

    exact_primary_analysis = subparsers.add_parser("write-exact-primary-analysis")
    exact_primary_analysis.add_argument(
        "--panel",
        type=Path,
        default=Path("data/processed/holder_absorption_panel_exact_primary.csv"),
    )
    exact_primary_analysis.add_argument(
        "--migration-gate",
        type=Path,
        default=Path("data/processed/z1_exact_primary_migration_gate.csv"),
    )
    exact_primary_analysis.add_argument(
        "--exact-sidecar-analysis-dir",
        type=Path,
        default=Path("data/processed/z1_exact_analysis"),
    )
    exact_primary_analysis.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/exact_primary_analysis"),
    )

    validation = subparsers.add_parser("write-validation")
    validation.add_argument("--panel", type=Path, default=Path("data/processed/holder_absorption_panel.csv"))
    validation.add_argument("--output-dir", type=Path, default=Path("data/processed"))

    figure = subparsers.add_parser("write-figures")
    figure.add_argument("--panel", type=Path, default=Path("data/processed/holder_absorption_panel.csv"))
    figure.add_argument("--output", type=Path, default=Path("output/holder_absorption_shares.png"))
    figure.add_argument("--start", default="1990-01-01")

    design = subparsers.add_parser("write-design")
    design.add_argument("--panel", type=Path, default=Path("data/processed/holder_absorption_panel.csv"))
    design.add_argument("--validation-summary", type=Path, default=Path("data/processed/validation_summary.csv"))
    design.add_argument("--exact-residual-overlap", type=Path, default=Path("data/processed/exact_residual_overlap_diagnostics.csv"))
    design.add_argument("--output-dir", type=Path, default=Path("data/processed"))

    estimates = subparsers.add_parser("write-estimates")
    estimates.add_argument("--matrices", type=Path, default=Path("data/processed/horizon_design_matrices.csv"))
    estimates.add_argument("--stop-rule-status", type=Path, default=Path("data/processed/stop_rule_status.csv"))
    estimates.add_argument("--pretrend-matrices", type=Path, default=None)
    estimates.add_argument("--output-dir", type=Path, default=Path("data/processed"))

    estimate_figures = subparsers.add_parser("write-estimate-figures")
    estimate_figures.add_argument("--profile-plot-data", type=Path, default=Path("data/processed/lp_diagnostic_profile_plot_data.csv"))
    estimate_figures.add_argument("--output", type=Path, default=Path("output/lp_diagnostic_profiles.png"))

    priors = subparsers.add_parser("write-priors")
    priors.add_argument("--panel", type=Path, default=Path("data/processed/holder_absorption_panel.csv"))
    priors.add_argument("--sector-map", type=Path, default=DEFAULT_SECTOR_MAP)
    priors.add_argument("--output-dir", type=Path, default=Path("data/processed"))

    decision = subparsers.add_parser("write-decision")
    decision.add_argument("--validation-summary", type=Path, default=Path("data/processed/validation_summary.csv"))
    decision.add_argument("--design-readiness", type=Path, default=Path("data/processed/design_readiness.csv"))
    decision.add_argument("--stop-rule-status", type=Path, default=Path("data/processed/stop_rule_status.csv"))
    decision.add_argument("--stability-summary", type=Path, default=Path("data/processed/lp_shortlist_stability_summary.csv"))
    decision.add_argument("--economics-review", type=Path, default=Path("data/processed/lp_shortlist_economics_review.csv"))
    decision.add_argument("--output-dir", type=Path, default=Path("data/processed"))

    closeout = subparsers.add_parser("write-closeout")
    closeout.add_argument("--project-decision", type=Path, default=Path("data/processed/project_decision_status.csv"))
    closeout.add_argument(
        "--exact-primary-analysis-status",
        type=Path,
        default=Path("data/processed/exact_primary_analysis/exact_primary_analysis_status.csv"),
    )
    closeout.add_argument(
        "--exact-primary-artifact-comparison",
        type=Path,
        default=Path("data/processed/exact_primary_analysis/exact_primary_artifact_comparison.csv"),
    )
    closeout.add_argument(
        "--exact-primary-migration-comparison",
        type=Path,
        default=Path("data/processed/exact_primary_migration_comparison.csv"),
    )
    closeout.add_argument(
        "--primary-migration-gate",
        type=Path,
        default=Path("data/processed/z1_exact_primary_migration_gate.csv"),
    )
    closeout.add_argument(
        "--appendix-source-contract",
        type=Path,
        default=Path("data/processed/z1_exact_analysis/exact_appendix_source_contract.csv"),
    )
    closeout.add_argument(
        "--final-status-packet",
        type=Path,
        default=Path("data/processed/z1_exact_analysis/exact_final_status_packet.csv"),
    )
    closeout.add_argument("--output-dir", type=Path, default=Path("data/processed"))

    manifest = subparsers.add_parser("write-manifest")
    manifest.add_argument("--project-root", type=Path, default=Path("."))
    manifest.add_argument("--output-dir", type=Path, default=Path("data/processed"))

    downstream = subparsers.add_parser("write-downstream-contracts")
    downstream.add_argument(
        "--prior-contract",
        type=Path,
        default=Path("data/processed/holder_absorption_prior_contract.csv"),
    )
    downstream.add_argument(
        "--appendix-source-contract",
        type=Path,
        default=Path("data/processed/z1_exact_analysis/exact_appendix_source_contract.csv"),
    )
    downstream.add_argument(
        "--final-status-packet",
        type=Path,
        default=Path("data/processed/z1_exact_analysis/exact_final_status_packet.csv"),
    )
    downstream.add_argument(
        "--bucket-definition-contract",
        type=Path,
        default=Path("data/processed/z1_exact_bucket_definition_contract.csv"),
    )
    downstream.add_argument(
        "--bucket-migration-policy",
        type=Path,
        default=Path("data/processed/z1_exact_bucket_migration_policy.csv"),
    )
    downstream.add_argument(
        "--primary-migration-gate",
        type=Path,
        default=Path("data/processed/z1_exact_primary_migration_gate.csv"),
    )
    downstream.add_argument(
        "--default-source-decision",
        type=Path,
        default=Path("data/processed/downstream_default_source_decision.csv"),
    )
    downstream.add_argument("--output-dir", type=Path, default=Path("data/processed"))

    z1_exact = subparsers.add_parser("write-z1-exact")
    z1_exact.add_argument("--config", type=Path, default=Path("config/z1_exact_transaction_components.yml"))
    z1_exact.add_argument("--holder-panel", type=Path, default=Path("data/processed/holder_absorption_panel.csv"))
    z1_exact.add_argument("--output-dir", type=Path, default=Path("data/processed"))

    z1_exact_analysis = subparsers.add_parser("write-z1-exact-analysis")
    z1_exact_analysis.add_argument(
        "--exact-holder-panel",
        type=Path,
        default=Path("data/processed/z1_exact_holder_absorption_panel.csv"),
    )
    z1_exact_analysis.add_argument("--legacy-design", type=Path, default=Path("data/processed/design_panel.csv"))
    z1_exact_analysis.add_argument(
        "--legacy-summary",
        type=Path,
        default=Path("data/processed/lp_diagnostic_summary.csv"),
    )
    z1_exact_analysis.add_argument(
        "--legacy-stability",
        type=Path,
        default=Path("data/processed/lp_shortlist_stability_summary.csv"),
    )
    z1_exact_analysis.add_argument(
        "--exact-gate",
        type=Path,
        default=Path("data/processed/z1_exact_gate_review.csv"),
    )
    z1_exact_analysis.add_argument("--output-dir", type=Path, default=Path("data/processed/z1_exact_analysis"))

    args = parser.parse_args(argv)
    if args.command == "validate-contracts":
        load_source_contracts(args.contracts)
        return 0
    if args.command == "write-audits":
        contracts = load_source_contracts(args.contracts)
        sector_map = load_sector_map(args.sector_map)
        panel = contracts["panel"]
        write_sector_map_audit(
            sector_map,
            contracts,
            panel["sector_map_audit_path"],
        )
        write_source_overlap_audit(
            contracts,
            panel["source_overlap_audit_path"],
            project_root=args.project_root,
        )
        write_source_upgrade_recommendations(
            args.z1_candidate_output,
            args.source_upgrade_output,
            args.source_upgrade_report,
            contracts,
            project_root=args.project_root,
        )
        return 0
    if args.command == "write-panel-skeleton":
        contracts = load_source_contracts(args.contracts)
        mix_variables = _load_yaml(args.mix_variables)
        write_holder_panel_skeleton(
            contracts,
            mix_variables,
            contracts["panel"]["output_path"],
            quarters=args.quarter,
        )
        return 0
    if args.command == "write-panel":
        contracts = load_source_contracts(args.contracts)
        mix_variables = _load_yaml(args.mix_variables)
        write_holder_panel(
            contracts,
            mix_variables,
            contracts["panel"]["output_path"],
            project_root=args.project_root,
        )
        return 0
    if args.command == "write-exact-primary-panel":
        contracts = load_source_contracts(args.contracts)
        mix_variables = _load_yaml(args.mix_variables)
        write_exact_fu_primary_panel_outputs(
            contracts_config=contracts,
            mix_config=mix_variables,
            legacy_panel_path=args.legacy_panel,
            exact_sidecar_panel_path=args.exact_sidecar_panel,
            migration_gate_path=args.migration_gate,
            output_dir=args.output_dir,
        )
        return 0
    if args.command == "write-exact-primary-analysis":
        write_exact_primary_analysis_outputs(
            panel_path=args.panel,
            migration_gate_path=args.migration_gate,
            exact_sidecar_analysis_dir=args.exact_sidecar_analysis_dir,
            output_dir=args.output_dir,
        )
        return 0
    if args.command == "write-validation":
        import pandas as pd

        write_validation_outputs(pd.read_csv(args.panel), args.output_dir)
        return 0
    if args.command == "write-figures":
        import pandas as pd

        write_stacked_holder_absorption_figure(
            pd.read_csv(args.panel),
            args.output,
            start=args.start,
        )
        return 0
    if args.command == "write-design":
        import pandas as pd

        write_design_outputs(
            pd.read_csv(args.panel),
            validation_summary_path=args.validation_summary,
            exact_residual_overlap_path=args.exact_residual_overlap,
            output_dir=args.output_dir,
        )
        return 0
    if args.command == "write-estimates":
        write_estimate_outputs(
            matrices_path=args.matrices,
            stop_rule_status_path=args.stop_rule_status,
            pretrend_matrices_path=args.pretrend_matrices,
            output_dir=args.output_dir,
        )
        return 0
    if args.command == "write-estimate-figures":
        import pandas as pd

        write_lp_diagnostic_profile_figure(
            pd.read_csv(args.profile_plot_data),
            args.output,
        )
        return 0
    if args.command == "write-priors":
        import pandas as pd

        write_holder_absorption_priors(
            pd.read_csv(args.panel),
            load_sector_map(args.sector_map),
            args.output_dir,
        )
        return 0
    if args.command == "write-decision":
        write_project_decision_outputs(
            validation_summary_path=args.validation_summary,
            design_readiness_path=args.design_readiness,
            stop_rule_status_path=args.stop_rule_status,
            stability_summary_path=args.stability_summary,
            economics_review_path=args.economics_review,
            output_dir=args.output_dir,
        )
        return 0
    if args.command == "write-closeout":
        write_project_closeout_outputs(
            project_decision_path=args.project_decision,
            exact_primary_analysis_status_path=args.exact_primary_analysis_status,
            exact_primary_artifact_comparison_path=args.exact_primary_artifact_comparison,
            exact_primary_migration_comparison_path=args.exact_primary_migration_comparison,
            primary_migration_gate_path=args.primary_migration_gate,
            appendix_source_contract_path=args.appendix_source_contract,
            final_status_packet_path=args.final_status_packet,
            output_dir=args.output_dir,
        )
        return 0
    if args.command == "write-manifest":
        write_artifact_manifest_outputs(
            project_root=args.project_root,
            output_dir=args.output_dir,
        )
        return 0
    if args.command == "write-downstream-contracts":
        write_downstream_contract_bundle_outputs(
            prior_contract_path=args.prior_contract,
            appendix_source_contract_path=args.appendix_source_contract,
            final_status_packet_path=args.final_status_packet,
            bucket_definition_contract_path=args.bucket_definition_contract,
            bucket_migration_policy_path=args.bucket_migration_policy,
            primary_migration_gate_path=args.primary_migration_gate,
            default_source_decision_path=args.default_source_decision,
            output_dir=args.output_dir,
        )
        return 0
    if args.command == "write-z1-exact":
        write_z1_exact_outputs(
            config_path=args.config,
            holder_panel_path=args.holder_panel,
            output_dir=args.output_dir,
        )
        return 0
    if args.command == "write-z1-exact-analysis":
        write_z1_exact_analysis_outputs(
            exact_holder_panel_path=args.exact_holder_panel,
            legacy_design_path=args.legacy_design,
            legacy_summary_path=args.legacy_summary,
            legacy_stability_path=args.legacy_stability,
            exact_gate_path=args.exact_gate,
            output_dir=args.output_dir,
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must load as a mapping.")
    return loaded


if __name__ == "__main__":
    raise SystemExit(main())
