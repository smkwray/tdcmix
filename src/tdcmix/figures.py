from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd


HEADLINE_SHARE_COLUMNS = {
    "Fed": "pos_abs_share_fed",
    "Banks": "pos_abs_share_banks",
    "ROW": "pos_abs_share_row",
    "MMF": "pos_abs_share_mmf",
    "Domestic nonbank": "pos_abs_share_domestic_nonbank",
    "Dealers": "pos_abs_share_dealer_bridge",
    "Other financial": "pos_abs_share_other_financial",
    "Unmapped": "pos_abs_share_unmapped",
}


def write_stacked_holder_absorption_figure(
    panel: pd.DataFrame,
    output_path: str | Path = "output/holder_absorption_shares.png",
    *,
    start: str = "1990-01-01",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plot = panel.copy()
    plot["quarter_date"] = pd.to_datetime(plot["quarter"], errors="coerce", format="mixed")
    plot = plot.loc[plot["quarter_date"] >= pd.Timestamp(start)].sort_values("quarter_date")
    share_columns = [column for column in HEADLINE_SHARE_COLUMNS.values() if column in plot.columns]
    plot = plot.dropna(subset=share_columns, how="all")

    fig, ax = plt.subplots(figsize=(12, 6))
    if plot.empty:
        ax.text(0.5, 0.5, "No holder-share rows available", ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = [label for label, column in HEADLINE_SHARE_COLUMNS.items() if column in plot.columns]
        values = [plot[column].fillna(0).to_numpy() for column in share_columns]
        ax.stackplot(plot["quarter_date"], values, labels=labels, linewidth=0)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Share of positive Treasury absorption")
        ax.set_xlabel("")
        ax.set_title("Holder Absorption Mix")
        ax.legend(loc="upper left", ncols=2, frameon=False)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_lp_diagnostic_profile_figure(
    profile_plot_data: pd.DataFrame,
    output_path: str | Path = "output/lp_diagnostic_profiles.png",
    *,
    max_panels: int = 6,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = profile_plot_data.loc[profile_plot_data["status"].eq("ok")].copy()
    data["abs_t"] = pd.to_numeric(data["t_stat"], errors="coerce").abs()
    keys = (
        data.groupby(["outcome", "contrast"], as_index=False)["abs_t"]
        .max()
        .sort_values("abs_t", ascending=False)
        .head(max_panels)
    )

    if keys.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No diagnostic profile data available", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    n = len(keys)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.8 * nrows), squeeze=False)
    sample_styles = {
        "baseline": {"marker": "o", "linewidth": 1.8},
        "exclude_major_disruption": {"marker": "s", "linewidth": 1.4},
        "with_regime_flags": {"marker": "^", "linewidth": 1.4},
    }
    for ax, key in zip(axes.ravel(), keys.to_dict("records"), strict=False):
        sub = data.loc[
            data["outcome"].eq(key["outcome"]) & data["contrast"].eq(key["contrast"])
        ].copy()
        for sample, style in sample_styles.items():
            line = sub.loc[sub["sample"].eq(sample)].sort_values("horizon")
            if line.empty:
                continue
            ax.plot(
                line["horizon"],
                line["beta"],
                label=sample,
                **style,
            )
            ax.fill_between(
                line["horizon"],
                line["lower_approx_95"],
                line["upper_approx_95"],
                alpha=0.12,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{key['outcome']} / {key['contrast']}", fontsize=10)
        ax.set_xlabel("horizon")
        ax.set_ylabel("diagnostic beta")
        ax.grid(axis="y", alpha=0.25)
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncols=3, frameon=False)
    fig.suptitle("Diagnostic LP Profiles", y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
