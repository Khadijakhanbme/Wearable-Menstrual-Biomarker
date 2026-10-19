"""
figures.py — Publication-quality figures for the Sehat-Cycle project.

Generates 3 figures:
  1. Phase-level bars: RHR and HRV (RMSSD) by menstrual phase
     — sets up the mechanism (Luteal rise in RHR, drop in HRV)
  2. Example ovulatory cycle: one participant's RHR coloured by phase
     + PdG bars below — illustrates the pattern in a single cycle
  3. Cycle-level scatter: RHR amplitude vs luteal PdG
     — THE key result (Spearman r = 0.275, p = 0.040, n = 56)

Figure 1 loads raw data directly.
Figures 2-3 read results/tables/cycle_amplitudes.csv (made by an earlier analysis run).
Figures are displayed, not saved.

Usage:
    python figures.py                        # uses DEFAULT_DATA_DIR
    python figures.py <path_to_mcphases>     # explicit path
"""

import argparse
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ── Constants (self-contained, no config.py needed) ─────────
PHASE_ORDER = ["Menstrual", "Follicular", "Fertility", "Luteal"]
PHASE_COLORS = {
    "Menstrual":  "#E84855",
    "Follicular": "#2E86AB",
    "Fertility":  "#F9A620",
    "Luteal":     "#44AF69",
}
MERGE_KEYS = ["id", "study_interval", "day_in_study"]
RHR_MIN = 40
RHR_MAX = 120
PDG_OVULATION_THRESHOLD = 5.0
C_OVULATORY   = "#2E86AB"
C_ANOVULATORY = "#E84855"
C_ACCENT      = "#F9A620"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TABLES_DIR  = os.path.join(RESULTS_DIR, "tables")

# Dataset folder (one level above src/)
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mcphases-a-dataset-of-physiological-hormonal-and-self-reported-"
    "events-and-symptoms-for-menstrual-health-tracking-with-wearables-1.0.0",
)


def apply_plot_style():
    """Set publication-quality matplotlib defaults."""
    plt.rcParams.update({
        "figure.dpi":        100,
        "font.family":       "sans-serif",
        "font.size":         10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _load_amplitude_csv():
    """Load the per-cycle amplitude results from analysis.py."""
    path = os.path.join(TABLES_DIR, "cycle_amplitudes.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run analysis.py first."
        )
    return pd.read_csv(path)


def load_rhr(data_dir):
    """Daily RHR from sleep_score.csv, in the same shape as resting_heart_rate.csv
    (id, study_interval, day_in_study, value). resting_heart_rate.csv is not used:
    its 2024 interval (the only one with PdG) mixes several participants'
    series under each id, ~5 rows per day."""
    sleep = pd.read_csv(os.path.join(data_dir, "sleep_score.csv"))
    return (
        sleep.groupby(MERGE_KEYS)["resting_heart_rate"].mean()
        .rename("value")
        .reset_index()
    )


def _check_phase_order(cycle_df):
    """Check if cycle phases follow physiological order (M → F → Fer → L).

    Returns True if the first appearance of each phase is monotonic.
    """
    phase_rank = {"Menstrual": 0, "Follicular": 1, "Fertility": 2, "Luteal": 3}
    sorted_df = cycle_df.sort_values("day_in_study")

    first_appearances = {}
    for _, row in sorted_df.iterrows():
        p = row["phase"]
        if p not in first_appearances and p in phase_rank:
            first_appearances[p] = row["day_in_study"]

    present = sorted(first_appearances.items(), key=lambda x: x[1])
    ranks = [phase_rank[p] for p, _ in present]
    return ranks == sorted(ranks)


# ═══════════════════════════════════════════════════════════════
# FIGURE 1 — Phase-level bars: RHR + HRV (two panels)
# ═══════════════════════════════════════════════════════════════

def fig1_phase_bars(data_dir):
    """Two-panel bar chart: RHR and RMSSD by menstrual phase.

    Sets up the mechanism: Luteal phase shows elevated RHR
    (sympathetic activation) and reduced HRV (vagal withdrawal),
    consistent with progesterone-mediated cardiovascular changes.
    """
    rhr = load_rhr(data_dir)
    hrv = pd.read_csv(os.path.join(data_dir, "heart_rate_variability_details.csv"))
    horm = pd.read_csv(os.path.join(data_dir, "hormones_and_selfreport.csv"))

    # Clean RHR
    rhr = rhr[(rhr["value"] >= RHR_MIN) & (rhr["value"] <= RHR_MAX)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Panel A: RHR by phase ──
    merged_rhr = rhr.merge(
        horm[MERGE_KEYS + ["phase"]], on=MERGE_KEYS, how="inner"
    )
    merged_rhr = merged_rhr.dropna(subset=["phase", "value"])
    rhr_stats = (
        merged_rhr.groupby("phase")["value"]
        .agg(["mean", "sem"])
        .reindex(PHASE_ORDER)
    )

    colors = [PHASE_COLORS[p] for p in PHASE_ORDER]
    ax1.bar(PHASE_ORDER, rhr_stats["mean"],
            yerr=rhr_stats["sem"] * 1.96,
            color=colors, edgecolor="white", capsize=4, alpha=0.85)
    ax1.set_ylabel("Resting Heart Rate (bpm)")
    ax1.set_title("RHR by Menstrual Phase", fontweight="bold")
    # Zoom in to show differences clearly
    y_min = rhr_stats["mean"].min() - 1.5
    y_max = rhr_stats["mean"].max() + 1.5
    ax1.set_ylim(y_min, y_max)

    # ── Panel B: RMSSD by phase ──
    daily_hrv = hrv.drop_duplicates().groupby(MERGE_KEYS)["rmssd"].mean().reset_index()
    merged_hrv = daily_hrv.merge(
        horm[MERGE_KEYS + ["phase"]], on=MERGE_KEYS, how="inner"
    )
    merged_hrv = merged_hrv.dropna(subset=["phase", "rmssd"])
    hrv_stats = (
        merged_hrv.groupby("phase")["rmssd"]
        .agg(["mean", "sem"])
        .reindex(PHASE_ORDER)
    )

    ax2.bar(PHASE_ORDER, hrv_stats["mean"],
            yerr=hrv_stats["sem"] * 1.96,
            color=colors, edgecolor="white", capsize=4, alpha=0.85)
    ax2.set_ylabel("RMSSD (ms)")
    ax2.set_title("HRV by Menstrual Phase", fontweight="bold")
    y_min2 = hrv_stats["mean"].min() - 3
    y_max2 = hrv_stats["mean"].max() + 3
    ax2.set_ylim(y_min2, y_max2)

    fig.suptitle(
        "Cardiovascular Markers Across Menstrual Phases (n = 42)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))


# ═══════════════════════════════════════════════════════════════
# FIGURE 2 — Example ovulatory cycle (RHR + PdG, two panels)
# ═══════════════════════════════════════════════════════════════

def fig2_example_cycle(data_dir):
    """Two-panel figure showing one exemplary ovulatory cycle.

    Top: daily RHR coloured by menstrual phase, with amplitude arrow.
    Bottom: daily PdG bars with the ovulation threshold line.

    Participant selection criteria (from Improved_figures.py):
      - ovulatory cycle (PdG >= 7 in second half)
      - good RHR coverage (>= 15 days)
      - reasonable length (20-40 days)
      - phases follow physiological order (M → F → Fer → L)
      - PdG peak falls in Luteal phase
      - at least 3 phases present, preference for all 4
    """
    rhr = load_rhr(data_dir)
    horm = pd.read_csv(os.path.join(data_dir, "hormones_and_selfreport.csv"))
    rhr = rhr[(rhr["value"] >= RHR_MIN) & (rhr["value"] <= RHR_MAX)]
    rhr = rhr.rename(columns={"value": "rhr_mean"})

    amp = _load_amplitude_csv()

    # ── Smart participant selection ──
    # Start from ovulatory cycles with good coverage
    candidates = amp[
        (amp["ovulatory"] == True)
        & (amp["n_rhr"] >= 15)
        & (amp["n_days"] >= 20)
        & (amp["n_days"] <= 40)
        & (amp["pdg_second_half_max"] >= 7)
    ]

    if len(candidates) == 0:
        print("  Skipping fig2: no ideal cycle found")
        return

    # Score each candidate on additional quality criteria
    best_score = -1
    best_info = None

    for _, row in candidates.iterrows():
        pid = row["id"]
        interval = row["study_interval"]
        cycle_id = row["cycle_id"]

        # Re-segment to get this cycle's hormone data
        horm_sorted = horm[
            (horm["id"] == pid) & (horm["study_interval"] == interval)
        ].sort_values("day_in_study").copy()

        cycle_num = 0
        prev = None
        labels = []
        for phase in horm_sorted["phase"]:
            ps = str(phase).lower() if pd.notna(phase) else ""
            if "menstr" in ps and (prev is None or "menstr" not in prev):
                cycle_num += 1
            labels.append(f"P{pid}_{interval}_C{max(cycle_num, 1)}")
            prev = ps
        horm_sorted["cycle_id"] = labels

        cycle_horm = horm_sorted[horm_sorted["cycle_id"] == cycle_id]
        if len(cycle_horm) == 0:
            continue

        # Check phase order is physiological
        if not _check_phase_order(cycle_horm):
            continue

        # PdG peak should be in Luteal phase
        pdg_valid = cycle_horm.dropna(subset=["pdg"])
        if len(pdg_valid) < 5:
            continue
        pdg_max_phase = pdg_valid.loc[pdg_valid["pdg"].idxmax(), "phase"]
        if pdg_max_phase != "Luteal":
            continue

        n_phases = cycle_horm["phase"].dropna().nunique()
        phase_bonus = 1.5 if n_phases == 4 else 1.0

        # Score: RHR amplitude × PdG clarity × phase coverage
        score = row["rhr_amplitude"] * row["pdg_second_half_max"] * phase_bonus

        if score > best_score:
            best_score = score
            best_info = (pid, interval, cycle_id, cycle_horm)

    if best_info is None:
        print("  Skipping fig2: no cycle passed quality checks")
        return

    pid, interval, cycle_id, cycle_horm = best_info
    days = cycle_horm["day_in_study"].values

    # Get RHR for these days
    cycle_rhr = rhr[
        (rhr["id"] == pid)
        & (rhr["study_interval"] == interval)
        & (rhr["day_in_study"].isin(days))
    ].sort_values("day_in_study")

    cycle_rhr = cycle_rhr.merge(
        cycle_horm[["day_in_study", "phase", "pdg"]],
        on="day_in_study", how="left",
    )

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # Normalise days to start at 1
    day_offset = cycle_rhr["day_in_study"].min()
    cycle_days = cycle_rhr["day_in_study"] - day_offset + 1

    # Top panel: RHR coloured by phase
    for phase, color in PHASE_COLORS.items():
        mask = cycle_rhr["phase"] == phase
        ax1.scatter(cycle_days[mask], cycle_rhr.loc[mask, "rhr_mean"],
                    color=color, s=40, alpha=0.8, label=phase, zorder=3)
    ax1.plot(cycle_days, cycle_rhr["rhr_mean"],
             color="gray", alpha=0.3, lw=1, zorder=2)

    # Amplitude arrow
    rhr_min = cycle_rhr["rhr_mean"].min()
    rhr_max = cycle_rhr["rhr_mean"].max()
    amp_val = rhr_max - rhr_min
    ax1.annotate(
        "", xy=(cycle_days.iloc[-1] + 1, rhr_max),
        xytext=(cycle_days.iloc[-1] + 1, rhr_min),
        arrowprops=dict(arrowstyle="<->", color=C_ACCENT, lw=2),
    )
    ax1.text(cycle_days.iloc[-1] + 2, (rhr_max + rhr_min) / 2,
             f"Amplitude\n{amp_val:.1f} bpm",
             fontsize=10, color=C_ACCENT, va="center", fontweight="bold")

    ax1.set_ylabel("RHR (bpm)", fontsize=12)
    ax1.set_title(
        f"Example Ovulatory Cycle (Participant {int(pid)})",
        fontsize=13, fontweight="bold",
    )
    ax1.legend(loc="upper left", fontsize=9, ncol=4)

    # Bottom panel: PdG bars
    pdg_data = cycle_horm.dropna(subset=["pdg"])
    pdg_days = pdg_data["day_in_study"] - day_offset + 1
    ax2.bar(pdg_days, pdg_data["pdg"],
            color=PHASE_COLORS["Luteal"], alpha=0.7, width=0.8)
    ax2.axhline(y=PDG_OVULATION_THRESHOLD,
                color=C_ANOVULATORY, ls="--", alpha=0.5,
                label="Ovulation threshold")
    ax2.set_ylabel("PdG", fontsize=12)
    ax2.set_xlabel("Cycle Day", fontsize=12)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    print(f"    example cycle: "
          f"(P{int(pid)}, amplitude={amp_val:.1f} bpm)")


# ═══════════════════════════════════════════════════════════════
# FIGURE 3 — Cycle-level scatter (THE key result)
# ═══════════════════════════════════════════════════════════════

def fig3_cycle_scatter():
    """Scatter plot of RHR amplitude vs luteal PdG, one dot per cycle.

    Coloured by ovulatory/anovulatory status. This is the figure
    backing the abstract's Spearman r = 0.275, p = 0.040, n = 56.
    """
    amp = _load_amplitude_csv()
    valid = amp.dropna(subset=["rhr_amplitude", "pdg_second_half_max"])

    if len(valid) < 5:
        print("  Skipping fig3: not enough data")
        return

    r, p = spearmanr(valid["pdg_second_half_max"], valid["rhr_amplitude"])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for ov, color, label in [
        (True,  C_OVULATORY,   "Ovulatory"),
        (False, C_ANOVULATORY, "Anovulatory"),
    ]:
        mask = valid["ovulatory"] == ov
        ax.scatter(
            valid.loc[mask, "pdg_second_half_max"],
            valid.loc[mask, "rhr_amplitude"],
            c=color, s=60, alpha=0.7, edgecolors="white",
            linewidth=0.5, label=label, zorder=3,
        )

    # Trend line
    z = np.polyfit(valid["pdg_second_half_max"],
                   valid["rhr_amplitude"], 1)
    x_line = np.linspace(valid["pdg_second_half_max"].min(),
                         valid["pdg_second_half_max"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line),
            color="gray", ls="--", alpha=0.5, zorder=2)

    # PdG threshold
    ax.axvline(x=PDG_OVULATION_THRESHOLD,
               color=C_ANOVULATORY, ls=":", alpha=0.4,
               label=f"Ovulation threshold (PdG = {PDG_OVULATION_THRESHOLD})")

    # Stats annotation
    sig = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.05, 0.95,
            f"Spearman r = {r:.3f}\n{sig}\nn = {len(valid)} cycles",
            transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="gray", alpha=0.8))

    ax.set_xlabel("Luteal PdG (progesterone metabolite)", fontsize=12)
    ax.set_ylabel("RHR Amplitude (bpm)", fontsize=12)
    ax.set_title("Progesterone Level Predicts Heart Rate Swing",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    print(f"    r={r:.3f}, p={p:.4f}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main(data_dir):
    apply_plot_style()

    print("=" * 60)
    print("  GENERATING PUBLICATION FIGURES")
    print("=" * 60)

    print("\n  Figure 1: Phase-level bars (RHR + HRV)")
    fig1_phase_bars(data_dir)

    print("  Figure 2: Example ovulatory cycle")
    fig2_example_cycle(data_dir)

    print("  Figure 3: Cycle-level scatter (RHR amplitude vs PdG)")
    fig3_cycle_scatter()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate publication figures for Sehat-Cycle"
    )
    parser.add_argument("data_dir", nargs="?", default=DEFAULT_DATA_DIR,
                        help="Path to mcPHASES data folder (default: auto-detect)")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"  ERROR: Data directory not found: {args.data_dir}")
        sys.exit(1)

    main(args.data_dir)