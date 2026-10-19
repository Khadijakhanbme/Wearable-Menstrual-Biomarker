"""
explore_data.py — Exploratory overview of the mcPHASES dataset.

Prints a dataset summary and shows 3 figures that tell the core story
to a new reader, from one person to the whole group:
  1. One participant: progesterone, resting heart rate and HRV day by day,
     with menstrual phases shaded
  2. The average cycle: the same three signals by day of cycle, each
     person compared with their own average
  3. Phase summary: how much RHR and HRV change in each phase,
     relative to each person's own average

Run this FIRST to get a feel for the data before analysis.

Usage:
    python explore_data.py
"""

import argparse
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Constants ─────────────────────────────────────────────────
PHASE_COLORS = {
    "Menstrual":  "#E84855",   # red
    "Follicular": "#2E86AB",   # blue
    "Fertility":  "#F9A620",   # gold
    "Luteal":     "#44AF69",   # green
}
PHASE_ORDER = ["Menstrual", "Follicular", "Fertility", "Luteal"]
MERGE_KEYS = ["id", "study_interval", "day_in_study"]
RHR_MIN, RHR_MAX = 40, 120   # bpm; outside this = sensor error
PLOT2_PARTICIPANT = 10       # participant for figure 1; None = most PdG data
OVULATION_PDG = 5.0          # PdG above this = ovulation confirmed
CYCLE_MIN_DAYS, CYCLE_MAX_DAYS = 20, 45
MAX_CYCLE_DAY_SHOWN = 35     # x-axis limit for the average-cycle figure
MIN_CYCLES_PER_DAY = 20      # hide cycle days with fewer cycles than this

INK = "#333333"              # single-series lines (identity is the panel title)
MUTED = "#888888"

# Dataset folder (one level above src/)
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mcphases-a-dataset-of-physiological-hormonal-and-self-reported-"
    "events-and-symptoms-for-menstrual-health-tracking-with-wearables-1.0.0",
)


def apply_plot_style():
    plt.rcParams.update({
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    })


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_daily(data_dir):
    """Load the key CSVs into one table: one row per participant-day."""
    # RHR from sleep_score.csv, not resting_heart_rate.csv: in the latter the
    # 2024 interval mixes several participants' series under each id
    # (see cycle_level_analysis.py). One value per participant-day.
    sleep = pd.read_csv(os.path.join(data_dir, "sleep_score.csv"))
    rhr = sleep.groupby(MERGE_KEYS)["resting_heart_rate"].mean().rename("rhr")
    rhr = rhr[(rhr >= RHR_MIN) & (rhr <= RHR_MAX)]

    hrv = pd.read_csv(os.path.join(data_dir, "heart_rate_variability_details.csv"))
    rmssd = hrv.drop_duplicates().groupby(MERGE_KEYS)["rmssd"].mean()

    horm = pd.read_csv(os.path.join(data_dir, "hormones_and_selfreport.csv"))
    daily = (
        horm[MERGE_KEYS + ["phase", "pdg"]]
        .dropna(subset=["phase"])
        .merge(rhr.reset_index(), on=MERGE_KEYS, how="left")
        .merge(rmssd.reset_index(), on=MERGE_KEYS, how="left")
        .sort_values(MERGE_KEYS)
        .reset_index(drop=True)
    )

    # Each person compared with their own average (per study interval),
    # so differences between people don't hide the cycle effect.
    grp = daily.groupby(["id", "study_interval"])
    daily["rhr_change"] = daily["rhr"] - grp["rhr"].transform("mean")
    daily["rmssd_change"] = daily["rmssd"] - grp["rmssd"].transform("mean")

    daily["cycle_day"] = add_cycle_day(daily)
    return daily


def add_cycle_day(daily):
    """Day of cycle (1 = first day of a period) for every day that lies in a
    complete cycle of plausible length; NaN otherwise."""
    cycle_day = pd.Series(np.nan, index=daily.index)
    for _, chunk in daily.groupby(["id", "study_interval"]):
        days = chunk["day_in_study"].to_numpy()
        menstrual = (chunk["phase"] == "Menstrual").to_numpy()
        onsets = [days[i] for i in range(1, len(days))
                  if menstrual[i] and not menstrual[i - 1]]
        for start, end in zip(onsets[:-1], onsets[1:]):
            if not CYCLE_MIN_DAYS <= end - start <= CYCLE_MAX_DAYS:
                continue
            in_cycle = (days >= start) & (days < end)
            cycle_day[chunk.index[in_cycle]] = days[in_cycle] - start + 1
    return cycle_day


def print_summary(daily):
    print(f"\n  Dataset summary")
    print(f"  {'─' * 50}")
    print(f"  Participant-days: {len(daily):>7,}   ({daily['id'].nunique()} participants)")
    for col, name in [("rhr", "Resting HR"), ("rmssd", "HRV (RMSSD)"), ("pdg", "PdG")]:
        n = daily[col].notna().sum()
        print(f"  {name:<16}{n:>7,} days  ({100 * n / len(daily):.0f}%)")
    pdg_years = sorted(daily.loc[daily["pdg"].notna(), "study_interval"].unique())
    print(f"  PdG measured in: {', '.join(map(str, pdg_years))} only")
    n_cycles = daily.dropna(subset=["cycle_day"]).query("cycle_day == 1").shape[0]
    print(f"  Complete cycles:  {n_cycles:>7,}")

    print(f"\n  Phase distribution:")
    for phase, count in daily["phase"].value_counts().reindex(PHASE_ORDER).items():
        print(f"    {phase:<12} {count:>6,} days")


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def shade_phases(ax, x, phases):
    """Shade the background of ax by phase, one band per run of equal phase."""
    x = np.asarray(x)
    phases = np.asarray(phases)
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or phases[i] != phases[start] or x[i] - x[i - 1] > 1:
            if phases[start] in PHASE_COLORS:
                ax.axvspan(x[start] - 0.5, x[i - 1] + 0.5,
                           color=PHASE_COLORS[phases[start]], alpha=0.15, lw=0)
            start = i


def phase_legend(fig):
    """Phase key on its own line, just under the figure title."""
    handles = [Patch(color=PHASE_COLORS[p], alpha=0.35, label=p) for p in PHASE_ORDER]
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.005, 0.955))


# ═══════════════════════════════════════════════════════════════
# FIGURE 1 — One participant, day by day
# ═══════════════════════════════════════════════════════════════

def fig1_one_participant(daily):
    """PdG, RHR and RMSSD for one participant, with phases shaded.
    Shows the raw pattern: when progesterone rises, RHR rises and HRV falls."""
    pdg_counts = daily.dropna(subset=["pdg"]).groupby("id").size()
    if len(pdg_counts) == 0:
        print("  Skipping figure 1: no PdG data")
        return
    pid = PLOT2_PARTICIPANT if PLOT2_PARTICIPANT is not None else pdg_counts.idxmax()

    # Use the study interval with PdG data for this person
    person = daily[(daily["id"] == pid) & daily["pdg"].notna()]
    interval = person["study_interval"].mode()[0]
    person = daily[(daily["id"] == pid) & (daily["study_interval"] == interval)]

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    panels = [
        ("pdg",   "PdG (ng/mL)",        "Progesterone (PdG)"),
        ("rhr",   "Resting HR (bpm)",   "Resting heart rate"),
        ("rmssd", "RMSSD (ms)",         "Heart rate variability (RMSSD)"),
    ]
    for ax, (col, ylabel, title) in zip(axes, panels):
        shade_phases(ax, person["day_in_study"], person["phase"])
        d = person.dropna(subset=[col])
        # Break the line across gaps of more than a few days with no data
        segment = (d["day_in_study"].diff() > 3).cumsum()
        for _, seg in d.groupby(segment):
            ax.plot(seg["day_in_study"], seg[col], color=INK, lw=1.5,
                    marker="o", ms=3)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    axes[0].axhline(OVULATION_PDG, color=MUTED, ls="--", lw=1)
    axes[0].text(person["day_in_study"].min(), OVULATION_PDG, " ovulation threshold",
                 va="bottom", fontsize=8, color=MUTED)
    axes[-1].set_xlabel("Day in study")

    phase_legend(fig)
    fig.suptitle(f"Participant {pid} ({interval}): heart rate rises and HRV "
                 f"falls when progesterone is high", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))


# ═══════════════════════════════════════════════════════════════
# FIGURE 2 — The average cycle
# ═══════════════════════════════════════════════════════════════

def fig2_average_cycle(daily):
    """PdG, RHR change and RMSSD change by day of cycle, averaged over all
    complete cycles. RHR/RMSSD are relative to each person's own average."""
    cyc = daily.dropna(subset=["cycle_day"])
    cyc = cyc[cyc["cycle_day"] <= MAX_CYCLE_DAY_SHOWN]

    # Most common phase on each cycle day → background shading
    typical_phase = cyc.groupby("cycle_day")["phase"].agg(lambda p: p.mode()[0])

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    panels = [
        ("pdg",          "PdG (ng/mL)",       "Progesterone (PdG), median and middle 50%", "median"),
        ("rhr_change",   "Change (bpm)",      "Resting heart rate vs own average, mean ± SE", "mean"),
        ("rmssd_change", "Change (ms)",       "HRV (RMSSD) vs own average, mean ± SE", "mean"),
    ]
    for ax, (col, ylabel, title, how) in zip(axes, panels):
        shade_phases(ax, typical_phase.index, typical_phase.values)
        g = cyc.dropna(subset=[col]).groupby("cycle_day")[col]
        n = g.size()
        keep = n[n >= MIN_CYCLES_PER_DAY].index
        if how == "median":
            mid = g.median()[keep]
            lo, hi = g.quantile(0.25)[keep], g.quantile(0.75)[keep]
        else:
            mid = g.mean()[keep]
            se = g.sem()[keep]
            lo, hi = mid - se, mid + se
            ax.axhline(0, color=MUTED, lw=1)
        ax.fill_between(keep, lo, hi, color=INK, alpha=0.15, lw=0)
        ax.plot(keep, mid, color=INK, lw=2)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    axes[-1].set_xlabel("Day of cycle (day 1 = first day of period)")
    axes[-1].set_xlim(0.5, MAX_CYCLE_DAY_SHOWN + 0.5)

    n_cycles = (cyc["cycle_day"] == 1).sum()
    n_people = cyc["id"].nunique()
    phase_legend(fig)
    fig.suptitle(f"The average cycle ({n_cycles} cycles, {n_people} participants): "
                 f"heart rate peaks and HRV dips in the luteal phase",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))


# ═══════════════════════════════════════════════════════════════
# FIGURE 3 — Phase summary
# ═══════════════════════════════════════════════════════════════

def fig3_phase_summary(daily):
    """Change in RHR and RMSSD per phase vs each person's own average.
    Averaged per person first, so every participant counts once."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    panels = [
        ("rhr_change",   "Resting heart rate", "bpm"),
        ("rmssd_change", "HRV (RMSSD)",        "ms"),
    ]
    for ax, (col, title, unit) in zip(axes, panels):
        per_person = daily.groupby(["id", "phase"])[col].mean().unstack()
        per_person = per_person.reindex(columns=PHASE_ORDER)
        mean, se = per_person.mean(), per_person.sem()
        n = per_person.notna().sum().min()

        ax.axhline(0, color=MUTED, lw=1)
        for i, phase in enumerate(PHASE_ORDER):
            ax.errorbar(i, mean[phase], yerr=se[phase], fmt="o", ms=9,
                        color=PHASE_COLORS[phase], mec="white", mew=2,
                        ecolor=PHASE_COLORS[phase], elinewidth=2, capsize=0)
            ax.annotate(f"{mean[phase]:+.1f} {unit}", (i, mean[phase]),
                        xytext=(12, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.set_xticks(range(len(PHASE_ORDER)), PHASE_ORDER)
        ax.set_xlim(-0.5, len(PHASE_ORDER) - 0.2)
        ax.set_ylabel(f"Change vs own average ({unit})")
        ax.set_title(f"{title} (n = {n} participants, mean ± SE)",
                     loc="left", fontsize=10, fontweight="bold")
        ax.grid(axis="x", visible=False)

    fig.suptitle("By phase: heart rate is highest and HRV lowest in the "
                 "luteal phase",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main(data_dir):
    apply_plot_style()

    print("=" * 60)
    print("  EXPLORATORY DATA OVERVIEW")
    print("=" * 60)

    daily = load_daily(data_dir)
    print_summary(daily)

    print("\n  Generating figures (close all windows to finish)...")
    fig1_one_participant(daily)
    fig2_average_cycle(daily)
    fig3_phase_summary(daily)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explore the mcPHASES dataset with summary stats and plots"
    )
    parser.add_argument("data_dir", nargs="?", default=DEFAULT_DATA_DIR,
                        help="Path to mcPHASES data folder")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"  ERROR: Data directory not found: {args.data_dir}")
        sys.exit(1)

    main(args.data_dir)
