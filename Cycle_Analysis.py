"""
analysis.py — Core analysis pipeline for Sehat-Cycle.

Loads the mcPHASES dataset, segments individual menstrual cycles,
computes per-cycle cardiovascular amplitude and hormonal metrics,
runs statistical tests, and exports the results table.

Key result (from the Think Beyond Pink abstract):
    "RHR amplitude correlated with luteal progesterone:
     Spearman r = 0.275, p = 0.040, n = 56 cycles"

Pipeline:
    1. Load RHR, HRV, and hormone CSVs
    2. Segment menstrual cycles using phase transitions → Menstrual
    3. Compute per-cycle:
         - rhr_amplitude   = max(RHR) − min(RHR)
         - rmssd_amplitude = max(RMSSD) − min(RMSSD)
         - rhr_half_diff   = mean(2nd half RHR) − mean(1st half RHR)
         - pdg_second_half_max = max PdG in the second half of the cycle
    4. Spearman correlations + Mann-Whitney group comparison
    5. Export → results/tables/cycle_amplitudes.csv

Usage:
    python analysis.py <path_to_mcphases_data_folder>
"""

import argparse
import os
import sys

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

# ── Constants (self-contained, no config.py needed) ─────────
MERGE_KEYS = ["id", "study_interval", "day_in_study"]
RHR_MIN = 40            # bpm — below this is sensor noise
RHR_MAX = 120           # bpm — above this is sensor noise
MIN_CYCLE_DAYS = 10     # shortest fragment we trust for amplitude
PDG_OVULATION_THRESHOLD = 5.0   # PdG >= 5 in 2nd half → ovulatory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TABLES_DIR  = os.path.join(RESULTS_DIR, "tables")

# Dataset folder (one level above src/)
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mcphases-a-dataset-of-physiological-hormonal-and-self-reported-"
    "events-and-symptoms-for-menstrual-health-tracking-with-wearables-1.0.0",
)


# ═══════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_data(data_dir):
    """Load the three key tables and apply basic filters."""
    horm = pd.read_csv(os.path.join(data_dir, "hormones_and_selfreport.csv"))
    rhr  = pd.read_csv(os.path.join(data_dir, "resting_heart_rate.csv"))
    hrv  = pd.read_csv(os.path.join(data_dir, "heart_rate_variability_details.csv"))

    # Clean RHR: remove outliers and rename for clarity
    rhr = rhr.rename(columns={"value": "rhr_mean"})
    before = len(rhr)
    rhr = rhr[(rhr["rhr_mean"] >= RHR_MIN) & (rhr["rhr_mean"] <= RHR_MAX)]
    print(f"  RHR: {before:,} → {len(rhr):,} rows "
          f"(removed {before - len(rhr):,} outliers)")

    # Aggregate intraday HRV to one RMSSD value per participant-day
    daily_hrv = (
        hrv.groupby(MERGE_KEYS)
        .agg(rmssd_mean=("rmssd", "mean"))
        .reset_index()
    )
    print(f"  HRV: {len(hrv):,} raw → {len(daily_hrv):,} daily rows")
    print(f"  Hormones: {len(horm):,} rows, "
          f"{horm['id'].nunique()} participants")

    return horm, rhr, daily_hrv


# ═══════════════════════════════════════════════════════════════
# 2. CYCLE SEGMENTATION
# ═══════════════════════════════════════════════════════════════

def segment_cycles(horm):
    """Segment the hormone table into individual menstrual cycles.

    A new cycle starts wherever the phase transitions INTO 'Menstrual'
    from a non-Menstrual phase. Each row gets a unique cycle_id.
    """
    horm = horm.copy()
    horm.sort_values(MERGE_KEYS, inplace=True)

    cycle_ids = []
    for (_pid, _interval), group in horm.groupby(["id", "study_interval"]):
        cycle_num = 0
        prev_phase = None
        for phase in group["phase"]:
            phase_str = str(phase).lower() if pd.notna(phase) else ""
            is_menstrual = "menstr" in phase_str
            was_menstrual = prev_phase is not None and "menstr" in prev_phase
            if is_menstrual and not was_menstrual:
                cycle_num += 1
            cycle_ids.append(
                f"P{_pid}_{_interval}_C{max(cycle_num, 1)}"
            )
            prev_phase = phase_str

    horm["cycle_id"] = cycle_ids
    n_cycles = horm["cycle_id"].nunique()
    print(f"  Segmented {n_cycles} cycles from hormone data")
    return horm


# ═══════════════════════════════════════════════════════════════
# 3. PER-CYCLE AMPLITUDE COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_amplitudes(horm, rhr, daily_hrv):
    """For each cycle, pull matching RHR/HRV data and compute
    per-cycle amplitude metrics.

    Returns a DataFrame with one row per cycle.
    """
    records = []

    for cycle_id, cycle in horm.groupby("cycle_id"):
        n_days = len(cycle)
        if n_days < MIN_CYCLE_DAYS:
            continue

        pid = cycle["id"].iloc[0]
        interval = cycle["study_interval"].iloc[0]
        days = cycle["day_in_study"].values

        # ── Match RHR data to this cycle's days ──
        rhr_cycle = rhr[
            (rhr["id"] == pid)
            & (rhr["study_interval"] == interval)
            & (rhr["day_in_study"].isin(days))
        ].sort_values("day_in_study")

        # ── Match HRV data to this cycle's days ──
        hrv_cycle = daily_hrv[
            (daily_hrv["id"] == pid)
            & (daily_hrv["study_interval"] == interval)
            & (daily_hrv["day_in_study"].isin(days))
        ].sort_values("day_in_study")

        # Split into first / second halves
        mid = n_days // 2
        first_half_days = set(days[:mid])
        second_half_days = set(days[mid:])

        record = {
            "cycle_id": cycle_id,
            "id": pid,
            "study_interval": interval,
            "n_days": n_days,
            "n_rhr": len(rhr_cycle),
            "n_hrv": len(hrv_cycle),
        }

        # ── RHR amplitude (max − min within the whole cycle) ──
        if len(rhr_cycle) >= 5:
            vals = rhr_cycle["rhr_mean"]
            record["rhr_amplitude"] = vals.max() - vals.min()
            record["rhr_mean_overall"] = vals.mean()

            # Half-difference (2nd half mean − 1st half mean)
            rhr_1st = rhr_cycle[
                rhr_cycle["day_in_study"].isin(first_half_days)
            ]["rhr_mean"]
            rhr_2nd = rhr_cycle[
                rhr_cycle["day_in_study"].isin(second_half_days)
            ]["rhr_mean"]
            if len(rhr_1st) >= 3 and len(rhr_2nd) >= 3:
                record["rhr_half_diff"] = rhr_2nd.mean() - rhr_1st.mean()

        # ── RMSSD amplitude ──
        if len(hrv_cycle) >= 5:
            vals = hrv_cycle["rmssd_mean"]
            record["rmssd_amplitude"] = vals.max() - vals.min()
            record["rmssd_mean_overall"] = vals.mean()

            rmssd_1st = hrv_cycle[
                hrv_cycle["day_in_study"].isin(first_half_days)
            ]["rmssd_mean"]
            rmssd_2nd = hrv_cycle[
                hrv_cycle["day_in_study"].isin(second_half_days)
            ]["rmssd_mean"]
            if len(rmssd_1st) >= 3 and len(rmssd_2nd) >= 3:
                record["rmssd_half_diff"] = (
                    rmssd_1st.mean() - rmssd_2nd.mean()
                )

        # ── PdG (progesterone) from hormone table ──
        cycle_sorted = cycle.sort_values("day_in_study")
        pdg_all = cycle_sorted["pdg"].dropna()
        pdg_2nd = cycle_sorted.iloc[mid:]["pdg"].dropna()

        if len(pdg_all) > 0:
            record["pdg_max"] = pdg_all.max()
        if len(pdg_2nd) > 0:
            record["pdg_second_half_max"] = pdg_2nd.max()
            record["pdg_second_half_mean"] = pdg_2nd.mean()
            record["ovulatory"] = (
                record["pdg_second_half_max"] >= PDG_OVULATION_THRESHOLD
            )
        else:
            record["ovulatory"] = np.nan

        records.append(record)

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════
# 4. STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════

def run_statistics(amp):
    """Run Spearman correlations and Mann-Whitney group comparisons."""
    print("\n" + "=" * 60)
    print("  STATISTICAL RESULTS")
    print("=" * 60)

    results = {}

    # ── A. Spearman correlations: amplitude vs PdG ──
    print("\n  A. SPEARMAN CORRELATIONS (amplitude vs luteal PdG)")
    metrics = [
        "rhr_amplitude", "rmssd_amplitude",
        "rhr_half_diff", "rmssd_half_diff",
    ]
    for metric in metrics:
        if metric not in amp.columns:
            continue
        valid = amp.dropna(subset=[metric, "pdg_second_half_max"])
        if len(valid) < 5:
            print(f"\n     {metric}: too few cycles ({len(valid)})")
            continue
        r, p = spearmanr(valid[metric], valid["pdg_second_half_max"])
        sig = (
            "***" if p < 0.001 else
            "**"  if p < 0.01  else
            "*"   if p < 0.05  else "ns"
        )
        results[metric] = {"r": r, "p": p, "n": len(valid)}
        print(f"\n     {metric}:")
        print(f"       Spearman r = {r:.3f}, p = {p:.4f} {sig}  "
              f"(n = {len(valid)})")

    # ── B. Mann-Whitney: ovulatory vs anovulatory ──
    labeled = amp.dropna(subset=["ovulatory"])
    n_ov   = (labeled["ovulatory"] == True).sum()
    n_anov = (labeled["ovulatory"] == False).sum()

    print(f"\n  B. GROUP COMPARISON "
          f"({n_ov} ovulatory, {n_anov} anovulatory)")

    if n_anov >= 3:
        for metric in metrics:
            if metric not in labeled.columns:
                continue
            ov = labeled[labeled["ovulatory"] == True][metric].dropna()
            an = labeled[labeled["ovulatory"] == False][metric].dropna()
            if len(ov) < 3 or len(an) < 3:
                continue
            stat, p = mannwhitneyu(ov, an, alternative="two-sided")
            sig = (
                "***" if p < 0.001 else
                "**"  if p < 0.01  else
                "*"   if p < 0.05  else "ns"
            )
            print(f"\n     {metric}:")
            print(f"       Ovulatory:   {ov.mean():.2f} +/- {ov.std():.2f}  "
                  f"(n={len(ov)})")
            print(f"       Anovulatory: {an.mean():.2f} +/- {an.std():.2f}  "
                  f"(n={len(an)})")
            print(f"       Mann-Whitney p = {p:.4f} {sig}")
    else:
        print("     Too few anovulatory cycles for group comparison.")

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main(data_dir):
    print("=" * 60)
    print("  SEHAT-CYCLE: CARDIOVASCULAR AMPLITUDE ANALYSIS")
    print("=" * 60)

    # Step 1: Load
    horm, rhr, daily_hrv = load_data(data_dir)

    # Step 2: Segment cycles
    horm = segment_cycles(horm)

    # Step 3: Compute amplitudes
    amp = compute_amplitudes(horm, rhr, daily_hrv)
    print(f"\n  Cycles with amplitude computed: {len(amp)}")

    has_rhr = amp["rhr_amplitude"].notna().sum()
    has_pdg = amp["pdg_second_half_max"].notna().sum()
    has_both = amp.dropna(
        subset=["rhr_amplitude", "pdg_second_half_max"]
    ).shape[0]
    print(f"  Cycles with RHR amplitude: {has_rhr}")
    print(f"  Cycles with PdG data:      {has_pdg}")
    print(f"  Cycles with BOTH (usable): {has_both}")

    for col in ["rhr_amplitude", "rmssd_amplitude",
                "rhr_half_diff", "rmssd_half_diff"]:
        if col in amp.columns:
            v = amp[col].dropna()
            if len(v) > 0:
                print(f"    {col}: {v.mean():.2f} +/- {v.std():.2f} "
                      f"(n={len(v)})")

    # Step 4: Statistics
    results = run_statistics(amp)

    # Step 5: Export
    os.makedirs(TABLES_DIR, exist_ok=True)
    out_path = os.path.join(TABLES_DIR, "cycle_amplitudes.csv")
    amp.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # Summary for abstract
    print("\n" + "=" * 60)
    print("  ABSTRACT SUMMARY")
    print("=" * 60)
    if results:
        best = max(results.items(), key=lambda x: abs(x[1]["r"]))
        name, vals = best
        print(f"\n  Best metric: {name}")
        print(f"  Spearman r = {vals['r']:.3f}, "
              f"p = {vals['p']:.4f}, n = {vals['n']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sehat-Cycle: cardiovascular amplitude analysis"
    )
    parser.add_argument("data_dir", nargs="?", default=DEFAULT_DATA_DIR,
                        help="Path to mcPHASES data folder (default: auto-detect)")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"  ERROR: Data directory not found: {args.data_dir}")
        sys.exit(1)

    main(args.data_dir)