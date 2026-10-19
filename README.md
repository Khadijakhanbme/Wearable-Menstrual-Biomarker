# Wearable-Based Cardiovascular Amplitude as a Progesterone Biomarker for Menstrual Health

Using the [mcPHASES dataset](https://www.physionet.org/content/mcphases/) (42 participants, Fitbit Sense + daily urinary hormones), this project validates that **resting heart rate (RHR) amplitude correlates with luteal progesterone** (Spearman r = 0.275, p = 0.040, n = 56 cycles), bridging a gap between large-scale wearable studies and biochemical ground truth.

## Why This Project

As a South Asian woman and biomedical engineer, menstrual health is personal to me. In South Asian communities, menstrual cycles are rarely discussed openly, and conditions like Polycystic Ovary Syndrome (PCOS) often go undiagnosed for years, despite South Asian women being at disproportionately higher risk. Clinical hormone testing is inaccessible for many. If a consumer wearable on someone's wrist could flag irregular ovulation from cardiovascular signals alone, it could change how millions of women understand their own bodies without a single blood draw. That possibility is what drives this work.

## Background

Progesterone drives a well-documented rise in Resting Heart Rate (RHR) and fall in Heart Rate Variability (HRV) during the luteal phase. Recent work has quantified this at scale:

- **Jasinski et al. (2024)** introduced *cardiovascular amplitude*; the within-cycle swing of RHR and HRV as a digital biomarker, analysing 45,000+ cycles from WHOOP straps ([npj Digital Medicine](https://doi.org/10.1038/s41746-024-01394-0)). They proposed that reduced amplitude could flag anovulatory cycles, but had **no hormone measurements to validate the link**.
- **Oura Health (US Patent 20220313223A1)** patented anovulatory cycle detection from wearable data, relying on skin temperature slopes with HR/HRV as supplementary signals ([Google Patents](https://patents.google.com/patent/US20220313223A1/en)).
- **Goodale et al. (2019)** used the Ava bracelet to predict the fertile window from wrist-worn HR, HRV, and temperature (193 women, 708 cycles; [JMIR](https://doi.org/10.2196/13404)), but focused on ML-based fertile window detection rather than the amplitude–progesterone relationship.
- **Lang et al. (2024)** confirmed luteal RHR elevation using Fitbit in a cohort the same size as ours (n = 42), but without HRV or hormone data ([JMIR mHealth](https://doi.org/10.2196/50135)).

**What's missing is validation**: Does the cardiovascular swing actually track progesterone? The mcPHASES dataset uniquely provides both daily Fitbit cardiovascular data and concurrent urinary hormone assays (PdG, LH, E3G), letting us close that loop.

## Key Results

| Metric | Value |
|---|---|
| RHR amplitude vs luteal PdG | Spearman r = 0.275, p = 0.040 |
| Usable cycles | 56 (from 42 participants) |
| Ovulatory cycles (PdG ≥ 5.0) | 53 |
| Anovulatory cycles | 3 |
| Luteal RHR (group mean) | ~69 bpm (highest phase) |
| Follicular RHR (group mean) | ~65 bpm (lowest phase) |

Cycles with higher luteal progesterone show larger RHR swings. The three anovulatory cycles cluster at low PdG with moderate amplitude, consistent with the hypothesis that cardiovascular amplitude can serve as a non-invasive proxy for ovulation confirmation.

## Dataset

**mcPHASES** (Lin et al., 2025) — *Menstrual Cycle Physiological, Hormonal, and Self-Reported Events and Symptoms*

- 42 Canadian women, two 3-month study intervals
- Fitbit Sense: RHR, HRV (RMSSD), skin temperature, sleep, activity
- Mira Plus: daily urinary LH, E3G, and PdG (progesterone metabolite)
- 23 linked CSV tables, ~3.4 GB total
- Access: [PhysioNet (restricted)](https://www.physionet.org/content/mcphases/)

> Lin, B., Li, J. Y., Kalani, K., Truong, K., & Mariakakis, A. (2025). *mcPHASES: A Dataset of Physiological, Hormonal, and Self-reported Events and Symptoms for Menstrual Health Tracking with Wearables* (version 1.0.0). PhysioNet. https://doi.org/10.13026/zx6a-2c81

## Pipeline

The analysis runs in three steps. All scripts are self-contained (no shared config file) with the data path hardcoded — just run them in order:

```
explore_data.py  →  analysis.py  →  figures.py
   (Step 1)           (Step 2)        (Step 3)
```

### Step 1 — Explore (`explore_data.py`)

Exploratory overview of the dataset. Prints a summary table and generates 3 figures:

1. **One participant** (P10): PdG, RHR, and RMSSD day by day, with menstrual phases shaded
2. **Average cycle**: the same signals by cycle day, each person normalised to their own mean
3. **Phase summary**: within-person change in RHR and RMSSD per phase (dot plot with error bars)

```bash
python explore_data.py
```

### Step 2 — Analyse (`analysis.py`)

Core analysis pipeline:

1. Loads RHR (from `resting_heart_rate.csv`), HRV, and hormone tables
2. Segments menstrual cycles using phase transitions (new cycle = transition into Menstrual)
3. Computes per-cycle metrics:
   - `rhr_amplitude` = max(daily RHR) − min(daily RHR)
   - `rmssd_amplitude` = max(RMSSD) − min(RMSSD)
   - `rhr_half_diff` = mean(2nd half RHR) − mean(1st half RHR)
   - `pdg_second_half_max` = max PdG in the second half of the cycle
4. Runs Spearman correlations (amplitude vs PdG) and Mann-Whitney group comparisons (ovulatory vs anovulatory)
5. Exports `results/tables/cycle_amplitudes.csv`

```bash
python analysis.py
```

### Step 3 — Figures (`figures.py`)

Generates 3 publication-quality figures (PNG + PDF):

1. **Phase-level bars** — Two-panel bar chart of RHR and RMSSD by menstrual phase (n = 42). Sets up the mechanism: Luteal RHR peaks, HRV dips.
2. **Example ovulatory cycle** — One participant's RHR trajectory coloured by phase (top) with PdG bars (bottom). Automatically selects the best-scoring cycle based on amplitude, PdG clarity, phase order, and coverage.
3. **Cycle-level scatter** — RHR amplitude vs luteal PdG, coloured by ovulatory/anovulatory status. The key result (r = 0.275, p = 0.040).

```bash
python figures.py
```

Figures are saved to `results/figures/`.

## Project Structure

```
wearable-menstrual-biomarker/
├── explore_data.py        # Step 1: data exploration (3 figures)
├── analysis.py            # Step 2: cycle segmentation + amplitude analysis
├── figures.py             # Step 3: publication figures (3 figures)
├── requirements.txt       # Python dependencies
├── README.md
└── results/
    ├── tables/
    │   └── cycle_amplitudes.csv   # per-cycle metrics (generated by Step 2)
    └── figures/
        ├── fig1_phase_bars.png/pdf
        ├── fig2_example_cycle.png/pdf
        └── fig3_cycle_scatter.png/pdf
```

## Setup

```bash
# Clone the repo
git clone https://github.com/your-username/wearable-menstrual-biomarker/
cd menstrual-cv-amplitude

# Install dependencies
pip install -r requirements.txt

# Download the mcPHASES dataset from PhysioNet (requires credentialed access)
# Place the dataset folder one level above this directory, or edit DEFAULT_DATA_DIR
# in each script to point to your data location.

# Run the pipeline
python explore_data.py
python analysis.py
python figures.py
```

**Requirements:** Python 3.9+, pandas ≥ 2.0, numpy ≥ 1.24, matplotlib ≥ 3.7, scipy ≥ 1.10

## References

1. Jasinski, S. R. et al. (2024). A novel method for quantifying fluctuations in wearable-derived daily cardiovascular parameters across the menstrual cycle. *npj Digital Medicine*, 7. https://doi.org/10.1038/s41746-024-01394-0

2. Goodale, B. M. et al. (2019). Wearable sensors reveal menses-driven changes in physiology and enable prediction of the fertile window. *JMIR*, 21(4), e13404. https://doi.org/10.2196/13404

3. Lang, A. et al. (2024). Feasibility study on menstrual cycles with Fitbit device (FEMFIT). *JMIR mHealth and uHealth*, 12, e50135. https://doi.org/10.2196/50135

4. Smarr, B. L. et al. (2019). Nocturnal finger skin temperature in menstrual cycle tracking: ambulatory pilot study using a wearable Oura ring. *BMC Women's Health*, 19, 150. https://doi.org/10.1186/s12905-019-0844-9

5. Thigpen, N. N. et al. (2022). Anovulatory cycle detection from wearable-based physiological data. US Patent Application 20220313223A1. https://patents.google.com/patent/US20220313223A1

6. Lin, B. et al. (2025). mcPHASES: A dataset of physiological, hormonal, and self-reported events and symptoms for menstrual health tracking with wearables (version 1.0.0). PhysioNet. https://doi.org/10.13026/zx6a-2c81

## Author

**Khadija Khan**

MSc Biomedical Engineering 

## License

This project is for academic and research purposes. The mcPHASES dataset is subject to the [PhysioNet Restricted Health Data Use Agreement](https://www.physionet.org/content/mcphases/).
