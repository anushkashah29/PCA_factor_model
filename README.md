# Treasury Yield PCA Factor Model & Value-at-Risk

A self-contained Python model that pulls live US Treasury yield data, decomposes the yield curve into its three dominant risk factors using Principal Component Analysis, maps a bond portfolio onto those factors, and estimates 10-day 99% Value-at-Risk.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Methodology Overview](#2-methodology-overview)
3. [Data](#3-data)
4. [Step-by-Step Implementation](#4-step-by-step-implementation)
   - [Step 1 — Data Collection](#step-1--data-collection)
   - [Step 2 — Data Preparation](#step-2--data-preparation)
   - [Step 3 — Daily Yield Changes](#step-3--daily-yield-changes)
   - [Step 4 — Covariance Matrix & Eigendecomposition](#step-4--covariance-matrix--eigendecomposition)
   - [Step 5 — Factor Extraction (PCA)](#step-5--factor-extraction-pca)
   - [Step 6 — Portfolio DV01 & Factor Risk Weights](#step-6--portfolio-dv01--factor-risk-weights)
   - [Step 7 — VaR Calculation](#step-7--var-calculation)
5. [Key Results](#5-key-results)
6. [Output Files](#6-output-files)
   - [Excel Workbook — Sheet Guide](#excel-workbook--sheet-guide)
   - [Charts (pca_plots/)](#charts-pca_plots)
7. [How to Run](#7-how-to-run)
8. [Configuration & Customisation](#8-configuration--customisation)
9. [Dependencies](#9-dependencies)
10. [Theoretical Background](#10-theoretical-background)

---

## 1. Problem Statement

Interest rate risk is the dominant source of risk for fixed-income portfolios. A portfolio of Treasury bonds is exposed to movements across the entire yield curve — a parallel shift up, a steepening or flattening, or a change in curvature can each produce very different P&L outcomes.

Naive approaches model each maturity independently, which ignores the strong correlations between tenors and leads to over-counting of risk. The PCA factor model solves this by identifying the smallest number of uncorrelated "super-factors" that explain the vast majority of yield curve variance — historically, just three factors explain ~95% of all yield curve movements.

**This model answers two questions:**
1. What are the three principal risk factors driving the US Treasury yield curve, and how much variance does each explain?
2. Given a portfolio of 2-year and 10-year Treasury bonds, what is the 10-day 99% VaR expressed in dollar terms?

---

## 2. Methodology Overview

```
FRED API  →  Raw CMT Yields  →  Daily Changes (bps)
                                        │
                               Covariance Matrix Σ
                                        │
                               Eigendecomposition
                               λ₁, λ₂, λ₃ … λ₁₀
                               e₁, e₂, e₃ … e₁₀
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
              PC Scores                            Portfolio DV01
         (Level, Slope, Curve)              ($ per bp at each tenor)
                    │                                       │
                    └────────────────┬──────────────────────┘
                                     │
                              Factor Risk Weights
                              w_k = DV01 · e_k
                                     │
                         10-day Monte Carlo Simulation
                         (normal draws, sqrt-of-time scaling)
                                     │
                              P&L Distribution
                                     │
                           99% VaR  ≈  $390,000
```

---

## 3. Data

| Property | Detail |
|---|---|
| Source | Federal Reserve Economic Data (FRED), St. Louis Fed |
| Series | Constant Maturity Treasury (CMT) yields |
| Maturities | 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y |
| FRED IDs | DGS3MO, DGS6MO, DGS1, DGS2, DGS3, DGS5, DGS7, DGS10, DGS20, DGS30 |
| Frequency | Daily |
| Start Date | 2 January 2007 |
| End Date | Current (fetched live at run time) |
| Coverage after cleaning | ~4,864 business days |
| Unit | Percent (e.g., 4.25 = 4.25%) |

CMT yields are published by the US Treasury and represent the yield a bond would carry if it were priced at par with a given remaining maturity. They are the market standard for yield-curve analysis.

---

## 4. Step-by-Step Implementation

### Step 1 — Data Collection

**What:** Fetch daily CMT yields for all 10 maturities from the FRED REST API.

**How:** Each series is retrieved individually via `GET /fred/series/observations` with the API key, start/end dates, and JSON format. Observations marked `"."` (FRED's missing-data sentinel) are converted to `NaN`.

**Output:** A 10-column DataFrame indexed by date — one column per maturity tenor.

```
FRED endpoint: https://api.stlouisfed.org/fred/series/observations
Parameters   : series_id, api_key, observation_start, observation_end
```

---

### Step 2 — Data Preparation

**What:** Remove non-trading days and fill isolated missing values to produce a clean, complete panel.

**Rules applied in order:**

| Rule | Reason |
|---|---|
| Drop weekends (`dayofweek >= 5`) | Markets closed; FRED sometimes includes Saturday entries |
| Drop US Federal holidays | Per `pandas.tseries.holiday.USFederalHolidayCalendar` — New Year's, MLK Day, Presidents' Day, Memorial Day, Juneteenth, Independence Day, Labor Day, Columbus Day, Veterans Day, Thanksgiving, Christmas |
| Drop rows where **all** tenors are NaN | Truly missing dates with no data at any tenor |
| Forward-fill isolated NaN cells | Some tenors (e.g., 20Y) have occasional gaps; last known value used |
| Drop any remaining rows with NaN | Ensures a fully populated matrix for covariance computation |

**Output:** ~4,864 clean business days from 2007-01-02 to the current date.

---

### Step 3 — Daily Yield Changes

**What:** Convert the level series to a first-difference series, then scale to basis points.

**Formula:**

```
Δy(t, τ) = [y(t, τ) − y(t−1, τ)] × 100   (bps)
```

where `y(t, τ)` is the yield at date `t` for tenor `τ` expressed as a percentage (e.g., 4.25).

Multiplying by 100 converts percentage-point changes to basis points (1 bp = 0.01%). All subsequent covariance and PCA computations operate in basis points.

**Output:** A (4,863 × 10) matrix of daily yield changes in bps.

> The level data and change data are both exported to Excel (Sheets 1 and 2) so you can visualise the raw time series alongside the differenced series.

---

### Step 4 — Covariance Matrix & Eigendecomposition

**What:** Summarise the co-movement structure of yield changes across all ten tenors.

#### Covariance Matrix

```
Σ = (1 / (T−1)) × ΔY' ΔY       shape: (10 × 10)
```

Each entry `Σ[i,j]` is the sample covariance (in bps²) between daily changes at tenor `i` and tenor `j`. The diagonal contains the variance of each tenor's daily change.

#### Eigendecomposition

```
Σ = E Λ E'

where:
  E = matrix of eigenvectors  (10 × 10, columns are eigenvectors)
  Λ = diagonal matrix of eigenvalues  (λ₁ ≥ λ₂ ≥ … ≥ λ₁₀)
```

Computed via `numpy.linalg.eigh` (exploits symmetry of Σ for numerical stability). Eigenvalues and their corresponding eigenvectors are sorted in descending order by eigenvalue.

**Interpretation of eigenvalues:**

```
% variance explained by PCₖ = λₖ / Σλᵢ × 100
```

| PC | Eigenvalue (bps²) | Variance Explained | Cumulative |
|---|---|---|---|
| PC1 Level | 208.89 | 75.30% | 75.30% |
| PC2 Slope | 38.37 | 13.83% | 89.13% |
| PC3 Curvature | 17.64 | 6.36% | 95.49% |
| PC4–PC10 | residual | 4.51% | 100% |

Three factors capture 95.5% of all yield curve variance. This is a well-established empirical regularity in interest rate markets.

---

### Step 5 — Factor Extraction (PCA)

**What:** Project every historical daily yield change onto the first three eigenvectors to obtain a PC score for each trading day. These 4,863 daily PC scores are used internally by the Historical Simulation VaR engine (Step 7). They are **not** exported as a daily time series to Excel — instead, Sheet 6 (`6_PC_Summary`) exports only the **summary statistics** for the full 2007–2026 period (standard deviation, mean, min, max, skewness, kurtosis) which are then referenced by the VaR formula sheet. A PNG chart (`04_pc_timeseries.png`) is saved separately for visual inspection of the daily history.

**Formula (computed in Python for all T = 4,863 trading days):**

```
PC(t) = ΔY(t) × E[:,0:3]       shape: (T × 3)
```

Each PC score at time `t` is the inner product of that day's 10 yield changes with the factor's eigenvector — it represents how strongly that factor moved on that particular day. The distribution of these scores over the full history is what drives the Historical Simulation VaR.

**What goes into Excel vs what stays in Python:**

| Item | Python | Excel |
|------|--------|-------|
| Daily PC scores (4,863 × 3 array) | Computed — used for HistSim VaR and chart | **Not exported** |
| PC summary stats (std dev, mean, min, max, skew, kurtosis) | Derived from daily scores | **Sheet 6** `6_PC_Summary` |
| Daily PC time-series chart | — | **`04_pc_timeseries.png`** (PNG only) |

#### Economic interpretation of the three factors

| Factor | Shape of Eigenvector | Economic Meaning |
|---|---|---|
| **PC1 Level** | Loadings roughly equal across all tenors | A parallel shift of the entire yield curve up or down. When PC1 moves, all yields move together in the same direction and by similar magnitudes. Controls ~75% of variance. |
| **PC2 Slope** | Loadings positive at short end, negative at long end (or vice versa) | A steepening or flattening of the curve. Short yields and long yields move in opposite directions. Controls ~14% of variance. |
| **PC3 Curvature** | Loadings positive at short and long ends, negative in the middle (butterfly shape) | The middle of the curve moves relative to the wings. Controls ~6% of variance. |

**PC Summary Statistics for full period 2007–2026 (Sheet 6 `6_PC_Summary`):**

| Statistic | PC1 Level | PC2 Slope | PC3 Curvature |
|---|---|---|---|
| 1-Day Std Dev | ~14.5 bps | ~6.2 bps | ~4.2 bps |
| 10-Day Std Dev | ~45.7 bps | ~19.6 bps | ~13.3 bps |

---

### Step 6 — Portfolio DV01 & Factor Risk Weights

**What:** Translate the abstract PC factor movements into dollar P&L for a specific bond portfolio.

#### Portfolio

| Bond | Notional | Coupon | Tenor |
|---|---|---|---|
| 2Y US T-Note | $10,000,000 | 4.50% | 2 years |
| 10Y US T-Note | $10,000,000 | 4.20% | 10 years |

#### Dollar Value of a Basis Point (DV01)

DV01 is the change in a bond's dollar value when its yield moves by exactly 1 basis point.

**Formula (full bond pricing):**

```
Price = Σ [C/freq / (1 + ytm/freq)^t]  +  Face / (1 + ytm/freq)^n
      t=1..n

DV01 = Price(ytm) − Price(ytm + 0.01%)
```

**In Excel (Sheet 7):**

```excel
DV01 = (Notional/100) × PRICE(settlement, maturity, coupon, ytm, 100, 2)
                       × MDURATION(settlement, maturity, coupon, ytm, 2)
                       / 10,000
```

| Bond | DV01 (USD per bp) |
|---|---|
| 2Y T-Note | $1,906 |
| 10Y T-Note | $7,871 |

The 10Y bond has ~4× the rate sensitivity of the 2Y bond despite the same notional, reflecting its longer duration.

#### Factor Risk Weights

The DV01 at each tenor forms a sparse 10-element vector `d` (non-zero only at the 2Y and 10Y positions). Projecting this onto each eigenvector gives the portfolio's dollar sensitivity to a 1-bp movement of each PC:

```
w_k = d · e_k  =  Σᵢ DV01ᵢ × e_k[i]
               =  DV01_2Y × e_k[2Y] + DV01_10Y × e_k[10Y]
```

| Factor | Risk Weight (USD/bp) | Interpretation |
|---|---|---|
| PC1 Level | −$3,628 | Portfolio loses ~$3,628 per bp of parallel rate rise |
| PC2 Slope | +$1,101 | Portfolio gains ~$1,101 per bp of slope widening |
| PC3 Curvature | +$115 | Small curvature exposure |

The negative PC1 weight is expected: a long bond portfolio loses value when yields rise uniformly.

> **In Excel (Sheet 7, Section C):** Risk weights are live formulas linking to the eigenvector values in Sheet 5 and the DV01 values computed by PRICE/MDURATION in Section B. Changing the coupon, YTM, or notional in the yellow input cells automatically updates all downstream risk weights and VaR.

---

### Step 7 — VaR Calculation

**What:** Estimate the 10-day 99% P&L distribution and extract the loss at the tail. The primary method is **Historical Simulation** — it uses every actual trading day as a scenario with no distributional assumption. Two parametric methods (Analytical and Monte Carlo) serve as cross-checks.

#### Why PCs simplify VaR

Because the eigenvectors are orthogonal, the three PC scores are uncorrelated by construction. This means the portfolio variance decomposes cleanly:

```
Var(P&L) = w₁² σ₁² T  +  w₂² σ₂² T  +  w₃² σ₃² T
```

No cross-factor covariance terms. No matrix inversion. Each factor contributes independently.

#### Square-Root-of-Time Scaling

Daily volatilities and daily P&Ls are scaled to the 10-day horizon using the square-root-of-time rule (assumes i.i.d. returns):

```
σ_k (10-day) = σ_k (1-day) × √10        P&L_10d = P&L_daily × √10
```

---

#### Method 1 — Historical Simulation  *(primary)*

Uses every actual trading day in the dataset as a scenario. No distributional assumption — the empirical P&L distribution determines the VaR directly from observed market moves.

**Four-step P&L formula for each historical trading day:**

```
Step 1 — Daily yield changes (bps):
    Δy_i,t = (Yield_i,t − Yield_i,t−1) × 100    for each tenor i = 1 … 10

Step 2 — PC scores (dot product with eigenvectors):
    PC_k,t = Σᵢ  Δy_i,t × e_k,i                 summed across all 10 tenors

Step 3 — Daily P&L (sign-corrected for long bond portfolio):
    P&L_t = −( PC1_t × w₁  +  PC2_t × w₂  +  PC3_t × w₃ )

Step 4 — Scale to 10-day horizon:
    P&L_10d = P&L_t × √10
```

The 99% VaR is the absolute value of the 1st percentile of the 4,863 daily P&L observations scaled to 10 days.

---

##### Worked Example — Worst Day: 2008-09-19 (Lehman Aftermath)

Lehman Brothers collapsed on 15 September 2008. Four days later, short-term Treasury yields spiked as money-market stress peaked, while longer yields also rose as the Fed's emergency support was not yet certain.

**Step 1 — Raw yield levels and daily changes**

| Tenor | Yield (Sep 18) | Yield (Sep 19) | Change Δy (bps) |
|-------|---------------|---------------|-----------------|
| 3M  | 0.2300% | 0.9900% | **+76.00** |
| 6M  | 0.7900% | 1.5400% | **+75.00** |
| 1Y  | 1.5300% | 2.0500% | +52.00 |
| 2Y  | 1.7800% | 2.1600% | +38.00 |
| 3Y  | 2.0500% | 2.4200% | +37.00 |
| 5Y  | 2.6700% | 3.0100% | +34.00 |
| 7Y  | 3.0800% | 3.3700% | +29.00 |
| 10Y | 3.5400% | 3.7800% | +24.00 |
| 20Y | 4.1900% | 4.4200% | +23.00 |
| 30Y | 4.1400% | 4.3600% | +22.00 |

**Step 2 — PC1 and PC2 scores (from Sheet 5 eigenvector loadings)**

Formula: PC_k = Σ ( Δy_i × loading_k,i ) across all 10 tenors

*PC1 — Level factor (parallel shift):*

| Tenor | Δy (bps) | × | PC1 Loading | = | Product |
|-------|----------|---|-------------|---|---------|
| 3M  | +76.00 | × | −0.100380 | = | −7.629 |
| 6M  | +75.00 | × | −0.130077 | = | −9.756 |
| 1Y  | +52.00 | × | −0.193589 | = | −10.067 |
| 2Y  | +38.00 | × | −0.312981 | = | −11.893 |
| 3Y  | +37.00 | × | −0.357671 | = | −13.234 |
| 5Y  | +34.00 | × | −0.403546 | = | −13.721 |
| 7Y  | +29.00 | × | −0.411205 | = | −11.925 |
| 10Y | +24.00 | × | −0.385077 | = | −9.242 |
| 20Y | +23.00 | × | −0.347759 | = | −7.999 |
| 30Y | +22.00 | × | −0.329366 | = | −7.246 |
| **SUM** | | | | | **−102.710** |

*PC2 — Slope factor (short vs long end):*

| Tenor | Δy (bps) | × | PC2 Loading | = | Product |
|-------|----------|---|-------------|---|---------|
| 3M  | +76.00 | × | −0.547436 | = | −41.605 |
| 6M  | +75.00 | × | −0.441904 | = | −33.143 |
| 1Y  | +52.00 | × | −0.397768 | = | −20.684 |
| 2Y  | +38.00 | × | −0.265205 | = | −10.078 |
| 3Y  | +37.00 | × | −0.172037 | = | −6.365 |
| 5Y  | +34.00 | × | −0.004611 | = | −0.157 |
| 7Y  | +29.00 | × | +0.116298 | = | +3.373 |
| 10Y | +24.00 | × | +0.204114 | = | +4.899 |
| 20Y | +23.00 | × | +0.301666 | = | +6.938 |
| 30Y | +22.00 | × | +0.317290 | = | +6.980 |
| **SUM** | | | | | **−89.842** |

> PC2 is strongly negative because the short end spiked far more than the long end (+76 bps at 3M vs +22 bps at 30Y). The PC2 loadings capture this flattening: large negative weights at the short end, positive at the long end.

**Step 3 — Daily P&L (with sign correction)**

Risk weights from Sheet 7: w₁ = −$3,627.58/bp, w₂ = +$1,101.19/bp, w₃ = +$115.47/bp

```
Gross = PC1 × w₁   +   PC2 × w₂   +   PC3 × w₃
      = (−102.710 × −3,627.58)  +  (−89.842 × +1,101.19)  +  (+40.790 × +115.47)
      =   +372,590               +    −98,933               +     +4,710
      =   +278,367

Daily P&L = −Gross = −$278,367   ← LOSS (correct: yields rose, long bonds fell)
```

> The negation is required because `numpy.linalg.eigh` orients all PC1 loadings negative. A uniform yield rise produces a negative PC1 score; multiplying by a negative risk weight gives a positive gross — which would incorrectly appear as a gain. Negating restores: **yield rise → loss for a long bond portfolio**.

**Step 4 — Scale to 10-day horizon**

```
P&L_10d = −$278,367 × √10 = −$278,367 × 3.1623 = −$880,275
```

This is the **#1 worst scenario** in Sheet 10. The 99th-percentile loss (VaR) is the day at the 1% tail of all 4,863 such observations.

---

**Why Historical VaR exceeds the Analytical result:**

The actual daily P&L series has **excess kurtosis of 2.62** (a normal distribution has 0). The GFC (2008), COVID (2020), and the 2022 Fed hiking cycle all generated yield spikes far beyond what a normal distribution predicts, pushing the empirical 1st percentile further into the left tail:

| Method | 10-Day 99% VaR | Assumption | vs Analytical |
|--------|---------------|------------|---------------|
| **Historical Simulation** | **$426,210** | **Empirical — no distribution assumed** | **+9.6%** |
| Analytical | $388,969 | Normal distribution, closed-form | baseline |
| Parametric MC | $389,867 | Normal draws (same as Analytical) | +0.2% |

---

#### Method 2 — Analytical Cross-Check  *(closed-form parametric)*

The exact VaR formula under the normal distribution assumption:

```
VaR (analytical) = z₀.₉₉ × √(w₁²σ₁²T + w₂²σ₂²T + w₃²σ₃²T)

where z₀.₉₉ = NORM.S.INV(0.99) ≈ 2.326
```

**In Excel (Sheet 8):**

```excel
= NORM.S.INV($B$4) * SQRT(($D$9*$C$9)^2 + ($D$10*$C$10)^2 + ($D$11*$C$11)^2)
```

where `B4` = confidence, `C9:C11` = 10-day std devs, `D9:D11` = risk weights — all linked from Sheets 6 and 7.

---

#### Method 3 — Parametric Monte Carlo  *(secondary cross-check)*

10,000 paths of 10-day PC movements drawn from independent normal distributions — identical distributional assumption to Method 2:

```
PC_k_move ~ N(0, σ_k × √10)    for k = 1, 2, 3
P&L = w₁ × PC₁_move  +  w₂ × PC₂_move  +  w₃ × PC₃_move
```

Both Method 2 and Method 3 assume normality and converge to the same result (~$389K). They are included as internal consistency checks. All Monte Carlo paths are computed in Python memory; only the final VaR scalar appears in Sheet 8.

**In Excel (Sheet 10 — `10_HistSim_PnL`):**

The Historical Simulation audit trail is sorted worst-to-best with 8 columns: Rank, Date, PC1/2/3 Move (bps), Daily P&L ($), 10-Day Scaled P&L ($), Cumulative Percentile. All values are Python-computed static numbers. VaR tail rows are highlighted red; the exact boundary row is bold. To trace any row back to raw yields, cross-reference the date in Sheet 2 (`2_Daily_Changes_bps`) and apply the eigenvector loadings from Sheet 5 (`5_Eigenvectors`) using the step-by-step formula above.

---

## 5. Key Results

| Metric | Value |
|---|---|
| Dataset | 4,864 business days, 2007-01-02 to present |
| PC1 (Level) variance explained | 75.30% |
| PC2 (Slope) variance explained | 13.83% |
| PC3 (Curvature) variance explained | 6.36% |
| 3-factor cumulative | **95.49%** |
| 2Y T-Note DV01 | $1,906 / bp |
| 10Y T-Note DV01 | $7,871 / bp |
| PC1 Level risk weight | −$3,628 / bp of PC |
| PC2 Slope risk weight | +$1,101 / bp of PC |
| PC3 Curvature risk weight | +$115 / bp of PC |
| **99% VaR, 10-day — Historical Simulation** | **$426,210** |
| 99% VaR, 10-day — Analytical (parametric cross-check) | $388,969 |
| 99% VaR, 10-day — Monte Carlo (parametric cross-check) | $389,867 |
| Empirical P&L excess kurtosis | 2.62 (vs 0 for normal) |
| Historical vs Analytical VaR premium | +9.6% (fat tails) |
| Worst scenario (Rank 1) | 2008-09-19 (Lehman): PC1=−102.71, PC2=−89.84 → Daily P&L=−$278,367 → 10-Day=−$880,275 |

Historical Simulation ($426,210) is the primary VaR figure — it captures fat tails from the GFC (2008), COVID (2020), and the 2022 Fed hiking cycle that the normal distribution (Analytical and MC, both ~$389K) cannot account for.

---

## 6. Output Files

### Excel Workbook — Sheet Guide

Output file: `treasury_pca_var.xlsx`

| Sheet | Name | Contents |
|---|---|---|
| 1 | `1_Raw_Yields` | Daily CMT yields in % for all 10 tenors, 2007 to present. 4,864 rows. |
| 2 | `2_Daily_Changes_bps` | Daily first differences of yields converted to basis points. 4,863 rows. |
| 3 | `3_Covariance_Matrix` | 10×10 sample covariance matrix of daily yield changes (bps²). |
| 4 | `4_Eigenvalues` | All 10 eigenvalues with individual and cumulative variance explained. First 3 highlighted green. |
| 5 | `5_Eigenvectors` | Factor loadings (eigenvectors) for the first 3 PCs at each of the 10 tenors. Referenced by Sheet 7 formulas. |
| 6 | `6_PC_Summary` | Summary statistics for PC1/2/3 over the full 2007–2026 period: eigenvalue, % variance, std dev (1-day and 10-day), mean, min, max, skewness, excess kurtosis. Referenced by Sheet 8 formulas. |
| 7 | `7_DV01_Risk_Weights` | Three-section formula sheet. **Section A** (yellow cells): editable bond parameters. **Section B**: DV01 computed live via Excel `PRICE` and `MDURATION`. **Section C**: Factor risk weights linked to Sheets 5 and 6. |
| 8 | `8_VaR_Results` | Four-section formula sheet. **Section A**: editable confidence and horizon. **Section B**: PC stats linked to Sheets 6 & 7. **Section C**: Analytical VaR formula. **Section D**: Standalone PC VaR contributions with diversification benefit. |
| 9 | `9_PC_Correlations` | Correlation matrix of the simulated PC moves — should be identity (confirms orthogonality). |
| 10 | `10_HistSim_PnL` | Full Historical Simulation audit trail, sorted worst-to-best. 8 columns: Rank, Date, PC1/2/3 Move (bps), Daily P&L ($), 10-Day Scaled P&L ($), Cumulative Percentile. Red rows = VaR tail. Bold boundary row = exact VaR cutoff day. All values are Python-computed static numbers. Cross-reference dates against Sheet 2 and eigenvectors in Sheet 5 to reconstruct any PC score from raw yields (see Step 7 worked example). |

#### Formula Chain in Excel

```
Sheet 5 (Eigenvectors)
        │
        ├─ B7, C7, D7  →  PC1/2/3 loading at 2Y tenor
        └─ B11, C11, D11  →  PC1/2/3 loading at 10Y tenor
                │
                ▼
Sheet 6 (PC Summary)
        E4, E5, E6  →  1-day std dev for PC1, PC2, PC3
                │
                ▼
Sheet 7 (Portfolio)
        Section B:  PRICE × MDURATION → DV01
        Section C:  Loading × DV01 → Risk Weight (F15, F16, F17)
                │
                ▼
Sheet 8 (VaR)
        B9:B11   ← '6_PC_Summary'!E4:E6      (1-day std)
        C9:C11    = B9 * SQRT($B$5)           (10-day std)
        D9:D11   ← '7_DV01_Risk_Weights'!F15:F17  (risk weights)
        E9:E11    = ABS(C * D)                (10-day $ risk std)
        B17       = NORM.S.INV(B4) * SQRT(E9²+E10²+E11²)  (Analytical VaR)
```

Changing the highlighted yellow input cells in Sheet 7 (notional, coupon, YTM) or Sheet 8 (confidence, horizon) cascades through all formulas automatically.

---

### Charts (pca_plots/)

Five PNG charts are saved to the `pca_plots/` subdirectory:

| File | Chart | Description |
|---|---|---|
| `01_yield_curves.png` | Line chart | Annual snapshots of the full yield curve from 2007 to present. Colour gradient shows the passage of time (viridis scale). Illustrates how the level and shape of the curve evolved through the GFC, zero-rate era, and post-2022 hiking cycle. |
| `02_scree_plot.png` | Bar + line | Individual variance explained (bars, left axis) and cumulative (red line, right axis) for all 10 PCs. The 95% threshold line makes it visually clear that 3 factors suffice. |
| `03_factor_loadings.png` | Line chart | The three eigenvectors plotted against maturity. PC1 is nearly flat (parallel shift), PC2 slopes from positive to negative (tilt), PC3 has a hump shape (curvature/butterfly). |
| `04_pc_timeseries.png` | 3-panel time series | Daily PC scores for PC1, PC2, and PC3 over the full 2007–2026 history — visualised here for inspection but **not exported to Excel** (Sheet 6 holds only the summary statistics). Large spikes in PC1 during 2022–2023 reflect the Fed's rapid tightening cycle. |
| `05_pnl_distribution.png` | Histogram | Simulated 10-day P&L distribution for the portfolio (10,000 paths). The red dashed line marks the 99% VaR loss level. The approximately normal shape validates the parametric assumption. |

---

## 7. How to Run

**Prerequisites:** Python 3.9+ with the `.venv` virtual environment already created in the project folder.

```bash
# 1. Activate the virtual environment (Windows)
.venv\Scripts\activate

# 2. Install dependencies (first time only)
pip install pandas numpy scipy openpyxl requests matplotlib

# 3. Run the model
python pca_treasury_var.py
```

The script prints live progress to the console and takes approximately 30–60 seconds (dominated by 10 FRED API calls, one per tenor).

**Expected console output:**

```
============================================================
  Treasury Yield PCA Factor Model & VaR
============================================================

[1/7] Fetching FRED data ...
  Fetching 3M  (DGS3MO) ... 5078 obs
  ...
[2/7] Cleaning data (removing holidays & NaNs) ...
  4864 business days  |  2007-01-02 to 2026-06-17
[3/7] Computing daily yield changes (bps) ...
[4/7] Eigendecomposition of covariance matrix ...
  PC1: eigenvalue=208.8914  (75.30% variance explained)
  PC2: eigenvalue=38.3688   (13.83% variance explained)
  PC3: eigenvalue=17.6376   (6.36% variance explained)
[5/7] Extracting principal components ...
[6/7] Computing DV01 & factor risk weights ...
  2Y:  DV01 = $1,905.93 / bp
  10Y: DV01 = $7,871.31 / bp
[7/7] Computing VaR (99%, 10-day) ...
  Simulated VaR  :  $389,867
  Analytical VaR :  $388,969
[+]  Generating plots ...
[+]  Writing Excel workbook ...
  Saved: treasury_pca_var.xlsx
```

**Output location:** Both output files are written to the working directory.

```
PCAFactor/
├── pca_treasury_var.py          ← model script
├── treasury_pca_var.xlsx        ← 9-sheet Excel workbook
└── pca_plots/
    ├── 01_yield_curves.png
    ├── 02_scree_plot.png
    ├── 03_factor_loadings.png
    ├── 04_pc_timeseries.png
    └── 05_pnl_distribution.png
```

---

## 8. Configuration & Customisation

All parameters are defined at the top of `pca_treasury_var.py`:

| Variable | Default | Description |
|---|---|---|
| `FRED_API_KEY` | *(provided)* | FRED API key. Free registration at fred.stlouisfed.org |
| `START_DATE` | `"2007-01-01"` | History start date |
| `END_DATE` | `date.today()` | Automatically set to run date |
| `OUTPUT_FILE` | `"treasury_pca_var.xlsx"` | Output workbook name |
| `PORTFOLIO` | 2Y: $10M, 10Y: $10M | Bond notionals in USD |
| `COUPON_RATES` | 2Y: 4.50%, 10Y: 4.20% | Fixed coupon rates |
| `CONFIDENCE` | `0.99` | VaR confidence level |
| `HORIZON` | `10` | VaR holding period in days |
| `N_SIM` | `10,000` | Monte Carlo simulation paths |

In addition, the **yellow input cells** in the Excel workbook (Sheets 7 and 8) can be edited directly to reprice the portfolio and recompute VaR without re-running the script — all downstream cells use live Excel formulas.

---

## 9. Dependencies

| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 2.0 | Data wrangling, date handling, holiday calendar |
| `numpy` | ≥ 1.24 | Matrix algebra, eigendecomposition, simulation |
| `scipy` | ≥ 1.10 | `stats.norm.ppf`, skewness, kurtosis |
| `requests` | ≥ 2.28 | FRED REST API calls |
| `openpyxl` | ≥ 3.1 | Excel workbook creation, formatting, formulas |
| `matplotlib` | ≥ 3.7 | Chart generation (saved as PNG) |

All are installable via `pip install pandas numpy scipy openpyxl requests matplotlib`.

---

## 10. Theoretical Background

### Why PCA for yield curves?

Yield curve movements across different maturities are highly correlated — when the Fed raises rates, short-end yields move more than long-end yields, but both move in the same direction. This high correlation means a 10-dimensional yield curve system can be effectively reduced to 3 dimensions without losing material information.

PCA finds the linear combinations of the original variables (tenor yields) that successively maximise explained variance, subject to orthogonality constraints. The resulting factors are uncorrelated by construction, which greatly simplifies VaR aggregation.

### Why does the parallel shift explain 75% of variance?

Across economic cycles, the dominant source of yield curve volatility is the central bank policy rate — it moves the whole curve up or down. Slope and curvature changes (driven by growth expectations, term premiums, and risk appetite) are secondary effects. This is why Level always dominates in empirical studies regardless of market.

### Assumptions and limitations

| Assumption | Impact if violated |
|---|---|
| **Normal distribution of PC scores** | Fat tails in actual data (crisis periods) mean the normal-VaR understates tail losses. Consider historical simulation VaR as a complement. |
| **Square-root-of-time scaling** | Assumes i.i.d. daily returns. In practice, volatility is autocorrelated (GARCH effects). The 10-day VaR may be understated during stress periods. |
| **Static eigenvectors** | The PCA is estimated over the full 2007–2026 window. In reality, the factor structure shifts over time (e.g., pre- and post-GFC). Rolling-window PCA would capture regime changes. |
| **Coupon-bearing bonds at par** | The DV01 calculation uses a fixed coupon approximation. For seasoned bonds trading significantly off par, the actual price and modified duration will differ. |
| **No convexity adjustment** | For large parallel shifts (>50 bps), the linear DV01 approximation breaks down. Adding a gamma/convexity term improves accuracy for stress scenarios. |

### Diversification in the PCA VaR framework

Because PCs are orthogonal, total portfolio variance is the sum of individual PC variances:

```
Var(P&L) = Σₖ (wₖ σₖ √T)²     (no cross terms)
```

Standalone VaR for each PC is `z × |wₖ σₖ √T|`. The diversified VaR is:

```
VaR_diversified = z × √(Σₖ (wₖ σₖ √T)²) ≤ Σₖ VaR_k
```

The difference `Σ VaR_k − VaR_diversified` is the diversification benefit — it is non-zero here because PC2 (slope) and PC3 (curvature) produce opposite-signed P&L moves to PC1 (level) for this particular portfolio. Sheet 8 Section D shows this breakdown explicitly.
