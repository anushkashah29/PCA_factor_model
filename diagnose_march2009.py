"""
Step-by-step breakdown of the 2009-03-18 "worst day" calculation.
Shows: raw yield levels -> daily changes -> eigenvector projection -> PC scores -> P&L
"""
import numpy as np
import pandas as pd
import requests

FRED_API_KEY = "0a7bcf57e2f38f50b9f230c76a4ff850"
SERIES = {
    "3M":"DGS3MO","6M":"DGS6MO","1Y":"DGS1","2Y":"DGS2","3Y":"DGS3",
    "5Y":"DGS5","7Y":"DGS7","10Y":"DGS10","20Y":"DGS20","30Y":"DGS30"
}
TENORS   = list(SERIES.keys())
TENOR_YR = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]

# ── Fetch full dataset (same as main script) ──────────────────────────────────
print("Fetching full dataset ...")
frames = {}
for label, sid in SERIES.items():
    r = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={"series_id": sid, "api_key": FRED_API_KEY,
                "file_type": "json", "observation_start": "2007-01-01",
                "observation_end": "2026-06-21"},
        timeout=30,
    )
    obs = r.json()["observations"]
    frames[label] = pd.Series(
        {o["date"]: float(o["value"]) if o["value"] != "." else float("nan")
         for o in obs}, dtype=float
    )
df = pd.DataFrame(frames)
df.index = pd.to_datetime(df.index)
df.columns = TENORS

# Clean (same rules as main script)
from pandas.tseries.holiday import USFederalHolidayCalendar
df = df[df.index.dayofweek < 5]
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=df.index.min(), end=df.index.max())
df = df[~df.index.isin(holidays)]
df = df.dropna(how="all").ffill().dropna().sort_index()

# ── Recompute eigenvectors (same as main script) ──────────────────────────────
changes = df.diff().dropna() * 100
Sigma   = np.cov(changes.values.T)
vals, vecs = np.linalg.eigh(Sigma)
idx  = np.argsort(vals)[::-1]
vals = vals[idx]; vecs = vecs[:, idx]

# Portfolio risk weights (same as main script)
PORTFOLIO    = {"2Y": 10_000_000, "10Y": 10_000_000}
COUPON_RATES = {"2Y": 0.045, "10Y": 0.042}

def bond_dv01(face, coupon, ytm, mat_yrs, freq=2):
    n = int(mat_yrs * freq)
    c = coupon / freq * face
    r = ytm / freq
    pv = lambda y: sum(c/(1+y)**t for t in range(1,n+1)) + face/(1+y)**n
    return pv(r) - pv(r + 0.0001/freq)

last = df.iloc[-1]
dv01_vec = np.zeros(len(TENORS))
for tenor, face in PORTFOLIO.items():
    ytm = last[tenor] / 100
    mat = TENOR_YR[TENORS.index(tenor)]
    dv01 = bond_dv01(face, COUPON_RATES[tenor], ytm, mat)
    dv01_vec[TENORS.index(tenor)] = dv01
weights = vecs[:, :3].T @ dv01_vec

# ── TARGET DATE ───────────────────────────────────────────────────────────────
TARGET = "2009-03-18"
idx_t  = df.index.get_loc(TARGET)
y1     = df.iloc[idx_t]           # yields on target date
y0     = df.iloc[idx_t - 1]       # yields on prior business day
prev_d = str(df.index[idx_t - 1].date())
dy     = (y1 - y0) * 100          # daily yield changes in bps

SEP = "=" * 72

# ── STEP 1: Yield Levels & Changes ───────────────────────────────────────────
print(f"\n{SEP}")
print(f"  STEP 1 — Actual Yield Levels & Daily Changes")
print(f"  Prior day: {prev_d}     Target: {TARGET}")
print(SEP)
print(f"  {'Tenor':>5}  {'Prior Yield (%)':>15}  {'Target Yield (%)':>16}  {'Change (bps)':>13}")
print(f"  {'-'*5}  {'-'*15}  {'-'*16}  {'-'*13}")
for t in TENORS:
    print(f"  {t:>5}  {y0[t]:>15.4f}  {y1[t]:>16.4f}  {dy[t]:>+13.4f}")

# ── STEP 2: Eigenvector Loadings ─────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 2 — Eigenvector Loadings for PC1, PC2, PC3")
print("  (From eigendecomposition of the 10x10 covariance matrix)")
print(SEP)
print(f"  {'Tenor':>5}  {'PC1 Loading':>12}  {'PC2 Loading':>12}  {'PC3 Loading':>12}")
print(f"  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*12}")
for i, t in enumerate(TENORS):
    print(f"  {t:>5}  {vecs[i,0]:>12.6f}  {vecs[i,1]:>12.6f}  {vecs[i,2]:>12.6f}")

# ── STEP 3: PC Score Computation (dot product) ───────────────────────────────
pc1 = float(dy.values @ vecs[:, 0])
pc2 = float(dy.values @ vecs[:, 1])
pc3 = float(dy.values @ vecs[:, 2])

print(f"\n{SEP}")
print("  STEP 3 — PC Score = sum(Yield_Change_i x Eigenvector_Loading_i)")
print("  Formula: PC_k = dy[3M]*e_k[3M] + dy[6M]*e_k[6M] + ... + dy[30Y]*e_k[30Y]")
print(SEP)

for k, (pc_val, label) in enumerate([(pc1,"PC1 Level"),(pc2,"PC2 Slope"),(pc3,"PC3 Curvature")]):
    terms = [(dy[t], vecs[i, k]) for i, t in enumerate(TENORS)]
    detail = " + ".join(f"({d:+.3f}x{e:+.6f})" for d, e in terms)
    total = sum(d*e for d, e in terms)
    print(f"\n  {label}:")
    print(f"    = {detail}")
    print(f"    = {total:+.4f} bps   [reported: {[pc1,pc2,pc3][k]:+.4f}]")

# ── STEP 4: Risk Weights ──────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 4 — Portfolio Risk Weights = DV01_vector dot Eigenvector")
print("  w_k = DV01_2Y * e_k[2Y]  +  DV01_10Y * e_k[10Y]  (others zero)")
print(SEP)
idx_2y  = TENORS.index("2Y")
idx_10y = TENORS.index("10Y")
print(f"  {'':>20}  {'DV01 ($/bp)':>12}  {'PC1 loading':>12}  {'PC2 loading':>12}  {'PC3 loading':>12}")
print(f"  {'2Y T-Note':>20}  {dv01_vec[idx_2y]:>12.2f}  {vecs[idx_2y,0]:>12.6f}  {vecs[idx_2y,1]:>12.6f}  {vecs[idx_2y,2]:>12.6f}")
print(f"  {'10Y T-Note':>20}  {dv01_vec[idx_10y]:>12.2f}  {vecs[idx_10y,0]:>12.6f}  {vecs[idx_10y,1]:>12.6f}  {vecs[idx_10y,2]:>12.6f}")
print()
for k, (w, label) in enumerate(zip(weights, ["PC1","PC2","PC3"])):
    calc = dv01_vec[idx_2y]*vecs[idx_2y,k] + dv01_vec[idx_10y]*vecs[idx_10y,k]
    print(f"  w_{label} = {dv01_vec[idx_2y]:.2f} x {vecs[idx_2y,k]:+.6f}  +  {dv01_vec[idx_10y]:.2f} x {vecs[idx_10y,k]:+.6f}  =  {calc:+,.2f}  $/bp")

# ── STEP 5: Daily P&L ─────────────────────────────────────────────────────────
daily_pnl_wrong = pc1*weights[0] + pc2*weights[1] + pc3*weights[2]
daily_pnl_correct = -daily_pnl_wrong   # correct sign: yield up = loss for long bonds

print(f"\n{SEP}")
print("  STEP 5 — Daily P&L  =  PC1*w1  +  PC2*w2  +  PC3*w3")
print(SEP)
print(f"  PC1 x w_PC1  =  {pc1:+.4f}  x  {weights[0]:+,.2f}  =  {pc1*weights[0]:+,.2f}")
print(f"  PC2 x w_PC2  =  {pc2:+.4f}  x  {weights[1]:+,.2f}  =  {pc2*weights[1]:+,.2f}")
print(f"  PC3 x w_PC3  =  {pc3:+.4f}  x  {weights[2]:+,.2f}  =  {pc3*weights[2]:+,.2f}")
print(f"  {'-'*50}")
print(f"  Daily P&L (as coded)     =  {daily_pnl_wrong:+,.2f}")
print(f"  10-Day P&L (x sqrt(10))  =  {daily_pnl_wrong * np.sqrt(10):+,.2f}")

print(f"\n{SEP}")
print("  SIGN CONVENTION CHECK")
print(SEP)
print(f"  On 2009-03-18: Fed announced QE1 ($300B Treasury purchases)")
print(f"  10Y yield change: {dy['10Y']:+.2f} bps   (yields FELL = bond prices ROSE)")
print(f"  Long bond portfolio should have had a LARGE GAIN, not a LARGE LOSS")
print()
print(f"  Current formula:  daily_pnl = +PC @ weights  = {daily_pnl_wrong:+,.0f}  (WRONG sign)")
print(f"  Correct formula:  daily_pnl = -PC @ weights  = {daily_pnl_correct:+,.0f}  (GAIN = positive)")
print()
print(f"  => March 18 2009 is the BEST day for long bonds, not the WORST")
print(f"  => The VaR model has the sign inverted: tail should come from yield RISES")
print()

# Show what the actual worst days SHOULD be (after sign fix)
pnl_correct = -(changes.values @ vecs[:, :3]) @ weights  # (T,)
sorted_idx  = np.argsort(pnl_correct)                     # worst first
print("  CORRECT worst 5 trading days (after sign fix):")
print(f"  {'Rank':>4}  {'Date':>12}  {'Daily P&L':>14}  {'10-Day Scaled P&L':>20}  {'10Y chg':>8}")
for rank, i in enumerate(sorted_idx[:5], 1):
    d = changes.index[i]
    p = pnl_correct[i]
    chg_10y = changes.iloc[i]["10Y"]
    print(f"  {rank:>4}  {str(d.date()):>12}  {p:>+14,.0f}  {p*np.sqrt(10):>+20,.0f}  {chg_10y:>+8.2f} bps")

print()
print("  WRONG worst 5 days (current sign — actually the biggest GAINS):")
pnl_wrong = (changes.values @ vecs[:, :3]) @ weights
sorted_wrong = np.argsort(pnl_wrong)
for rank, i in enumerate(sorted_wrong[:5], 1):
    d = changes.index[i]
    p = pnl_wrong[i]
    chg_10y = changes.iloc[i]["10Y"]
    print(f"  {rank:>4}  {str(d.date()):>12}  {p:>+14,.0f}  {p*np.sqrt(10):>+20,.0f}  {chg_10y:>+8.2f} bps")

print(f"\n{SEP}")
print("  VaR IMPACT of sign fix")
print(SEP)
confidence = 0.99
horizon    = 10
var_wrong   = abs(np.percentile(pnl_wrong  * np.sqrt(horizon), (1-confidence)*100))
var_correct = abs(np.percentile(pnl_correct * np.sqrt(horizon), (1-confidence)*100))
print(f"  Current (wrong sign) Historical VaR:  ${var_wrong:>12,.0f}")
print(f"  Correct (right sign) Historical VaR:  ${var_correct:>12,.0f}")
print(f"  Analytical VaR (sign-independent):    ${388969:>12,.0f}")
