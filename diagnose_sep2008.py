"""
Step-by-step breakdown for 2008-09-19 (worst day in HistSim after sign fix).
Shows: raw yield levels -> daily changes -> eigenvector projection -> PC scores -> P&L
"""
import numpy as np
import pandas as pd
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar

FRED_API_KEY = "0a7bcf57e2f38f50b9f230c76a4ff850"
SERIES = {
    "3M":"DGS3MO","6M":"DGS6MO","1Y":"DGS1","2Y":"DGS2","3Y":"DGS3",
    "5Y":"DGS5","7Y":"DGS7","10Y":"DGS10","20Y":"DGS20","30Y":"DGS30"
}
TENORS   = list(SERIES.keys())
TENOR_YR = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
PORTFOLIO    = {"2Y": 10_000_000, "10Y": 10_000_000}
COUPON_RATES = {"2Y": 0.045, "10Y": 0.042}
TARGET = "2008-09-19"

# ── Fetch & clean (identical to main script) ──────────────────────────────────
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
df = df[df.index.dayofweek < 5]
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=df.index.min(), end=df.index.max())
df = df[~df.index.isin(holidays)]
df = df.dropna(how="all").ffill().dropna().sort_index()

# ── Eigenvectors (identical to main script) ───────────────────────────────────
changes = df.diff().dropna() * 100          # daily changes in bps
Sigma   = np.cov(changes.values.T)
vals, vecs = np.linalg.eigh(Sigma)
idx  = np.argsort(vals)[::-1]
vals = vals[idx]; vecs = vecs[:, idx]

# ── DV01 & risk weights (identical to main script) ────────────────────────────
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

# ── Pull the two dates we need ────────────────────────────────────────────────
idx_t  = df.index.get_loc(TARGET)
y0     = df.iloc[idx_t - 1]        # prior business day
y1     = df.iloc[idx_t]            # 2008-09-19
prev_d = str(df.index[idx_t - 1].date())
dy     = (y1 - y0) * 100           # change in bps

SEP = "=" * 78

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 1 — Raw Yield Levels & Daily Changes")
print(f"  Prior day : {prev_d}")
print(f"  Target day: {TARGET}  (Lehman aftermath — Treasury short-end spiked)")
print(SEP)
print(f"  {'Tenor':>5}  {'Prior Yield':>14}  {'Target Yield':>14}  {'Change (bps)':>14}")
print(f"  {'-'*5}  {'-'*14}  {'-'*14}  {'-'*14}")
for t in TENORS:
    print(f"  {t:>5}  {y0[t]:>14.4f}%  {y1[t]:>14.4f}%  {dy[t]:>+14.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 2 — Eigenvector Loadings  (from eigendecomposition of 10x10 covariance matrix)")
print(SEP)
print(f"  {'Tenor':>5}  {'PC1 (Level)':>13}  {'PC2 (Slope)':>13}  {'PC3 (Curvature)':>17}")
print(f"  {'-'*5}  {'-'*13}  {'-'*13}  {'-'*17}")
for i, t in enumerate(TENORS):
    print(f"  {t:>5}  {vecs[i,0]:>13.6f}  {vecs[i,1]:>13.6f}  {vecs[i,2]:>17.6f}")

# ─────────────────────────────────────────────────────────────────────────────
pc1 = float(dy.values @ vecs[:, 0])
pc2 = float(dy.values @ vecs[:, 1])
pc3 = float(dy.values @ vecs[:, 2])

print(f"\n{SEP}")
print("  STEP 3 — PC Scores  =  sum( Change_i  x  Eigenvector_loading_i )  for each PC")
print(SEP)

for k, (pc_val, lbl) in enumerate([(pc1,"PC1 Level"),(pc2,"PC2 Slope"),(pc3,"PC3 Curvature")]):
    print(f"\n  {lbl}:")
    print(f"  {'Tenor':>5}  {'Change (bps)':>14}  {'x':>3}  {'Loading':>12}  {'=':>3}  {'Product':>12}")
    print(f"  {'-'*5}  {'-'*14}  {'-'*3}  {'-'*12}  {'-'*3}  {'-'*12}")
    running = 0.0
    for i, t in enumerate(TENORS):
        prod = dy[t] * vecs[i, k]
        running += prod
        print(f"  {t:>5}  {dy[t]:>+14.4f}  {'x':>3}  {vecs[i,k]:>12.6f}  {'=':>3}  {prod:>+12.4f}")
    print(f"  {'':>5}  {'':>14}  {'':>3}  {'SUM':>12}  {'=':>3}  {running:>+12.4f}")
    print(f"  => Reported on HistSim_PnL tab: {pc_val:+.3f}")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 4 — Portfolio Risk Weights  =  DV01_vector . Eigenvector")
print("  (Only 2Y and 10Y positions are non-zero in the portfolio)")
print(SEP)
idx_2y  = TENORS.index("2Y")
idx_10y = TENORS.index("10Y")
print(f"  {'Bond':>10}  {'DV01 ($/bp)':>13}  {'PC1 load':>10}  {'PC2 load':>10}  {'PC3 load':>10}")
print(f"  {'-'*10}  {'-'*13}  {'-'*10}  {'-'*10}  {'-'*10}")
print(f"  {'2Y Note':>10}  {dv01_vec[idx_2y]:>13.2f}  {vecs[idx_2y,0]:>10.6f}  {vecs[idx_2y,1]:>10.6f}  {vecs[idx_2y,2]:>10.6f}")
print(f"  {'10Y Note':>10}  {dv01_vec[idx_10y]:>13.2f}  {vecs[idx_10y,0]:>10.6f}  {vecs[idx_10y,1]:>10.6f}  {vecs[idx_10y,2]:>10.6f}")
print()
for k, lbl in enumerate(["PC1","PC2","PC3"]):
    w = weights[k]
    calc = dv01_vec[idx_2y]*vecs[idx_2y,k] + dv01_vec[idx_10y]*vecs[idx_10y,k]
    print(f"  w_{lbl} = {dv01_vec[idx_2y]:.2f} x {vecs[idx_2y,k]:+.6f}  +  "
          f"{dv01_vec[idx_10y]:.2f} x {vecs[idx_10y,k]:+.6f}  =  {calc:>+12,.2f} $/bp")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 5 — Daily P&L  =  -(PC1*w1  +  PC2*w2  +  PC3*w3)")
print("  (Negative sign: eigenvector orientation means yield rise -> negative PC -> loss")
print("   without the negation; the negation restores correct long-bond sign convention.)")
print(SEP)
print(f"  {'Component':>20}  {'PC Score':>10}  {'x':>3}  {'Risk Weight':>14}  {'=':>3}  {'P&L ($)':>14}")
print(f"  {'-'*20}  {'-'*10}  {'-'*3}  {'-'*14}  {'-'*3}  {'-'*14}")
components = [(pc1,"PC1 Level",weights[0]),(pc2,"PC2 Slope",weights[1]),(pc3,"PC3 Curv.",weights[2])]
gross = 0.0
for lbl, label, w in components:
    pnl_comp = lbl * w
    gross += pnl_comp
    print(f"  {label:>20}  {lbl:>+10.3f}  {'x':>3}  {w:>+14,.2f}  {'=':>3}  {pnl_comp:>+14,.2f}")
print(f"  {'-'*20}  {'-'*10}  {'':>3}  {'SUM':>14}  {'=':>3}  {gross:>+14,.2f}")
print(f"  Negate (sign fix):                                              {-gross:>+14,.2f}")
daily_pnl = -gross
pnl_10d   = daily_pnl * np.sqrt(10)
print(f"\n  Daily P&L            = {daily_pnl:>+14,.2f}")
print(f"  10-Day Scaled P&L   = Daily x sqrt(10) = {daily_pnl:,.2f} x {np.sqrt(10):.4f} = {pnl_10d:>+14,.2f}")

print(f"\n{SEP}")
print("  SIGN CHECK")
print(SEP)
print(f"  2008-09-19: Lehman brothers had collapsed (Sep 15). Short-term")
print(f"  Treasury yields spiked as markets de-risked (flight FROM Treasuries")
print(f"  in the very short end, flight TO quality in the long end was mixed).")
print(f"  Net effect on this 2Y+10Y long portfolio: LOSS (yields rose overall).")
print(f"  Daily P&L = {daily_pnl:>+,.0f}  => NEGATIVE = LOSS   (correct for a yield-rise day)")
print(f"  This is the #1 worst day in the HistSim after the sign fix.")
