"""재무 질문 2(실적 발표 뒤 반응)와 3(추정치 오차).
먼저: python fin_parse.py --all --workers 4   -> fin_parquet/날짜.parquet (전체 날짜)
실행: PARQUET_DIR=~/parquet python3 fin_q23.py 2>&1 | tee fin_q23_out.txt
정의
  실적 반영일 = 종목의 '최근 실제 분기'(last_q)가 처음 새 분기로 바뀐 날 (표가 갱신된 날)
  초과수익률 = 종목 일 수익률 - 그날 시장(시총가중) 수익률, 반영일부터 누적"""
import os
from pathlib import Path
from common import *

# ---------- 주가와 시장 수익률 (종가끼리 비교) ----------
df = load_panel()
df["ret"] = df.groupby("code")["close"].pct_change()
df["w"] = df.groupby("code")["mktcap_100mn"].shift(1)
r = df.dropna(subset=["ret", "w"]); r = r[r["ret"].abs() <= MAXRET]
mkt = (r["ret"] * r["w"]).groupby(r["date"]).sum() / r["w"].groupby(r["date"]).sum()
ar = r.pivot(index="date", columns="code", values="ret").sub(mkt, axis=0)
cs = ar.fillna(0).cumsum()
didx = {d: i for i, d in enumerate(cs.index)}; cidx = {c: i for i, c in enumerate(cs.columns)}
CS = cs.values

# ---------- 실적 표 읽기와 반영일 찾기 ----------
cols = ["date", "code", "has_table", "last_q", "q_rev", "q_op", "est_q", "estq_rev", "estq_op"]
parts = [pd.read_parquet(f, columns=cols) for f in sorted(Path("fin_parquet").glob("*.parquet"))]
fin = pd.concat(parts, ignore_index=True)
fin["date"] = pd.to_datetime(fin["date"]); fin = fin[fin["has_table"] & fin["last_q"].notna()]
fin = fin.sort_values(["code", "date"]).reset_index(drop=True)
print(f"실적표 읽은 행 {len(fin):,}, 종목 {fin['code'].nunique()}, 날짜 {fin['date'].nunique()}")
g = fin.groupby("code")
for c in ["last_q", "q_op", "q_rev", "est_q", "estq_op", "estq_rev", "date"]:
    fin["p_" + c] = g[c].shift(1)
ev = fin[fin["p_last_q"].notna() & (fin["last_q"] > fin["p_last_q"])].copy()
print(f"실적 반영 사건 {len(ev):,}건 (종목 {ev['code'].nunique()}개), 분기: {ev['last_q'].min()} ~ {ev['last_q'].max()}")
print("반영일이 전 수집일과 5일 넘게 떨어진 사건(수집 공백 의심):", int(((ev["date"] - ev["p_date"]).dt.days > 5).sum()))

# 같은 분기의 1년 전 값(전년 동기)
last_val = fin.groupby(["code", "last_q"])["q_op"].last()
def prior_year(q):
    y, m = q.split("."); return f"{int(y)-1}.{m}"
ev["yoy_base"] = [last_val.get((c, prior_year(q)), np.nan) for c, q in zip(ev["code"], ev["last_q"])]
est_ok = ev["p_est_q"].notna() & (ev["p_est_q"].str.replace("(E)", "", regex=False) == ev["last_q"])
ev["est_op"] = ev["p_estq_op"].where(est_ok); ev["est_rev"] = ev["p_estq_rev"].where(est_ok)

# ---------- 재무 2: 실적 발표 뒤 반응 ----------
ev["di"] = ev["date"].map(didx); ev["ci"] = ev["code"].map(cidx)
ev = ev.dropna(subset=["di", "ci"]); ev["di"] = ev["di"].astype(int); ev["ci"] = ev["ci"].astype(int)
K = list(range(-5, 21))
ev = ev[(ev["di"] >= 6) & (ev["di"] + 20 < len(CS))]
path = np.vstack([CS[ev["di"] + k, ev["ci"]] - CS[ev["di"] - 6, ev["ci"]] for k in K]).T   # 시점 -5부터 누적
ev["car0_5"] = CS[ev["di"] + 5, ev["ci"]] - CS[ev["di"] - 1, ev["ci"]]
ev["car0_20"] = CS[ev["di"] + 20, ev["ci"]] - CS[ev["di"] - 1, ev["ci"]]
ev["pre5"] = CS[ev["di"] - 1, ev["ci"]] - CS[ev["di"] - 6, ev["ci"]]
print(f"\n[재무2] 정의: 실적 반영일부터 5·20 거래일 누적 초과수익률(종목-시장). 사건 {len(ev):,}건")
print("전체: 반영 전 5일 평균 %.4f, 반영 후 5일 평균 %.4f, 20일 평균 %.4f (중앙값 %.4f)" % (ev["pre5"].mean(), ev["car0_5"].mean(), ev["car0_20"].mean(), ev["car0_20"].median()))
defs = {
    "직전 분기보다 영업이익 증가": (ev["q_op"] > ev["p_q_op"]).where(ev["q_op"].notna() & ev["p_q_op"].notna()),
    "전년 같은 분기보다 영업이익 증가": (ev["q_op"] > ev["yoy_base"]).where(ev["q_op"].notna() & ev["yoy_base"].notna()),
    "추정치보다 영업이익 높음": (ev["q_op"] > ev["est_op"]).where(ev["q_op"].notna() & ev["est_op"].notna()),
}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for a_, (name, flag) in zip(ax, defs.items()):
    m = flag.notna(); up, dn = (m & (flag == True)).to_numpy(), (m & (flag == False)).to_numpy()
    print(f"\n정의: {name} (분류된 사건 {int(m.sum()):,}건)")
    for lab, sel in (("좋음", up), ("나쁨", dn)):
        print(f"  {lab}: {int(sel.sum()):6d}건  20일 평균 {ev.loc[sel, 'car0_20'].mean():+.4f}  중앙값 {ev.loc[sel, 'car0_20'].median():+.4f}  5일 평균 {ev.loc[sel, 'car0_5'].mean():+.4f}")
        a_.plot(K, np.nanmean(path[sel], axis=0), label=f"{lab} (n={int(sel.sum())})")
    a_.axvline(0, color="k", lw=.8, ls="--"); a_.axhline(0, color="gray", lw=.5)
    a_.set_title(T(name, name)); a_.set_xlabel(T("반영일 기준 거래일", "trading days")); a_.legend()
ax[0].set_ylabel(T("누적 초과수익률(-5일부터)", "cum. abnormal return"))
plt.tight_layout(); plt.savefig("fig_fin2.png", dpi=150); plt.close()

# ---------- 재무 3: 추정치 오차 ----------
e = ev.dropna(subset=["est_op", "q_op"]).copy()
e = e[e["est_op"].abs() > 0]
e["err_op"] = (e["q_op"] - e["est_op"]) / e["est_op"].abs()
print(f"\n[재무3] 정의: 영업이익 추정치 오차 = (발표된 실제 영업이익 - 직전 추정치) / |추정치|. 추정치가 있는 사건 {len(e):,}건")
print("오차 중앙값 %.3f, 평균 %.3f, 1~99%% 절사 평균 %.3f, 실제가 추정보다 높은 비율 %.3f" % (
    e["err_op"].median(), e["err_op"].mean(), e["err_op"].clip(e["err_op"].quantile(.01), e["err_op"].quantile(.99)).mean(), (e["q_op"] > e["est_op"]).mean()))
print("절대 오차 중앙값 %.3f / 오차가 ±20%% 안인 비율 %.3f / ±50%% 안 %.3f" % (e["err_op"].abs().median(), (e["err_op"].abs() <= .2).mean(), (e["err_op"].abs() <= .5).mean()))
e["err_rev"] = (e["q_rev"] - e["est_rev"]) / e["est_rev"].abs()
er = e.dropna(subset=["err_rev"]); er = er[np.isfinite(er["err_rev"])]
print("정의 변경 1: 영업이익 대신 매출액 오차 중앙값 %.3f, 절대 오차 중앙값 %.3f (n=%d)" % (er["err_rev"].median(), er["err_rev"].abs().median(), len(er)))
e2 = e.dropna(subset=["est_rev"]); e2 = e2[e2["est_rev"] > 0]
e2["err_pt"] = (e2["q_op"] - e2["est_op"]) / e2["est_rev"]
print("정의 변경 2: 오차를 추정 매출액으로 나누면(영업이익률 %%p 단위) 중앙값 %.4f, 절대 오차 중앙값 %.4f" % (e2["err_pt"].median(), e2["err_pt"].abs().median()))
big = e[e["est_op"].abs() < e["est_op"].abs().quantile(.1)]
print("정의 변경 3: 추정 영업이익이 작은 하위 10%%만 보면 절대 오차 중앙값 %.3f (전체 %.3f)" % (big["err_op"].abs().median(), e["err_op"].abs().median()))
byq = e.groupby("last_q")["err_op"].agg(["count", "median"])
print("분기별 오차 중앙값:\n", byq.round(3).to_string())
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].hist(e["err_op"].clip(-2, 2), bins=60); ax[0].axvline(0, color="k", lw=.8)
ax[0].set_title(T("영업이익 추정치 오차 분포 (±200%로 자름)", "op. profit estimate error")); ax[0].set_xlabel(T("(실제-추정)/|추정|", "(act-est)/|est|"))
byq[byq["count"] >= 30]["median"].plot(kind="bar", ax=ax[1]); ax[1].axhline(0, color="k", lw=.8)
ax[1].set_title(T("분기별 오차 중앙값", "median error by quarter"))
plt.tight_layout(); plt.savefig("fig_fin3.png", dpi=150)
print("done")
