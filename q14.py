import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np, pandas as pd
from pathlib import Path

names = {f.name for f in fm.fontManager.ttflist}
for n in ["NanumGothic","Noto Sans CJK KR","Noto Sans KR","Malgun Gothic","AppleGothic","UnDotum"]:
    if n in names:
        plt.rcParams["font.family"] = n; print("font:", n); break
else:
    print("한글 폰트 없음: 글자가 네모로 보일 수 있음")
plt.rcParams["axes.unicode_minus"] = False

START = "2024-04-15"
BAD = ["2024-09-04","2024-09-11","2024-09-12","2024-10-10","2024-10-15",
       "2024-10-18","2024-11-21","2024-11-22","2025-04-25","2025-08-08"]

# ---------- 데이터 읽기 (처음 한 번만 오래 걸리고 all.parquet로 저장) ----------
cache = Path("all.parquet")
if cache.exists():
    df = pd.read_parquet(cache)
else:
    parts = []
    for f in sorted(Path("parquet").glob("*.parquet")):
        d = pd.read_parquet(f, columns=["date","code","name","sector","close","volume","mktcap_100mn","error"])
        d["error"] = d["error"].astype("object")
        d = d[d["error"].isna()].drop(columns="error")
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(cache)

df["date"] = pd.to_datetime(df["date"])
df = df[(df["date"] >= START) & (~df["date"].isin(pd.to_datetime(BAD)))]
df = df.dropna(subset=["close","mktcap_100mn"])
df = df[(df["close"] > 0) & (df["mktcap_100mn"] > 0)].sort_values(["code","date"]).reset_index(drop=True)
dates = sorted(df["date"].unique()); first, last = dates[0], dates[-1]
print("분석 기간:", pd.Timestamp(first).date(), "~", pd.Timestamp(last).date(),
      "/ 날짜", len(dates), "/ 종목", df["code"].nunique(), "/ 행", len(df))

# ---------- 질문 1: 시가총액 순위 ----------
t0 = df[df["date"] == first].nlargest(30, "mktcap_100mn")
t1 = df[df["date"] == last].nlargest(30, "mktcap_100mn")
stay = set(t0["code"]) & set(t1["code"])
new = t1[~t1["code"].isin(t0["code"])]
print("\n[Q1] 시작 상위30 중 마지막날에도 상위30:", len(stay), "개")
print("새로 들어온 종목:\n", new[["name","sector"]].to_string())
sec = df[df["date"].isin([first, last])].groupby(["date","sector"])["mktcap_100mn"].sum().unstack(0).dropna()
sec["r0"] = sec[first].rank(ascending=False); sec["r1"] = sec[last].rank(ascending=False)
sec["move"] = sec["r0"] - sec["r1"]
mv = pd.concat([sec["move"].nlargest(8), sec["move"].nsmallest(8)]).drop_duplicates().sort_values()
print("업종 순위 이동 상/하위:\n", mv.to_string())
fig, ax = plt.subplots(1, 2, figsize=(13, 6))
mv.plot(kind="barh", ax=ax[0], color=["tab:blue" if v < 0 else "tab:red" for v in mv])
ax[0].set_title("업종 시총 순위 이동 (양수=상승)"); ax[0].set_ylabel("")
new["sector"].value_counts().plot(kind="barh", ax=ax[1])
ax[1].set_title(f"새로 상위 30에 든 종목의 업종 (유지 {len(stay)}개 / 30)"); ax[1].set_ylabel("")
plt.tight_layout(); plt.savefig("fig_q1.png", dpi=150); plt.close()

# ---------- 수익률 만들기 ----------
df["ret"] = df.groupby("code")["close"].pct_change()
df["w"] = df.groupby("code")["mktcap_100mn"].shift(1)
r = df.dropna(subset=["ret","w"])
n0 = len(r); r = r[r["ret"].abs() <= 0.30]
print("\n하루 ±30% 초과로 제외한 행:", n0 - len(r), f"({(n0-len(r))/n0:.3%})")
r = r.assign(rw=r["ret"] * r["w"])
mk = r.groupby("date")["rw"].sum() / r.groupby("date")["w"].sum()
cnt = df.groupby("sector")["code"].nunique()
rs = r[r["sector"].isin(cnt[cnt >= 10].index)]
g = rs.groupby(["date","sector"])
sr = (g["rw"].sum() / g["w"].sum()).unstack()
print("업종 수:", sr.shape[1])

# ---------- 질문 2: 업종 수익률 차이 ----------
cum = (1 + sr.fillna(0)).cumprod() - 1
fin = cum.iloc[-1].sort_values()
print("\n[Q2] 업종 누적수익률 하위5:\n", fin.head(5).round(3).to_string())
print("상위5:\n", fin.tail(5).round(3).to_string())
disp = cum.std(axis=1)
core = fin.drop(list(fin.tail(3).index) + list(fin.head(3).index))
print(f"최종 업종간 표준편차 {fin.std():.3f} / 상하위 3개씩 뺀 뒤 {core.std():.3f}")
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
hi = list(fin.tail(3).index); lo = list(fin.head(3).index)
for c in cum.columns:
    if c in hi + lo: ax[0].plot(cum.index, cum[c], lw=2, label=c)
    else: ax[0].plot(cum.index, cum[c], color="lightgray", lw=0.7)
ax[0].legend(fontsize=8); ax[0].set_title("업종별 누적 수익률 (상·하위 3개 강조)")
ax[1].plot(disp.index, disp.values); ax[1].set_title("업종 간 누적수익률 표준편차의 변화")
plt.tight_layout(); plt.savefig("fig_q2.png", dpi=150); plt.close()

# ---------- 질문 3: 동조화 ----------
def avgcorr(x):
    c = x.corr().values; iu = np.triu_indices_from(c, 1); return np.nanmean(c[iu])
roll = pd.Series({sr.index[i]: avgcorr(sr.iloc[i-59:i+1]) for i in range(59, len(sr))})
crash = (mk <= mk.quantile(0.05)).reindex(sr.index).fillna(False)
c_crash = avgcorr(sr[crash]); c_norm = avgcorr(sr[~crash])
print(f"\n[Q3] 업종 간 평균 상관: 전체 {avgcorr(sr):.3f} / 급락 하위5% 날 {c_crash:.3f} / 보통 날 {c_norm:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
im = ax[0].imshow(sr.corr().values, vmin=-1, vmax=1, cmap="RdBu_r"); plt.colorbar(im, ax=ax[0])
ax[0].set_title("업종 일별 수익률 상관행렬")
ax[1].plot(roll.index, roll.values, label="60일 평균 상관")
ax[1].axhline(c_crash, color="red", ls="--", label=f"급락일 {c_crash:.2f}")
ax[1].axhline(c_norm, color="gray", ls="--", label=f"보통날 {c_norm:.2f}")
ax[1].legend(); ax[1].set_title("업종 간 평균 상관의 시간 변화")
plt.tight_layout(); plt.savefig("fig_q3.png", dpi=150); plt.close()

# ---------- 질문 4: 거래량과 가격 ----------
gg = df.groupby("code")
df["vbase"] = gg["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=15).mean())
df["vr"] = df["volume"] / df["vbase"]
H = (5, 10, 20)
for h in H: df[f"f{h}"] = gg["close"].shift(-h) / df["close"] - 1
base = {h: df[f"f{h}"].median() for h in H}
rows = []
for k in (3, 5, 10):
    s = df[(df["vr"] >= k) & (df["vbase"] >= 1000)]
    for h in H:
        rows.append({"배수": k, "일": h, "건수": len(s), "중앙값초과수익": s[f"f{h}"].median() - base[h]})
res = pd.DataFrame(rows)
print("\n[Q4] 급증 뒤 수익률(전체 날 중앙값 대비 초과):\n", res.round(4).to_string(index=False))
p = res.pivot(index="일", columns="배수", values="중앙값초과수익")
ax = p.plot(kind="bar", figsize=(8, 5)); ax.axhline(0, color="k", lw=0.8)
ax.set_title("거래량 급증 뒤 N일 수익률 (전체 대비, 급증 배수별)"); ax.set_ylabel("중앙값 초과수익")
plt.tight_layout(); plt.savefig("fig_q4.png", dpi=150)
print("\n완료")
