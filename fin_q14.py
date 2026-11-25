"""재무 질문 1(영업이익률 vs 수익률)과 4(부채비율 vs 변동성).
먼저: python fin_parse.py 2026-09-02   -> fin_2026-09-02.parquet
실행: PARQUET_DIR=~/parquet python3 fin_q14.py 2>&1 | tee fin_q14_out.txt
정의: 12월 결산 법인(최근 연간 = 2025.12)의 2025년 연간 영업이익률·부채비율을, 그 값이 공개된 뒤인 WIN_START 부터 마지막날까지의 수익률·변동성과 비교한다."""
import os
from common import *
WIN_START = os.environ.get("WIN_START", "2026-04-01")
FIN = os.environ.get("FIN_SNAP", "fin_2026-09-02.parquet")
FINANCE = {"은행", "증권", "손해보험", "생명보험"}

df = load_panel()
fin = pd.read_parquet(FIN)
fin = fin[fin["has_table"] & (fin["last_y"] == "2025.12")][["code", "y_opm", "y_debt"]]
print(f"\n12월 결산(2025.12 연간 있음) 종목: {len(fin)}")

dates = sorted(df["date"].unique())
t0 = next(d for d in dates if d >= pd.Timestamp(WIN_START)); t1 = dates[-1]
w = df[(df["date"] >= t0)].copy()
w["r"] = w.groupby("code")["close"].pct_change()
w = w[w["r"].abs() <= MAXRET]
vol = w.groupby("code")["r"].std() * np.sqrt(252)
c0 = df[df["date"] == t0].set_index("code")["close"]; c1 = df[df["date"] == t1].set_index("code")["close"]
sec = df[df["date"] == t1].set_index("code")["sector"]
d = fin.set_index("code").join(pd.DataFrame({"ret": c1 / c0 - 1, "vol": vol, "sector": sec})).dropna(subset=["ret", "vol", "sector"])
print(f"기간 {t0.date()} ~ {t1.date()}, 분석 종목 {len(d)}")

def q5(x, y, label):
    g = pd.qcut(x, 5, labels=False, duplicates="drop")
    t = y.groupby(g).agg(["median", "mean", "count"])
    print(f"\n{label}\n", t.round(4).to_string())
    return t["median"], x.corr(y, method="spearman")

# ---- 재무 1: 영업이익률 vs 수익률 ----
a = d.dropna(subset=["y_opm"])
print("\n[재무1] 정의: 2025년 연간 영업이익률(%) 5분위별, 이후 수익률(종가 기준)")
m1, s1 = q5(a["y_opm"], a["ret"], "기본: 전체 종목 5분위 (수익률 중앙값·평균)"); print("스피어만 상관", round(s1, 3))
a2 = a.assign(rel=a["y_opm"] - a.groupby("sector")["y_opm"].transform("median"))
a2 = a2[a2.groupby("sector")["y_opm"].transform("count") >= 5]
m1b, s1b = q5(a2["rel"], a2["ret"], "정의 변경: 같은 업종 중앙값과의 차이로 5분위"); print("스피어만 상관", round(s1b, 3))
a3 = a[a["y_opm"].between(-50, 50)]
m1c, s1c = q5(a3["y_opm"], a3["ret"], "정의 변경: 영업이익률 -50%~50% 안만"); print("스피어만 상관", round(s1c, 3))

# ---- 재무 4: 부채비율 vs 변동성 ----
b = d.dropna(subset=["y_debt"])
b = b[b["y_debt"] >= 0]
print("\n[재무4] 정의: 2025년 연간 부채비율(%) 5분위별, 이후 일 수익률 표준편차(연 환산)")
m4, s4 = q5(b["y_debt"], b["vol"], "기본: 전체 종목"); print("스피어만 상관", round(s4, 3))
b2 = b[~b["sector"].isin(FINANCE)]
m4b, s4b = q5(b2["y_debt"], b2["vol"], "정의 변경: 은행·증권·보험 제외"); print("스피어만 상관", round(s4b, 3))
b3 = b2.assign(rel=b2["y_debt"] - b2.groupby("sector")["y_debt"].transform("median"))
b3 = b3[b3.groupby("sector")["y_debt"].transform("count") >= 5]
m4c, s4c = q5(b3["rel"], b3["vol"], "정의 변경: 같은 업종 중앙값과의 차이로 5분위"); print("스피어만 상관", round(s4c, 3))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a_, m, s, t in zip(ax, (m1, m1b, m1c), (s1, s1b, s1c), (T("전체 5분위", "all"), T("업종 내 상대", "within sector"), T("영업이익률 ±50% 안", "|opm|<=50"))):
    m.plot(kind="bar", ax=a_); a_.axhline(0, color="k", lw=.8); a_.set_title(f"{t} (rho={s:.2f})"); a_.set_xlabel(T("영업이익률 분위 (0=낮음)", "OPM quintile"))
ax[0].set_ylabel(T("이후 수익률 중앙값", "median return"))
plt.suptitle(T("재무1 영업이익률과 이후 수익률", "Q1 OPM vs return")); plt.tight_layout(); plt.savefig("fig_fin1.png", dpi=150); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a_, m, s, t in zip(ax, (m4, m4b, m4c), (s4, s4b, s4c), (T("전체", "all"), T("금융업 제외", "ex-finance"), T("업종 내 상대", "within sector"))):
    m.plot(kind="bar", ax=a_); a_.set_title(f"{t} (rho={s:.2f})"); a_.set_xlabel(T("부채비율 분위 (0=낮음)", "debt quintile"))
ax[0].set_ylabel(T("변동성(연 환산) 중앙값", "median vol"))
plt.suptitle(T("재무4 부채비율과 변동성", "Q4 debt ratio vs volatility")); plt.tight_layout(); plt.savefig("fig_fin4.png", dpi=150)
print("done")
