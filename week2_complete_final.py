"""
2주차(공통 질문 1~4) 완결형 분석 및 시각화 스크립트 (week2_complete_final.py)
- 결함 14번(prev_close vs 전일 close 차이) 진단 및 pct_change vs prev_close 비교 출력
- 공통 질문 1~4 분석 및 시각화 저장 (w2_q1.png ~ w2_q4.png)
- 정의 변경 시 결론 변화 비교 수치(시총가중 vs 동일가중, 상·하위 극단 업종 제외, Q4 시장 초과수익률) 출력
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd
from pathlib import Path

PARQUET_DIR = Path("parquet")
START_DATE = "2024-04-15"
BAD_DATES = [
    "2024-09-04", "2024-09-11", "2024-09-12", "2024-10-10", "2024-10-15",
    "2024-10-18", "2024-11-21", "2024-11-22", "2025-04-25", "2025-08-08",
]
MAX_ABS_RET = 0.30
MIN_SECTOR_STOCKS = 10

names = {f.name for f in fm.fontManager.ttflist}
KO = next((n for n in ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "AppleGothic"] if n in names), None)
if KO:
    plt.rcParams["font.family"] = KO
plt.rcParams["axes.unicode_minus"] = False
T = (lambda ko, en: ko) if KO else (lambda ko, en: en)


def load_all_parquet() -> pd.DataFrame:
    files = sorted(PARQUET_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"'{PARQUET_DIR}' 폴더에 parquet 파일이 없습니다. 현재 경로: {Path.cwd()}")

    frames = []
    all_cols = set()
    for p in files:
        d = pd.read_parquet(p)
        all_cols.update(d.columns)
        if "error" in d.columns:
            d = d.copy()
            d["error"] = d["error"].astype("object")
            d = d[d["error"].isna()].drop(columns="error")
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    print("[컬럼 목록]", sorted(all_cols))
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.Timestamp(START_DATE)) & (~df["date"].isin(pd.to_datetime(BAD_DATES)))]
    df = df.dropna(subset=["sector", "close", "mktcap_100mn"])
    df = df[(df["close"] > 0) & (df["mktcap_100mn"] > 0)]
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    return df


def calc_sector_series(df: pd.DataFrame, ret_col: str = "ret_pct", weight_mode: str = "mktcap_lag1"):
    work = df.copy()
    if weight_mode == "mktcap_lag1":
        work["w"] = work.groupby("code")["mktcap_100mn"].shift(1)
    else:
        work["w"] = 1.0

    r = work.dropna(subset=[ret_col, "w"]).copy()
    before_n = len(r)
    r = r[r[ret_col].abs() <= MAX_ABS_RET]
    removed_n = before_n - len(r)

    sec_counts = df.groupby("sector")["code"].nunique()
    valid_sectors = sec_counts[sec_counts >= MIN_SECTOR_STOCKS].index
    rs = r[r["sector"].isin(valid_sectors)].copy()
    rs["rw"] = rs[ret_col] * rs["w"]

    mkt_ret = r.groupby("date").apply(lambda g: (g[ret_col] * g["w"]).sum() / g["w"].sum(), include_groups=False)
    sec_ret = rs.groupby(["date", "sector"]).apply(lambda g: g["rw"].sum() / g["w"].sum(), include_groups=False).unstack()
    sec_cum = (1.0 + sec_ret.fillna(0.0)).cumprod() - 1.0
    return mkt_ret, sec_ret, sec_cum, removed_n


def main():
    df = load_all_parquet()
    dates = sorted(df["date"].unique())
    d0, d1 = dates[0], dates[-1]
    print(f"기간: {d0.date()} ~ {d1.date()} | 거래일: {len(dates)} | 종목: {df['code'].nunique()} | 행: {len(df):,}")

    df["ret_pct"] = df.groupby("code")["close"].pct_change()
    lag_close = df.groupby("code")["close"].shift(1)
    if "prev_close" in df.columns:
        diff_mask = df["prev_close"].notna() & lag_close.notna() & (df["prev_close"] != lag_close)
        print(f"[결함 14 진단] prev_close != lag(close): {int(diff_mask.sum()):,} / {len(df):,} ({diff_mask.mean():.2%})")
        df["ret_prev"] = np.where(df["prev_close"] > 0, df["close"] / df["prev_close"] - 1.0, np.nan)
    else:
        print("[결함 14 진단] prev_close 컬럼이 없습니다.")
        df["ret_prev"] = df["ret_pct"]

    # ---------- Q1: 시가총액 순위 ----------
    top0 = df[df["date"] == d0].nlargest(30, "mktcap_100mn")
    top1 = df[df["date"] == d1].nlargest(30, "mktcap_100mn")
    kept = len(set(top0["code"]) & set(top1["code"]))
    new_top = top1[~top1["code"].isin(top0["code"])]

    sec_cap = df[df["date"].isin([d0, d1])].groupby(["date", "sector"])["mktcap_100mn"].sum().unstack()
    sec_rank = sec_cap.rank(ascending=False)
    move = (sec_rank[d0] - sec_rank[d1]).dropna()
    mv_plot = pd.concat([move.nlargest(8), move.nsmallest(8)])
    mv_plot = mv_plot[~mv_plot.index.duplicated()].sort_values()

    print(f"\n[Q1] 상위30 유지: {kept}/30 | 신규진입 종목수: {len(new_top)}")
    print("[Q1] 업종 시총순위 이동(상위/하위):\n", mv_plot.to_string())

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    mv_plot.plot(kind="barh", ax=ax[0], color=["tab:blue" if v < 0 else "tab:red" for v in mv_plot])
    ax[0].set_title(T("업종 시가총액 순위 변화 (+ 상승)", "Sector Rank Change (+ Up)"))
    new_top["sector"].value_counts().plot(kind="barh", ax=ax[1])
    ax[1].set_title(T(f"신규 Top30 종목 업종 (유지 {kept}/30)", f"New Top30 by Sector (Kept {kept}/30)"))
    plt.tight_layout()
    plt.savefig("w2_q1.png", dpi=150)
    plt.close()

    # ---------- Q2 & 정의 변경 비교 ----------
    mkt_cap_w, sec_ret, sec_cum, removed_n = calc_sector_series(df, "ret_pct", "mktcap_lag1")
    _, _, sec_cum_eq, _ = calc_sector_series(df, "ret_pct", "equal")
    _, _, sec_cum_prev, _ = calc_sector_series(df, "ret_prev", "mktcap_lag1")

    disp = sec_cum.std(axis=1)
    fin = sec_cum.iloc[-1].sort_values()
    core = fin.drop(list(fin.tail(3).index) + list(fin.head(3).index))
    print(f"\n[Q2] 이상치(|ret|>{MAX_ABS_RET:.0%}) 제외 행: {removed_n:,} | 유효 업종 수: {sec_ret.shape[1]}")
    print("[Q2] 하위 5개 누적수익률:\n", fin.head(5).round(3).to_string())
    print("[Q2] 상위 5개 누적수익률:\n", fin.tail(5).round(3).to_string())
    print(f"[Q2] 누적 표준편차 최고: {disp.idxmax().date()} ({disp.max():.3f}) / 말기: {disp.iloc[-1]:.3f}")
    print(f"[정의변경 비교] 말기 업종수익률 std 전체={fin.std():.3f} vs 상·하위3제외={core.std():.3f}")
    print(f"[정의변경 비교] 동일가중 말기 누적수익률 std={sec_cum_eq.iloc[-1].std():.3f} | prev_close기반 말기 누적수익률 std={sec_cum_prev.iloc[-1].std():.3f}")

    hi_lo = list(fin.tail(3).index) + list(fin.head(3).index)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for col in sec_cum.columns:
        if col in hi_lo:
            ax[0].plot(sec_cum.index, sec_cum[col], lw=2, label=col)
        else:
            ax[0].plot(sec_cum.index, sec_cum[col], color="lightgray", lw=0.7)
    ax[0].legend(fontsize=8)
    ax[0].set_title(T("업종별 누적 수익률 (상·하위 3개 강조)", "Cumulative Return by Sector"))
    ax[1].plot(disp.index, disp.values)
    ax[1].set_title(T("업종 간 누적 수익률 표준편차", "Cross-Sector Std of Cum. Return"))
    plt.tight_layout()
    plt.savefig("w2_q2.png", dpi=150)
    plt.close()

    # ---------- Q3: 업종 동조화 ----------
    def mean_upper_corr(frame: pd.DataFrame) -> float:
        m = frame.corr().values
        return float(np.nanmean(m[np.triu_indices_from(m, 1)]))

    roll60 = pd.Series({sec_ret.index[i]: mean_upper_corr(sec_ret.iloc[i - 59:i + 1]) for i in range(59, len(sec_ret))})
    crash = (mkt_cap_w <= mkt_cap_w.quantile(0.05)).reindex(sec_ret.index).fillna(False)
    c_all = mean_upper_corr(sec_ret)
    c_crash = mean_upper_corr(sec_ret[crash])
    c_norm = mean_upper_corr(sec_ret[~crash])
    print(f"\n[Q3] 전체 평균상관={c_all:.3f} | 급락일(하위5%)={c_crash:.3f} | 보통일={c_norm:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    im = ax[0].imshow(sec_ret.corr().values, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax[0])
    ax[0].set_title(T("업종 일별 수익률 상관행렬", "Sector Daily Return Correlation"))
    step = max(1, sec_ret.shape[1] // 15)
    ticks = np.arange(0, sec_ret.shape[1], step)
    ax[0].set_xticks(ticks)
    ax[0].set_xticklabels(sec_ret.columns[ticks], rotation=60, ha="right", fontsize=7)
    ax[0].set_yticks(ticks)
    ax[0].set_yticklabels(sec_ret.columns[ticks], fontsize=7)

    ax[1].plot(roll60.index, roll60.values, label=T("60일 롤링 평균상관", "60d Rolling Avg Corr"))
    ax[1].axhline(c_crash, color="red", ls="--", label=T("급락일 평균", "Crash Days"))
    ax[1].axhline(c_norm, color="gray", ls="--", label=T("보통일 평균", "Normal Days"))
    ax[1].legend()
    ax[1].set_title(T("업종 간 평균 상관 추이", "Avg Correlation Over Time"))
    plt.tight_layout()
    plt.savefig("w2_q3.png", dpi=150)
    plt.close()

    # ---------- Q4: 거래량과 가격 ----------
    g = df.groupby("code")
    vbase = g["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=15).mean())
    vr = df["volume"] / vbase

    mkt_cum_daily = (1.0 + mkt_cap_w.fillna(0.0)).cumprod()
    date_to_idx = {d: i for i, d in enumerate(mkt_cum_daily.index)}
    mkt_vals = mkt_cum_daily.values

    rows = []
    for mult in (3, 5, 10):
        for h in (5, 10, 20):
            fwd_ret = g["close"].shift(-h) / df["close"] - 1.0
            idxs = df["date"].map(date_to_idx).to_numpy()
            mkt_fwd = np.full(len(df), np.nan)
            valid_h = (idxs >= 0) & ((idxs + h) < len(mkt_vals))
            mkt_fwd[valid_h] = mkt_vals[idxs[valid_h] + h] / mkt_vals[idxs[valid_h]] - 1.0
            ex_ret = pd.Series(fwd_ret.to_numpy() - mkt_fwd, index=df.index)

            sub = df[(vr >= mult) & (vbase >= 1000)]
            sub_ex = ex_ret.loc[sub.index].dropna()
            rows.append({
                "mult": mult,
                "days": h,
                "n": int(len(sub_ex)),
                "excess_med_vs_mkt": float(sub_ex.median()),
            })

    q4_df = pd.DataFrame(rows)
    print("\n[Q4 - 시장 대비 h일 초과수익률 중앙값]\n", q4_df.round(4).to_string(index=False))
    piv = q4_df.pivot(index="days", columns="mult", values="excess_med_vs_mkt")
    piv.plot(kind="bar", figsize=(8, 5))
    plt.axhline(0, color="k", lw=0.8)
    plt.title(T("거래량 급증 뒤 시장 대비 초과 수익률 (중앙값)", "Post-Spike Excess Return vs Market (Median)"))
    plt.tight_layout()
    plt.savefig("w2_q4.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
