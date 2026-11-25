"""
3주차(재무제표 질문 5개) 분석 및 시각화 스크립트 (week3_fin_complete.py)
- 질문 1: 영업이익률 대 수익률 (fin_q1.png)
- 질문 2: 실적 발표 뒤 반응 (fin_q2.png)
- 질문 3: 추정치 오차 (fin_q3.png)
- 질문 4: 부채비율 대 변동성 (fin_q4.png)
- 질문 5: PBR < 1 (fin_q5.png)
* 주의: parquet 및 상위 폴더의 csv/parquet/html 경로를 자동 탐색해 매핑합니다.
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

names = {f.name for f in fm.fontManager.ttflist}
KO = next((n for n in ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "AppleGothic"] if n in names), None)
if KO:
    plt.rcParams["font.family"] = KO
plt.rcParams["axes.unicode_minus"] = False
T = (lambda ko, en: ko) if KO else (lambda ko, en: en)


def find_col(cols, candidates):
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for c in cols:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def load_data():
    files = sorted(PARQUET_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"'{PARQUET_DIR}' 폴더에 parquet 파일이 없습니다. 현재 경로: {Path.cwd()}")

    frames = []
    for p in files:
        d = pd.read_parquet(p)
        if "error" in d.columns:
            d = d.copy()
            d["error"] = d["error"].astype("object")
            d = d[d["error"].isna()].drop(columns="error")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.Timestamp(START_DATE)) & (~df["date"].isin(pd.to_datetime(BAD_DATES)))]
    df = df.dropna(subset=["sector", "close", "mktcap_100mn"])
    df = df[(df["close"] > 0) & (df["mktcap_100mn"] > 0)].sort_values(["code", "date"]).reset_index(drop=True)

    # 상위/하위 폴더에 별도 재무/실적 파싱 파일이 있는지 탐색
    extra_files = [p for p in Path(".").rglob("*") if p.suffix.lower() in (".parquet", ".csv") and "parquet" not in p.parts]
    print("[추가 데이터 후보 파일]", [str(p) for p in extra_files[:20]])
    print("[기본 df 컬럼]", sorted(df.columns))
    return df, extra_files


def analyze_financial_questions(df: pd.DataFrame):
    df["ret"] = df.groupby("code")["close"].pct_change()
    df.loc[df["ret"].abs() > MAX_ABS_RET, "ret"] = np.nan

    cols = list(df.columns)
    opm_col = find_col(cols, ["opm", "op_margin", "operating_margin", "영업이익률"])
    op_col = find_col(cols, ["op_profit", "operating_profit", "영업이익"])
    rev_col = find_col(cols, ["revenue", "sales", "매출액"])
    est_op_col = find_col(cols, ["est_op", "op_est", "consensus_op", "영업이익_추정", "추정영업이익"])
    debt_col = find_col(cols, ["debt_ratio", "부채비율"])
    liab_col = find_col(cols, ["total_liabilities", "liabilities", "부채총계"])
    eq_col = find_col(cols, ["total_equity", "equity", "자본총계"])
    pbr_col = find_col(cols, ["pbr", "pb"])
    bps_col = find_col(cols, ["bps"])
    ann_col = find_col(cols, ["announce_date", "earnings_date", "report_date", "실적발표일", "공시일"])

    print("[탐지된 재무 컬럼 매핑]", {
        "opm": opm_col, "op": op_col, "rev": rev_col, "est_op": est_op_col,
        "debt": debt_col, "liab": liab_col, "eq": eq_col, "pbr": pbr_col, "bps": bps_col, "ann": ann_col
    })

    # 파생 지표 생성 가능 시 계산
    if opm_col is None and op_col and rev_col:
        df["calc_opm"] = np.where(df[rev_col] > 0, df[op_col] / df[rev_col] * 100.0, np.nan)
        opm_col = "calc_opm"

    if debt_col is None and liab_col and eq_col:
        df["calc_debt_ratio"] = np.where(df[eq_col] > 0, df[liab_col] / df[eq_col] * 100.0, np.nan)
        debt_col = "calc_debt_ratio"

    if pbr_col is None and bps_col:
        df["calc_pbr"] = np.where(df[bps_col] > 0, df["close"] / df[bps_col], np.nan)
        pbr_col = "calc_pbr"

    # ---------- 재무 1: 영업이익률 대 수익률 ----------
    if opm_col:
        sub = df.dropna(subset=[opm_col, "ret"]).copy()
        sub = sub[sub[opm_col].between(-100, 100)]
        sub["opm_q"] = pd.qcut(sub[opm_col], q=5, labels=["Q1(저)", "Q2", "Q3", "Q4", "Q5(고)"], duplicates="drop")
        res1 = sub.groupby("opm_q", observed=False)["ret"].agg(["count", "mean", "median"])
        print("\n[재무 1: 영업이익률 분위별 일별 수익률]\n", res1.round(5).to_string())
        fig, ax = plt.subplots(figsize=(8, 5))
        (res1["mean"] * 252).plot(kind="bar", ax=ax, color="steelblue")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(T("영업이익률 분위별 연환산 평균 수익률", "Annualized Mean Return by OPM Quintile"))
        plt.tight_layout()
        plt.savefig("fin_q1.png", dpi=150)
        plt.close()
    else:
        print("\n[재무 1] 영업이익률/영업이익/매출액 컬럼이 parquet에 없어 별도 실적 파싱 병합이 필요합니다.")

    # ---------- 재무 2: 실적 발표 뒤 반응 ----------
    g = df.groupby("code")
    for h in (1, 5, 10, 20):
        df[f"fwd_{h}"] = g["close"].shift(-h) / df["close"] - 1.0

    event_mask = None
    if ann_col:
        event_mask = pd.to_datetime(df[ann_col], errors="coerce") == df["date"]
    elif op_col:
        # 실적 수치가 갱신되는 첫 거래일을 실적 반영일로 근사
        prev_op = g[op_col].shift(1)
        event_mask = df[op_col].notna() & prev_op.notna() & (df[op_col] != prev_op)

    if event_mask is not None and event_mask.sum() > 0:
        ev = df[event_mask].copy()
        res2 = pd.DataFrame({
            f"{h}d_med": ev[f"fwd_{h}"].median() for h in (1, 5, 10, 20)
        }, index=["event"]).T
        res2["all_med"] = [df[f"fwd_{h}"].median() for h in (1, 5, 10, 20)]
        res2["excess_med"] = res2["event"] - res2["all_med"]
        print(f"\n[재무 2: 실적 반영일(n={len(ev):,}) 이후 수익률]\n", res2.round(4).to_string())
        fig, ax = plt.subplots(figsize=(8, 5))
        res2[["event", "all_med"]].plot(kind="bar", ax=ax)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(T("실적 반영 후 기간별 수익률 중앙값 vs 전체", "Post-Earnings Return Median vs All"))
        plt.tight_layout()
        plt.savefig("fin_q2.png", dpi=150)
        plt.close()
    else:
        print("\n[재무 2] 실적 발표일 또는 실적 변동 시점 컬럼을 찾지 못했습니다.")

    # ---------- 재무 3: 추정치 오차 ----------
    if op_col and est_op_col:
        sub3 = df.dropna(subset=[op_col, est_op_col, "fwd_5"]).copy()
        sub3 = sub3[sub3[est_op_col].abs() > 0]
        sub3["surprise"] = (sub3[op_col] - sub3[est_op_col]) / sub3[est_op_col].abs()
        sub3 = sub3[sub3["surprise"].between(-5, 5)]
        sub3["surp_bin"] = pd.cut(sub3["surprise"], bins=[-5, -0.2, -0.05, 0.05, 0.2, 5], labels=["큰쇼크", "소폭쇼크", "부합", "소폭서프", "큰서프"])
        res3 = sub3.groupby("surp_bin", observed=False)[["fwd_1", "fwd_5", "fwd_20"]].median()
        print("\n[재무 3: 추정치 오차(서프라이즈) 구간별 사후 수익률 중앙값]\n", res3.round(4).to_string())
        fig, ax = plt.subplots(figsize=(8, 5))
        res3.plot(kind="bar", ax=ax)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(T("추정치 오차 구간별 사후 수익률 중앙값", "Post-Surprise Return Median"))
        plt.tight_layout()
        plt.savefig("fin_q3.png", dpi=150)
        plt.close()
    else:
        print("\n[재무 3] 추정치 컬럼이 parquet에 없어 HTML 실적 표 파싱 병합이 필요합니다.")

    # ---------- 재무 4: 부채비율 대 변동성 ----------
    if debt_col:
        vol_df = df.groupby("code").agg(
            debt=(debt_col, "last"),
            vol=("ret", "std"),
            n=("ret", "count"),
            sector=("sector", "last")
        ).dropna()
        vol_df = vol_df[(vol_df["n"] >= 60) & (vol_df["debt"].between(0, 1000))]
        vol_df["debt_q"] = pd.qcut(vol_df["debt"], q=5, labels=["Q1(저부채)", "Q2", "Q3", "Q4", "Q5(고부채)"], duplicates="drop")
        res4 = vol_df.groupby("debt_q", observed=False)["vol"].agg(["count", "mean", "median"])
        print("\n[재무 4: 부채비율 분위별 일별 수익률 변동성]\n", res4.round(5).to_string())
        fig, ax = plt.subplots(figsize=(8, 5))
        (res4["median"] * np.sqrt(252)).plot(kind="bar", ax=ax, color="darkorange")
        ax.set_title(T("부채비율 분위별 연환산 변동성 (중앙값)", "Annualized Volatility by Debt Ratio Quintile"))
        plt.tight_layout()
        plt.savefig("fin_q4.png", dpi=150)
        plt.close()
    else:
        print("\n[재무 4] 부채비율/부채총계/자본총계 컬럼이 parquet에 없습니다.")

    # ---------- 재무 5: PBR < 1 ----------
    if pbr_col:
        sub5 = df.dropna(subset=[pbr_col]).copy()
        sub5 = sub5[sub5[pbr_col] > 0]
        last_d = sub5["date"].max()
        snap5 = sub5[sub5["date"] == last_d].copy()
        snap5["under_1"] = snap5[pbr_col] < 1.0
        ratio_all = snap5["under_1"].mean()
        sec_pbr = snap5.groupby("sector")["under_1"].agg(["count", "mean"]).query("count >= 10").sort_values("mean", ascending=False)
        print(f"\n[재무 5: 말기({last_d.date()}) PBR < 1 비율] 전체={ratio_all:.2%} ({int(snap5['under_1'].sum())}/{len(snap5)})")
        print("[재무 5: 업종별 PBR < 1 비율 상위 10]\n", sec_pbr.head(10).round(3).to_string())

        sub5["pbr_grp"] = np.where(sub5[pbr_col] < 1.0, "PBR < 1", "PBR >= 1")
        ret_pbr = sub5.groupby("pbr_grp")[["ret", "fwd_20"]].agg(["mean", "median"])
        print("[재무 5: PBR < 1 vs >= 1 수익률 비교]\n", ret_pbr.round(5).to_string())

        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        sec_pbr.head(10)["mean"].sort_values().plot(kind="barh", ax=ax[0], color="teal")
        ax[0].set_title(T("업종별 PBR < 1 종목 비중 상위 10", "Top 10 Sectors by PBR < 1 Share"))
        (sub5.groupby("pbr_grp")["ret"].mean() * 252).plot(kind="bar", ax=ax[1], color=["teal", "gray"])
        ax[1].axhline(0, color="k", lw=0.8)
        ax[1].set_title(T("PBR < 1 vs >= 1 연환산 평균 수익률", "Annualized Mean Return: PBR < 1 vs >= 1"))
        plt.tight_layout()
        plt.savefig("fin_q5.png", dpi=150)
        plt.close()
    else:
        print("\n[재무 5] PBR/BPS 컬럼이 parquet에 없습니다.")


if __name__ == "__main__":
    df_all, _ = load_data()
    analyze_financial_questions(df_all)
