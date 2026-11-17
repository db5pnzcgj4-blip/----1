"""
1주차 제출용: 읽어 들인 범위를 보이는 그림과 숫자

사전 조건
    python stock_archive_example.py --step 4   # parquet/ 에 날짜별 파일 생성

실행
    python range_figure.py 2>&1 | tee range_summary.txt

만드는 것
    fig_range.png      날짜별로 읽은 종목 수 (수집 누락일과 초기 부분 수집 구간 표시)
    range_summary.txt  기간, 날짜 수, 종목 수, 행 수, 읽지 못한 부분
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import pandas as pd
from pathlib import Path

ROOT = Path("/mnt/data/stock")   # 서버의 원본 (읽기만 한다)
CACHE = Path("parquet")          # step 4 가 만든 날짜별 표

# 종목 수가 비정상적으로 적은 날 (수집 누락). 찾는 기준은 week1_defects.md 참고
BAD = ["2024-09-04", "2024-09-11", "2024-09-12", "2024-10-10", "2024-10-15",
       "2024-10-18", "2024-11-21", "2024-11-22", "2025-04-25", "2025-08-08"]
FULL_START = "2024-04-15"        # 이날부터 전 종목(약 3,900개)이 수집됨

# 한글 폰트가 있으면 한글 제목, 없으면 영어
names = {f.name for f in fm.fontManager.ttflist}
KO = next((n for n in ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR",
                       "Malgun Gothic", "AppleGothic"] if n in names), None)
if KO:
    plt.rcParams["font.family"] = KO
plt.rcParams["axes.unicode_minus"] = False
T = (lambda ko, en: ko) if KO else (lambda ko, en: en)

# 1) 날짜별 parquet 를 하나씩 읽는다 (폴더째 읽으면 error 열 형식이 충돌한다)
per_day, codes, n_err, n_rows = {}, set(), 0, 0
nulls = None
obs = None
files = sorted(CACHE.glob("*.parquet"))
for f in files:
    d = pd.read_parquet(f)
    per_day[pd.Timestamp(f.stem)] = len(d)
    codes.update(d["code"].unique())
    vc = d["code"].value_counts()
    obs = vc if obs is None else obs.add(vc, fill_value=0)
    n_err += int(d["error"].notna().sum())
    n_rows += len(d)
    nn = d.drop(columns=["date", "code", "error"]).isna().sum()
    nulls = nn if nulls is None else nulls.add(nn, fill_value=0)
s = pd.Series(per_day).sort_index()
miss = (nulls / n_rows * 100).sort_values()   # 열별 결측 비율(%)

# 2) 원본에서 읽지 못한 부분
src_dirs = [p for p in ROOT.iterdir() if p.is_dir()]
non_date = []
for p in src_dirs:
    try:
        pd.Timestamp(p.name)
    except ValueError:
        non_date.append(p.name)
date_dirs = len(src_dirs) - len(non_date)

print("=== 읽어 들인 범위 ===")
print("기간            :", s.index.min().date(), "~", s.index.max().date())
print("날짜 폴더(원본) :", date_dirs, "개")
print("읽은 날짜(표)   :", len(s), "개")
print("종목 수         :", f"{len(codes):,}")
print("행 수           :", f"{n_rows:,}")
print("하루 종목 수    : 최소", s.min(), "/ 중앙값", int(s.median()), "/ 최대", s.max())
print()
print("=== 읽지 못한 부분 ===")
print("날짜가 아니라서 제외한 폴더:", non_date)
print("표가 없는 날짜 폴더        :", date_dirs - len(s), "개")
print("파싱 실패 행               :", n_err, f"({n_err / max(n_rows, 1):.3%})")
print("종목 수 급락일(수집 누락)  :", len(BAD), "일")
print("\n=== 열별 결측 비율(%) ===")
print(miss.round(2).to_string())
print("\n=== 종목별 관측 일수 ===")
print("전 기간(", len(s), "일) 내내 관측된 종목:", int((obs == len(s)).sum()), "개")
print("관측 일수가 전체의 절반 미만인 종목   :", int((obs < len(s) / 2).sum()), "개 (상장·폐지 영향)")

# 3) 그림
fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4.5), gridspec_kw={"width_ratios": [2, 1, 1]})
ax.plot(s.index, s.values, lw=1.4, color="tab:blue")
ax.axvspan(s.index.min(), pd.Timestamp(FULL_START), color="orange", alpha=0.2,
           label=T("초기 부분 수집 (약 950종목)", "partial collection (~950 stocks)"))
bad = s.reindex(pd.to_datetime(BAD)).dropna()
ax.scatter(bad.index, bad.values, color="red", zorder=3, s=22,
           label=T("수집 누락일 10일", "missing-collection days (10)"))
ax.set_title(T("날짜별로 읽은 종목 수", "Stocks read per day"))
ax.set_ylabel(T("종목 수", "number of stocks"))
ax.legend(loc="lower right", fontsize=8)

# 오른쪽: 열별 결측 비율. 업종이 없는 종목(ETF 등), 재무 지표가 없는 종목이 여기서 보인다
miss.plot(kind="barh", ax=ax2, color="tab:gray")
ax2.set_title(T("열별 결측 비율", "Missing rate by column"))
ax2.set_xlabel("%")

# 가운데 아닌 오른쪽 끝: 종목마다 며칠 관측됐는가. 중간에 상장·폐지된 종목이 왼쪽 막대로 보인다
ax3.hist(obs.values, bins=30, color="tab:green")
ax3.set_title(T("종목별 관측 일수", "Days observed per stock"))
ax3.set_xlabel(T("관측된 날 수", "days"))
ax3.set_ylabel(T("종목 수", "stocks"))
plt.tight_layout()
plt.savefig("fig_range.png", dpi=150)

# 패널마다 따로도 저장한다 (fig_range_1: 기간·종목 수, _2: 결측 분포, _3: 종목별 관측 일수)
fig.canvas.draw()
for i, a in enumerate([ax, ax2, ax3], 1):
    bb = a.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"fig_range_{i}.png", dpi=150, bbox_inches=bb)
print("\n저장: fig_range.png, fig_range_1.png, fig_range_2.png, fig_range_3.png")
