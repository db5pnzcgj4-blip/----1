"""실적 표(기업실적분석) 읽기.
사용:
  python fin_parse.py 2026-09-02                      # 하루만 -> fin_2026-09-02.parquet
  python fin_parse.py --all --workers 4               # 모든 날짜 -> fin_parquet/날짜.parquet (이어서 실행 가능)
환경변수 STOCK_ROOT (기본 /mnt/data/stock)
날짜·종목마다 한 줄. 최근 실제 분기/연간 값과 추정치(E) 값만 뽑는다."""
import os, re, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import lxml.html

ROOT = Path(os.environ.get("STOCK_ROOT", "/mnt/data/stock"))
OUT = Path("fin_parquet")
METRICS = {"매출액": "rev", "영업이익": "op", "당기순이익": "ni", "영업이익률": "opm", "부채비율": "debt"}
PERIOD = re.compile(r"^\d{4}\.\d{2}")

def num(x):
    x = x.replace(",", "").strip()
    try: return float(x)
    except ValueError: return np.nan

def read_html(path):
    raw = Path(path).read_bytes()
    try: return raw.decode("utf-8")
    except UnicodeDecodeError: return raw.decode("euc-kr", "replace")

def pick(labels, vals, want_est):
    """라벨과 값 목록에서 (E)가 아닌 마지막 실제 열, 또는 (E) 열을 고른다."""
    idx = [i for i, l in enumerate(labels) if l.endswith("(E)") == want_est]
    if not want_est:
        idx = [i for i in idx if not np.isnan(vals.get("rev", [np.nan] * len(labels))[i]) or not np.isnan(vals.get("op", [np.nan] * len(labels))[i])]
    return idx[-1] if idx else None

def parse_file(path):
    code = Path(path).stem
    rec = {"code": code, "has_table": False}
    try:
        doc = lxml.html.fromstring(read_html(path))
        tb = doc.xpath('//div[contains(@class,"cop_analysis")]//table')
        if not tb: return rec
        trs = tb[0].xpath(".//tr")
        labels = []
        for tr in trs:
            ths = [" ".join(t.text_content().split()) for t in tr.xpath("./th")]
            if len(ths) >= 4 and all(PERIOD.match(t) for t in ths):
                labels = ths; break
        if not labels: return rec
        spans = [int(t.get("colspan", 1)) for t in trs[0].xpath("./th")[1:]]
        n_ann = spans[0] if len(spans) >= 2 else 4
        vals = {}
        for tr in trs:
            cells = tr.xpath("./th|./td")
            if len(cells) == len(labels) + 1:
                name = " ".join(cells[0].text_content().split())
                if name in METRICS:
                    vals[METRICS[name]] = [num(c.text_content()) for c in cells[1:]]
        rec["has_table"] = True
        for tag, sl in (("q", slice(n_ann, None)), ("y", slice(0, n_ann))):
            lab = labels[sl]; v = {k: x[sl] for k, x in vals.items()}
            i = pick(lab, v, False)
            rec[f"last_{tag}"] = lab[i] if i is not None else None
            for k in ("rev", "op", "ni", "opm", "debt"):
                rec[f"{tag}_{k}"] = v[k][i] if (i is not None and k in v) else np.nan
            j = pick(lab, v, True)
            rec[f"est_{tag}"] = lab[j] if j is not None else None
            for k in ("rev", "op"):
                rec[f"est{tag}_{k}"] = v[k][j] if (j is not None and k in v) else np.nan
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec

def parse_date(date):
    files = sorted((ROOT / date).glob("*.html"))
    df = pd.DataFrame([parse_file(f) for f in files])
    df.insert(0, "date", date)
    if "error" not in df: df["error"] = None
    df["error"] = df["error"].astype("object")
    return df

def run_day(date):
    out = OUT / f"{date}.parquet"
    if out.exists(): return date, -1
    df = parse_date(date); df.to_parquet(out); return date, len(df)

def is_date(p): return p.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", p.name)

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--all":
        w = int(args[args.index("--workers") + 1]) if "--workers" in args else 4
        OUT.mkdir(exist_ok=True)
        dates = sorted(p.name for p in ROOT.iterdir() if is_date(p))
        print("날짜 폴더:", len(dates), "workers:", w, flush=True)
        with ProcessPoolExecutor(w) as ex:
            for i, (d, n) in enumerate(ex.map(run_day, dates), 1):
                if i % 20 == 0 or i == len(dates): print(i, "/", len(dates), d, n, flush=True)
        print("done")
    else:
        date = args[0] if args else sorted(p.name for p in ROOT.iterdir() if is_date(p))[-1]
        df = parse_date(date); df.to_parquet(f"fin_{date}.parquet")
        print(date, "파일", len(df), "표 있음", int(df["has_table"].sum()))
        for c in ["last_q", "q_op", "q_opm", "q_debt", "est_q", "estq_op", "last_y", "y_opm", "est_y", "esty_op"]:
            if c in df: print(f"{c:8s} 채워진 비율 {df[c].notna().mean():.3f}")
        print("오류 행:", int(df["error"].notna().sum()))
        print(df.head(5).to_string())
