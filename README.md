# 금융통계 기말 프로젝트: 주식 아카이브 분석

학교 서버(stat-server)의 네이버 금융 HTML 아카이브(2024-03-27 ~ 2026-09-02, 날짜 폴더 591개)를 읽어 날짜별 표(parquet)로 줄이고, 질문에 맞게 분석하고 시각화한다. 원본 HTML은 읽기만 하고 복사하지 않는다.

## 구성
| 폴더 | 내용 |
|---|---|
| `week1/` | 1주차: 전체 자료를 읽어 들인 범위 (그림, 결함 기록, 코드) |

## 공통 실행 순서
```bash
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install pandas lxml pyarrow matplotlib

# 1) 원본 HTML -> 날짜별 parquet (서버에서, 약 20분)
nohup python -u stock_archive_example.py --step 4 --workers 4 > step4.log 2>&1 &
ls parquet | wc -l        # 날짜 폴더 수와 같아지면 끝

# 2) 읽어 들인 범위 그림과 숫자
python range_figure.py 2>&1 | tee range_summary.txt
```
`parquet/` 폴더는 용량이 커서 저장소에 올리지 않았다. 위 1)로 다시 만들 수 있다.

## 주의
- 서버 경로 `/mnt/data/stock` 은 학교 서버에서만 접근할 수 있다.
- 자료 결함과 처리 방법은 `week1/week1_defects.md` 에 기록한다.
