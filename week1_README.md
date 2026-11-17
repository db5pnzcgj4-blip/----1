# 1주차: 전체 자료 읽어 들이기

## 이번 주 질문 한 줄
2년 반 치 전체 자료를 빠짐없이 읽었는가, 읽지 못한 부분이나 비어 있는 날은 어디인가?

## 그림
`fig_range.png` : 왼쪽은 날짜별로 읽은 종목 수(기간과 종목 수), 가운데는 열별 결측 비율(결측 분포), 오른쪽은 종목마다 며칠 관측됐는가(상장·폐지로 중간에 시작하거나 끝난 종목).
- 주황색 구간: 2024-03-27 ~ 2024-04-12, 하루 약 950종목만 수집된 구간
- 빨간 점: 종목 수가 급락한 날 10일 (수집 누락)
- 2024-04-15 이후는 약 3,900 ~ 4,300종목으로 완만하게 늘어남

## 읽어 들인 범위
`range_figure.py` 를 실행하면 `range_summary.txt` 에 아래 값이 나온다. 아래 표는 그 출력으로 채운다.

| 항목 | 값 |
|---|---|
| 기간 | 2024-03-27 ~ 2026-09-02 |
| 날짜 폴더 | 591개 (읽은 날짜 591개) |
| 종목 수 | `range_summary.txt` 의 "종목 수" 줄 |
| 행 수 | `range_summary.txt` 의 "행 수" 줄 |
| 읽지 못한 부분 | 날짜가 아닌 폴더 2개(`html`, `stock_code`) 제외. 파싱 실패 행 수는 `range_summary.txt` 의 "파싱 실패 행" 줄 |
| 열별 결측 비율, 종목별 관측 일수 | `range_summary.txt` 와 `fig_range_2.png`, `fig_range_3.png` |

`range_summary.txt` 는 `range_figure.py` 의 실행 결과 그대로이며 이 저장소에 함께 올렸다.

## 결함
`week1_defects.md` 에 정리했다.

## 실행 순서
1. 서버(stat-server)에 접속한다.
2. 환경을 만든다.
   ```bash
   python3 -m venv ~/venv
   source ~/venv/bin/activate
   pip install pandas lxml pyarrow matplotlib
   ```
3. 원본 HTML을 날짜별 parquet 로 줄인다. 원본은 읽기만 한다.
   ```bash
   nohup python -u stock_archive_example.py --step 4 --workers 4 > step4.log 2>&1 &
   ls parquet | wc -l     # 날짜 폴더 수와 같아지면 끝 (약 20분)
   ```
4. 범위 그림과 숫자를 만든다.
   ```bash
   python range_figure.py 2>&1 | tee range_summary.txt
   ```

## 파일
| 파일 | 설명 |
|---|---|
| `stock_archive_example.py` | HTML을 읽어 날짜별 parquet 로 만드는 코드 (시가총액 파서 수정 포함) |
| `range_figure.py` | 읽어 들인 범위 그림과 숫자 |
| `fig_range.png` | 1주차 그림 |
| `week1_defects.md` | 자료 결함 기록 |

`parquet/` 폴더는 용량이 커서 올리지 않았다. 위 3번으로 다시 만들 수 있다.
