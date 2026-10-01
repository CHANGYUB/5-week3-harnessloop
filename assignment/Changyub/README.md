# 코드 리뷰 Agent — Changyub

## Agent 목적

Python 코드 파일을 자동으로 읽고 분석하여, **마크다운 표와 ASCII 막대 그래프**로 가독성을 높인 코드 리뷰 보고서를 생성합니다.

단순 텍스트 리뷰 대신, 메트릭 표·함수별 리뷰 표·복잡도 시각화·심각도 분류를 통해 리뷰어와 개발자 모두가 한눈에 파악할 수 있는 구조화된 보고서를 만듭니다.

---

## 주요 기능

- **코드 파일 읽기**: 저장소 내 소스 코드를 안전하게 읽음 (숨김 파일 차단)
- **AST 기반 메트릭 분석**: 함수 수·클래스 수·라인 수·인자 수·독스트링 유무를 자동 측정
- **모범 사례 검색**: 한국어 위키백과에서 언어·패턴 관련 모범 사례 검색
- **보고서 저장**: 완성된 리뷰를 마크다운 파일로 저장
- **검증 & 재실행**: 표·그래프·심각도 마커 포함 여부를 자동 검증하고 불충분하면 재작성 요청

---

## Tools

| Tool 이름 | 역할 | 방식 |
|---|---|---|
| `read_code_file` | 저장소 내 소스 파일을 UTF-8로 읽어 반환 | 직접 구현 |
| `analyze_code_metrics` | Python AST로 함수·클래스·라인·인자 수 등 메트릭 측정 | 직접 구현 |
| `search_best_practices` | 한국어 위키백과에서 프로그래밍 모범 사례 검색 | 베이스라인 `wikipedia_search` 재사용 |
| `save_review_report` | 완성된 리뷰 보고서를 `reviews/` 폴더에 저장 | 직접 구현 |

---

## Agent Flow

```
사용자 입력 (리뷰 대상 파일 경로)
          ↓
     [agent 노드]
    LangChain create_agent
          ↓
  Tool Call 있음? ──Yes──→ [tools 노드] ──→ [agent 노드]
          ↓ No
     [verify 노드]
   규칙 검증 + LLM Evaluator
          ↓
   통과? or MAX_RUNS? ──Yes──→ 종료 (최종 보고서)
          ↓ No
   [feedback 노드]
   피드백 메시지 → [agent 노드]
```

**Tool 실행 순서 (모델이 결정):**
1. `read_code_file` — 코드 파일 내용 읽기
2. `analyze_code_metrics` — AST 메트릭 측정
3. `search_best_practices` — 관련 개념 위키백과 검색
4. *(보고서 작성 후)* `save_review_report` — 결과 저장

---

## Harness

| 구성 요소 | 구현 방식 | 역할 |
|---|---|---|
| **Execution Budget** | `recursion_limit=40` | LangGraph Agent Loop의 최대 반복 횟수 제한 |
| **Timeout** | `ChatOpenAI(timeout=30.0)` | 모델 응답 30초 초과 시 예외 발생 |
| **Error Handling** | `try/except` in `run_agent_with_harness()` | LangChain/OpenAI 예외 포착 및 trace 기록 |
| **Trace** | `RunTrace` dataclass | Tool 호출 이름·인자·오류를 순서대로 기록 및 출력 |
| **Stop Condition** | `route_verify()` in Graph | 검증 통과 또는 MAX_RUNS 도달 시 Graph 종료 |

---

## Loop

| 구성 요소 | 구현 방식 | 역할 |
|---|---|---|
| **Verification (규칙)** | `verify_review()` | 표 행 수, ▓ 그래프, 심각도 마커, 섹션 수, Tool 사용 여부 검사 |
| **Verification (LLM)** | `evaluate_review_with_llm()` | 구체성·근거·가독성 기여 여부를 LLM으로 평가 |
| **Conditional Stop** | `route_verify()` | `passed=True` 또는 `iterations >= MAX_RUNS`이면 종료 |
| **Feedback** | `feedback_node()` | 검증 문제를 HumanMessage로 변환해 다음 Agent 실행에 전달 |
| **Loop Limit** | `MAX_RUNS = 3` | 최대 3회 재시도 후 강제 종료 |

---

## 리뷰 보고서 형식

Agent가 생성하는 보고서는 다음 섹션을 포함합니다:

1. **요약** — 코드 목적과 전반적 품질 (2~3 문장)
2. **메트릭 표** — `| 항목 | 값 | 평가 |` 형식
3. **함수별 리뷰 표** — `| 함수명 | 라인수 | 인자수 | 문제점 | 개선 제안 |` 형식
4. **복잡도 ASCII 막대 그래프** — `▓` 기호로 함수별 라인 수 시각화 (▓ 1개 = 5라인)
5. **문제점 목록** — 🔴 높음 / 🟡 중간 / 🟢 낮음 심각도 분류
6. **개선 제안** — 코드 예시 포함
7. **참고 자료** — Wikipedia URL

---

## 실행 방법

### 1. 환경 설정

```bash
# 저장소 루트의 .env 파일에 API 키 설정
cp .env.example .env
# .env를 열어 OPENAI_API_KEY 설정
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. Agent 실행

#### 방법 A: 직접 실행 (기본 예시)

```bash
cd 5-week3-harnessloop
python assignment/Changyub/agent.py
```

기본으로 `tools/wikipedia_search.py`를 리뷰하고 `assignment/Changyub/reviews/review_wikipedia_search.md`에 저장합니다.

#### 방법 B: Python에서 함수 호출

```python
# Notebook 또는 다른 스크립트에서
import sys
sys.path.insert(0, ".")   # 저장소 루트에서 실행

from assignment.Changyub.agent import run_code_review_loop

result = run_code_review_loop(
    target_file="tools/wikipedia_search.py",   # 리뷰할 파일
    extra_context="오류 처리와 타임아웃 패턴에 주목하세요.",
    save_as="my_review.md",                    # 저장 파일명
)

print("통과:", result["passed"])
print("시도 횟수:", result["iterations"])
```

---

## 실행 예시

```
Python: 3.12.14 | Model: gpt-4.1-mini | Max Runs: 3
[TOOL] read_code_file | input: ['path']
[TOOL] analyze_code_metrics | input: ['path']
[TOOL] search_best_practices | input: ['query']
[TOOL] save_review_report | input: ['filename', 'content']
[AGENT] CompiledStateGraph 생성 완료

[LOOP] 코드 리뷰 시작: tools/wikipedia_search.py
[LOOP] 최대 시도: 3회 | recursion_limit: 40

============================================================
[HARNESS] Run #1 시작 | thread: a3f2b1c0...
[HARNESS] Task: tools/wikipedia_search.py 코드 리뷰...

  [TRACE] Tool call: read_code_file | args: {"path": "tools/wikipedia_search.py"}
  [TRACE] Tool call: analyze_code_metrics | args: {"path": "tools/wikipedia_search.py"}
  [TRACE] Tool call: search_best_practices | args: {"query": "Python API 오류 처리 패턴"}
  [TRACE] Tool call: save_review_report | args: {"filename": "review_wikipedia_search.md"}

[VERIFY] PASS ✅ | rules: 0 | LLM: pass

============================================================
[LOOP] 완료 | 통과: ✅ | 시도: 1회

────────────────────────────────────────────────────────────
📋 최종 코드 리뷰 보고서
────────────────────────────────────────────────────────────

## 요약
wikipedia_search.py는 MediaWiki API를 통해 한국어 위키백과를 검색하는 단일 함수 모듈입니다...

## 메트릭
| 항목 | 값 | 평가 |
|---|---|---|
| 총 라인 수 | 49 | 🟢 적절 |
| 함수 수 | 1 | 🟢 단일 책임 |
| 평균 함수 길이 | 33줄 | 🟡 약간 김 |

## 복잡도 그래프
wikipedia_search  ▓▓▓▓▓▓▓ (33줄)

## 문제점
🟡 1. 타임아웃 파라미터가 키워드 전용(*)으로 제한되어 있어 위치 인자로 전달 불가...

[DONE] passed=True | iterations=1
[DONE] 리뷰 보고서: assignment/Changyub/reviews/review_wikipedia_search.md
```

---

## 파일 구조

```
assignment/Changyub/
├── README.md          # 이 문서
├── agent.py           # Agent 전체 코드
└── reviews/           # 생성된 리뷰 보고서 저장 폴더 (자동 생성)
    └── *.md
```
