"""
코드 리뷰 Agent — 표와 ASCII 그래프로 가독성 높은 리뷰를 생성합니다.

구성 요소
---------
Tools (4개)
  1. read_code_file    - 지정한 코드 파일을 읽어 내용을 반환합니다.
  2. analyze_code_metrics - AST로 함수·클래스 수, 라인 수, 복잡도 지표를 측정합니다.
  3. search_best_practices - 한국어 위키백과에서 언어·패턴 관련 모범 사례를 검색합니다.
  4. save_review_report - 완성된 리뷰 보고서를 파일로 저장합니다.

Agent Loop  : LangChain create_agent (Tool Loop 자동 관리)
Harness     : recursion_limit(Execution Budget), timeout, error handling, trace 출력
Loop Eng.   : 리뷰 품질 검증(표·그래프 포함 여부) → LLM Evaluator → 피드백 재실행 (MAX_RUNS=3)
"""

# ──────────────────────────────────────────────
# 0. 의존성 및 환경 설정
# ──────────────────────────────────────────────
import ast
import json
import os
import re
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

# 저장소 루트 탐색 — cwd 상위 경로와 __file__ 상위 경로를 모두 탐색합니다.
# IDE나 절대 경로로 실행해도 스크립트 파일 위치 기준으로 루트를 찾을 수 있습니다.
_search_paths = dict.fromkeys([
    *[Path.cwd(), *Path.cwd().parents],
    *[Path(__file__).resolve(), *Path(__file__).resolve().parents],
])  # dict.fromkeys로 순서 유지 + 중복 제거
PROJECT_ROOT = next(
    (path for path in _search_paths
     if path.is_dir() and (path / "requirements.txt").is_file() and (path / "tools").is_dir()),
    None,
)
if PROJECT_ROOT is None:
    raise RuntimeError(
        "저장소 루트를 찾을 수 없습니다. "
        "5-week3-harnessloop/ 안에 requirements.txt와 tools/ 폴더가 있어야 합니다."
    )

sys.path.insert(0, str(PROJECT_ROOT))
from tools.wikipedia_search import wikipedia_search  # 기존 베이스라인 Tool 재사용

load_dotenv(PROJECT_ROOT / ".env")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
EVALUATOR_MODEL = "gpt-4.1-mini"
MODEL_TIMEOUT = 30.0
WIKIPEDIA_TIMEOUT = 15.0
MAX_RUNS = 3          # 리뷰 작성 최대 시도 횟수
RECURSION_LIMIT = 40  # Harness: Agent Loop Execution Budget

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key or api_key == "your-openai-api-key":
    raise RuntimeError("실행 전에 루트 .env에 OPENAI_API_KEY를 설정하세요.")

model = ChatOpenAI(model=MODEL, timeout=MODEL_TIMEOUT, max_retries=0)
evaluator_model = ChatOpenAI(model=EVALUATOR_MODEL, timeout=MODEL_TIMEOUT, max_retries=0)

print(f"Python: {sys.version.split()[0]} | Model: {MODEL} | Max Runs: {MAX_RUNS}")


# ──────────────────────────────────────────────
# 1. Tools 정의
# ──────────────────────────────────────────────

# ── Tool 1: 코드 파일 읽기 ──────────────────────
@tool("read_code_file")
def read_code_file_tool(path: str) -> str:
    """저장소 안의 소스 코드 파일을 읽어 전체 내용을 반환합니다.
    숨김 파일(.으로 시작하는 경로)은 읽을 수 없습니다."""
    file_path = (PROJECT_ROOT / path).resolve()
    if not file_path.is_relative_to(PROJECT_ROOT):
        raise ValueError("저장소 안의 파일만 읽을 수 있습니다.")
    if any(part.startswith(".") for part in file_path.relative_to(PROJECT_ROOT).parts):
        raise ValueError("숨김 파일은 읽을 수 없습니다.")
    return file_path.read_text(encoding="utf-8")


# ── Tool 2: 코드 메트릭 분석 ─────────────────────
@tool("analyze_code_metrics")
def analyze_code_metrics_tool(path: str) -> str:
    """Python 파일을 AST로 분석하여 함수 수, 클래스 수, 총 라인 수,
    각 함수의 라인 수·인자 수를 JSON으로 반환합니다.
    Python 외 파일은 기본 라인 통계만 반환합니다."""
    file_path = (PROJECT_ROOT / path).resolve()
    if not file_path.is_relative_to(PROJECT_ROOT):
        raise ValueError("저장소 안의 파일만 분석할 수 있습니다.")

    source = file_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    total_lines = len(lines)
    code_lines = sum(1 for l in lines if l.strip() and not l.strip().startswith("#"))
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))

    # Python이 아닌 파일은 기본 통계만 반환
    if file_path.suffix not in (".py",):
        return json.dumps({
            "file": path,
            "language": file_path.suffix.lstrip(".") or "unknown",
            "total_lines": total_lines,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "functions": [],
            "classes": [],
            "note": "AST 분석은 Python 파일(.py)에서만 지원됩니다.",
        }, ensure_ascii=False, indent=2)

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return json.dumps({"error": f"구문 오류: {e}", "file": path}, ensure_ascii=False)

    functions = []
    classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            func_lines = (node.end_lineno or node.lineno) - node.lineno + 1
            functions.append({
                "name": node.name,
                "line": node.lineno,
                "lines": func_lines,
                "args": len(node.args.args),
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "has_docstring": (
                    isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    if node.body else False
                ),
            })
        elif isinstance(node, ast.ClassDef):
            class_lines = (node.end_lineno or node.lineno) - node.lineno + 1
            methods = [n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
            classes.append({
                "name": node.name,
                "line": node.lineno,
                "lines": class_lines,
                "methods": methods,
            })

    return json.dumps({
        "file": path,
        "language": "python",
        "total_lines": total_lines,
        "code_lines": code_lines,
        "comment_lines": comment_lines,
        "function_count": len(functions),
        "class_count": len(classes),
        "functions": functions,
        "classes": classes,
    }, ensure_ascii=False, indent=2)


# ── Tool 3: 모범 사례 검색 ──────────────────────
@tool("search_best_practices")
def search_best_practices_tool(query: str) -> str:
    """한국어 위키백과에서 프로그래밍 언어·디자인 패턴·리팩토링 관련 모범 사례를 검색합니다.
    예: 'Python 코드 가독성', '단일 책임 원칙', '클린 코드'"""
    return wikipedia_search(query, timeout_seconds=WIKIPEDIA_TIMEOUT)


# ── Tool 4: 리뷰 보고서 저장 ─────────────────────
REVIEW_OUTPUT_DIR = PROJECT_ROOT / "assignment" / "Changyub" / "reviews"

@tool("save_review_report")
def save_review_report_tool(filename: str, content: str) -> str:
    """완성된 코드 리뷰 보고서를 assignment/Changyub/reviews/ 폴더에 저장합니다.
    filename: 저장할 파일 이름 (예: review_agent.md)
    content: 저장할 마크다운 리뷰 내용"""
    REVIEW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REVIEW_OUTPUT_DIR / filename
    out_path.write_text(content, encoding="utf-8")
    return f"리뷰 보고서가 저장되었습니다: {out_path.relative_to(PROJECT_ROOT)}"


review_tools = [read_code_file_tool, analyze_code_metrics_tool,
                search_best_practices_tool, save_review_report_tool]

for t in review_tools:
    print(f"[TOOL] {t.name} | input: {list(t.args.keys())}")


# ──────────────────────────────────────────────
# 2. Agent 생성 (LangChain create_agent)
# ──────────────────────────────────────────────

SYSTEM_INSTRUCTIONS = (
    "당신은 전문 코드 리뷰어입니다. "
    "코드를 읽고 메트릭을 분석한 뒤, 아래 형식을 반드시 지켜 리뷰 보고서를 작성하세요.\n\n"
    "## 필수 보고서 형식\n"
    "1. **요약** — 코드 목적과 전반적 품질을 2~3 문장으로 설명합니다.\n"
    "2. **메트릭 표** — | 항목 | 값 | 평가 | 형식의 마크다운 표를 반드시 포함합니다.\n"
    "3. **함수별 리뷰 표** — | 함수명 | 라인수 | 인자수 | 문제점 | 개선 제안 | 형식의 표를 포함합니다.\n"
    "4. **복잡도 ASCII 막대 그래프** — 함수별 라인 수를 ▓ 문자로 시각화합니다. "
    "예: `my_func  ▓▓▓▓▓ (25줄)`. 각 ▓은 5라인을 나타냅니다.\n"
    "5. **문제점 목록** — 심각도(🔴 높음 / 🟡 중간 / 🟢 낮음)와 함께 번호 목록으로 기술합니다.\n"
    "6. **개선 제안** — 코드 예시를 포함한 구체적 개선 방법을 제시합니다.\n"
    "7. **참고 자료** — search_best_practices로 찾은 Wikipedia URL을 넣습니다.\n\n"
    "추측이나 지어낸 정보는 쓰지 마세요. 반드시 read_code_file과 analyze_code_metrics를 "
    "모두 사용한 뒤 리뷰를 작성하세요."
)

langchain_agent = create_agent(model=model, tools=review_tools, system_prompt=SYSTEM_INSTRUCTIONS)
print(f"[AGENT] {type(langchain_agent).__name__} 생성 완료")


# ──────────────────────────────────────────────
# 3. Harness 구성 요소
# ──────────────────────────────────────────────

# ── 3-A. Trace 로거 ──────────────────────────
@dataclass
class RunTrace:
    """한 번의 Agent Run에서 발생한 Tool 호출과 오류를 기록합니다."""
    run_id: int
    tool_calls: list[dict] = field(default_factory=list)
    tool_errors: list[str] = field(default_factory=list)
    answer: str = ""

    def log_call(self, name: str, args: dict) -> None:
        self.tool_calls.append({"tool": name, "args": args})
        print(f"  [TRACE] Tool call: {name} | args: {json.dumps(args, ensure_ascii=False)[:80]}")

    def log_error(self, msg: str) -> None:
        self.tool_errors.append(msg)
        print(f"  [TRACE] ERROR: {msg}")

    def summary(self) -> str:
        return (
            f"Run #{self.run_id} | Tool calls: {len(self.tool_calls)} | "
            f"Errors: {len(self.tool_errors)}"
        )


def extract_trace(messages: list[AnyMessage], run_id: int) -> RunTrace:
    """메시지 목록에서 Tool 호출과 오류를 추출해 RunTrace를 만듭니다."""
    trace = RunTrace(run_id=run_id)
    tool_result_map: dict[str, str] = {}

    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_result_map[msg.tool_call_id] = msg.status or "success"

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                status = tool_result_map.get(call["id"], "pending")
                if status == "error":
                    trace.log_error(f"{call['name']} 실패")
                else:
                    trace.log_call(call["name"], call["args"])
    return trace


# ── 3-B. Stop Condition & Budget ─────────────
def run_agent_with_harness(
    task: str,
    run_id: int,
    thread_id: str,
    graph: CompiledStateGraph,
) -> tuple[str, RunTrace]:
    """
    Harness로 감싼 Agent 실행.
    - Execution Budget: recursion_limit=RECURSION_LIMIT
    - Error Handling: LangChain/OpenAI 예외를 잡아 trace에 기록
    - Stop Condition: recursion_limit 초과 시 빈 문자열 반환
    - Trace: RunTrace로 Tool 호출 기록
    """
    print(f"\n{'='*60}")
    print(f"[HARNESS] Run #{run_id} 시작 | thread: {thread_id[:8]}...")
    print(f"[HARNESS] Task: {task[:100]}")
    print(f"{'='*60}")

    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=task)]},
            config={
                "recursion_limit": RECURSION_LIMIT,
                "configurable": {"thread_id": thread_id},
            },
        )
    except Exception as e:  # Harness: Error Handling
        trace = RunTrace(run_id=run_id)
        trace.log_error(f"Agent 실행 실패: {type(e).__name__}: {e}")
        print(f"[HARNESS] ❌ 실행 오류: {e}")
        return "", trace

    messages = result.get("messages", [])
    trace = extract_trace(messages, run_id)

    # 마지막 AIMessage의 내용을 최종 답변으로 추출
    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            answer = str(msg.content)
            break

    trace.answer = answer
    print(f"\n[HARNESS] Run #{run_id} 완료 | {trace.summary()}")
    return answer, trace


# ──────────────────────────────────────────────
# 4. 검증(Verification) 함수
# ──────────────────────────────────────────────

# 리뷰 보고서가 반드시 포함해야 할 섹션과 패턴
REQUIRED_SECTIONS = [
    "## ",           # 최소 하나의 섹션 헤더
    "|",             # 마크다운 표 (파이프 문자)
    "▓",             # ASCII 막대 그래프
]
SEVERITY_MARKERS = ["🔴", "🟡", "🟢"]  # 문제점 심각도 표시


def verify_review(answer: str, messages: list[AnyMessage]) -> list[str]:
    """
    리뷰 보고서의 필수 구성 요소를 검증합니다.
    반환값: 발견된 문제 목록 (빈 목록이면 통과)
    """
    issues = []

    # 표 포함 여부 (최소 2개의 파이프 행)
    table_rows = [l for l in answer.splitlines() if "|" in l]
    if len(table_rows) < 4:
        issues.append("마크다운 표가 충분하지 않습니다. 메트릭 표와 함수별 리뷰 표를 모두 포함하세요.")

    # ASCII 막대 그래프 포함 여부
    if "▓" not in answer:
        issues.append("함수별 복잡도 ASCII 막대 그래프(▓)가 없습니다. 각 함수의 라인 수를 ▓로 시각화하세요.")

    # 심각도 마커 포함 여부
    if not any(m in answer for m in SEVERITY_MARKERS):
        issues.append("문제점 심각도(🔴/🟡/🟢)가 없습니다. 각 문제에 심각도를 표시하세요.")

    # 필수 섹션(##) 포함 여부
    section_count = sum(1 for l in answer.splitlines() if l.strip().startswith("## "))
    if section_count < 3:
        issues.append(f"섹션(##)이 {section_count}개입니다. 요약·메트릭·문제점·개선 제안 등 3개 이상의 섹션이 필요합니다.")

    # Tool 사용 검증: read_code_file과 analyze_code_metrics 모두 호출했는지
    called_tools = set()
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.status != "error":
            pass  # ToolMessage에는 name 속성이 있음
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls:
                called_tools.add(call["name"])

    if "read_code_file" not in called_tools:
        issues.append("read_code_file 도구를 사용하여 코드를 읽어야 합니다.")
    if "analyze_code_metrics" not in called_tools:
        issues.append("analyze_code_metrics 도구를 사용하여 코드 메트릭을 분석해야 합니다.")

    return issues


# ──────────────────────────────────────────────
# 5. LLM Evaluator
# ──────────────────────────────────────────────

class ReviewEvaluation(TypedDict):
    """코드 리뷰 평가 결과."""
    passed: bool
    feedback: str


evaluator = evaluator_model.with_structured_output(ReviewEvaluation, strict=True)

EVALUATOR_INSTRUCTIONS = (
    "당신은 코드 리뷰 품질 평가자입니다. 리뷰를 직접 다시 쓰지 마세요.\n\n"
    "아래 세 가지 기준을 모두 충족하면 passed=true를 반환하세요:\n"
    "1. 코드에 근거한 구체적인 문제점이 하나 이상 제시되어 있는가\n"
    "2. 실행 가능한 개선 제안(코드 예시 또는 구체적 설명)이 포함되어 있는가\n"
    "3. 근거 없이 지어낸 정보(존재하지 않는 함수·클래스 언급 등)가 없는가\n\n"
    "⚠️ 중요: '더 좋아질 수 있다'는 이유만으로 passed=false를 반환하지 마세요.\n"
    "리팩토링 제안 부재, 테스트 코드 미언급 등 개선 여지가 있어도 위 세 기준을 충족하면 passed=true입니다.\n"
    "passed=false는 오직 위 기준 중 하나라도 명확히 위반된 경우에만 반환하세요.\n"
    "문제가 없으면 passed=true와 빈 feedback('')을 반환하세요."
)


def evaluate_review_with_llm(answer: str, target_file: str) -> tuple[bool, str]:
    """LLM으로 리뷰의 품질(구체성, 근거, 가독성)을 평가합니다."""
    result = evaluator.invoke([
        SystemMessage(content=EVALUATOR_INSTRUCTIONS),
        HumanMessage(content=(
            f"리뷰 대상 파일: {target_file}\n\n"
            f"평가할 코드 리뷰:\n{answer}"
        )),
    ])
    return result["passed"], result["feedback"].strip()


def make_feedback_message(rule_issues: list[str], llm_passed: bool, llm_feedback: str) -> str:
    """검증 문제와 LLM 피드백을 다음 실행을 위한 지시 메시지로 변환합니다."""
    all_issues = list(rule_issues)
    if not llm_passed and llm_feedback:
        all_issues.append(f"LLM 평가: {llm_feedback}")
    if not all_issues:
        return ""
    return "다음 문제를 고쳐 리뷰를 다시 작성하세요:\n- " + "\n- ".join(all_issues)


# ──────────────────────────────────────────────
# 6. LangGraph: Tool Loop + Verification Loop
# ──────────────────────────────────────────────

class ReviewState(TypedDict):
    """코드 리뷰 Graph의 전체 상태."""
    messages: Annotated[list[AnyMessage], add_messages]
    answer: str
    passed: bool
    feedback: str
    issues: list[str]
    iterations: int
    target_file: str   # 리뷰 대상 파일 경로 (검증에 활용)


AgentNode = Callable[[ReviewState], dict | Awaitable[dict]]


def route_agent(state: ReviewState) -> str:
    """마지막 AI 메시지에 Tool Call이 있으면 tools로, 없으면 verify로 이동합니다."""
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "verify"


def route_verify(state: ReviewState) -> str:
    """검증 통과 또는 MAX_RUNS 도달 시 end, 아니면 feedback으로 이동합니다."""
    return "end" if state["passed"] or state["iterations"] >= MAX_RUNS else "feedback"


def verify_node(state: ReviewState) -> dict:
    """규칙 검증 + LLM 평가를 결합해 통과 여부와 피드백을 State에 반환합니다."""
    answer = str(state["messages"][-1].content)
    rule_issues = verify_review(answer, state["messages"])
    llm_passed, llm_feedback = evaluate_review_with_llm(answer, state.get("target_file", ""))

    issues = list(rule_issues)
    if not llm_passed:
        issues.append(f"LLM 평가: {llm_feedback or '리뷰를 다시 검토하세요.'}")
    feedback = make_feedback_message(rule_issues, llm_passed, llm_feedback)

    print(
        f"\n[VERIFY] {'PASS ✅' if not issues else 'FAIL ❌'}"
        f" | rules: {len(rule_issues)} | LLM: {'pass' if llm_passed else 'fail'}"
    )
    if feedback:
        print(f"[FEEDBACK] {feedback[:300]}")

    return {
        "answer": answer,
        "passed": not issues,
        "feedback": feedback,
        "issues": issues,
    }


def feedback_node(state: ReviewState) -> dict:
    """피드백 메시지를 대화에 추가하고 시도 횟수를 늘립니다."""
    return {
        "messages": [HumanMessage(content=state["feedback"])],
        "iterations": state["iterations"] + 1,
    }


def build_review_graph(agent_node: AgentNode, tools: list[BaseTool]) -> CompiledStateGraph:
    """
    코드 리뷰 Graph를 조립합니다.

    노드: agent → (tools | verify) → (feedback → agent | END)

    Harness 적용:
    - recursion_limit은 invoke() 시 config로 전달 (RECURSION_LIMIT 상수)
    - InMemorySaver Checkpointer로 thread별 State 격리
    """
    builder = StateGraph(ReviewState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("verify", verify_node)
    builder.add_node("feedback", feedback_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_agent, {"tools": "tools", "verify": "verify"})
    builder.add_edge("tools", "agent")
    builder.add_conditional_edges("verify", route_verify, {"end": END, "feedback": "feedback"})
    builder.add_edge("feedback", "agent")

    return builder.compile(checkpointer=InMemorySaver())


# 모델에 Tool을 바인딩하고 Agent Node 함수를 정의합니다.
model_with_tools = model.bind_tools(review_tools)


def review_agent_node(state: ReviewState) -> dict:
    """시스템 프롬프트와 현재 메시지를 모델에 전달하고 응답을 State에 추가합니다."""
    response = model_with_tools.invoke(
        [SystemMessage(content=SYSTEM_INSTRUCTIONS), *state["messages"]]
    )
    return {"messages": [response]}


review_graph = build_review_graph(review_agent_node, review_tools)
print(f"[GRAPH NODES] {list(review_graph.get_graph().nodes)}")


# ──────────────────────────────────────────────
# 7. Loop Engineering: 검증 → 피드백 → 재실행
# ──────────────────────────────────────────────

def run_code_review_loop(
    target_file: str,
    extra_context: str = "",
    save_as: str = "",
) -> dict[str, Any]:
    """
    코드 리뷰 Loop Engineering의 진입점.

    흐름:
      Task 구성
        ↓
      Agent + Harness (LangGraph Graph)
        ↓
      Verification (규칙 + LLM Evaluator)
        ↓
      완료 조건 만족? ──Yes──→ End
        ↓ No
      Feedback / 재실행 (MAX_RUNS까지)

    Args:
        target_file: 리뷰할 파일의 프로젝트 루트 기준 상대 경로
        extra_context: 추가 지시사항 (선택)
        save_as: 결과를 저장할 파일명 (비어 있으면 저장 안 함)

    Returns:
        {"answer": str, "passed": bool, "iterations": int, "traces": list[RunTrace]}
    """
    task = (
        f"다음 파일을 코드 리뷰하세요: {target_file}\n\n"
        "반드시 다음 순서로 진행하세요:\n"
        "1. read_code_file로 코드를 읽는다\n"
        "2. analyze_code_metrics로 메트릭을 분석한다\n"
        "3. search_best_practices로 관련 모범 사례를 검색한다\n"
        "4. 분석 결과를 바탕으로 지시사항의 형식에 맞게 리뷰 보고서를 작성한다\n"
    )
    if save_as:
        task += f"5. save_review_report로 '{save_as}' 파일명으로 보고서를 저장한다\n"
    if extra_context:
        task += f"\n추가 지시사항: {extra_context}"

    thread_id = str(uuid.uuid4())
    traces: list[RunTrace] = []
    final_answer = ""
    final_passed = False
    iterations = 0

    # ── Loop: MAX_RUNS 제한, Graph 내부에서 verify·feedback 반복 ──
    # Graph 자체가 검증 루프를 내장하므로, 여기서는 단일 invoke로 전체 루프를 실행합니다.
    # (외부 while 루프 없이 Graph의 Conditional Edge가 반복을 제어합니다)
    print(f"\n[LOOP] 코드 리뷰 시작: {target_file}")
    print(f"[LOOP] 최대 시도: {MAX_RUNS}회 | recursion_limit: {RECURSION_LIMIT}")

    try:
        final_state = review_graph.invoke(
            {
                "messages": [HumanMessage(content=task)],
                "answer": "",
                "passed": False,
                "feedback": "",
                "issues": [],
                "iterations": 1,
                "target_file": target_file,
            },
            config={
                "recursion_limit": RECURSION_LIMIT,
                "configurable": {"thread_id": thread_id},
            },
        )
        final_answer = final_state.get("answer", "")
        final_passed = final_state.get("passed", False)
        iterations = final_state.get("iterations", 1)
        trace = extract_trace(final_state.get("messages", []), run_id=1)
        traces.append(trace)

    except Exception as e:
        print(f"[LOOP] ❌ 루프 실행 실패: {type(e).__name__}: {e}")
        trace = RunTrace(run_id=1)
        trace.log_error(str(e))
        traces.append(trace)

    # ── 최종 결과 출력 ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"[LOOP] 완료 | 통과: {'✅' if final_passed else '❌'} | 시도: {iterations}회")
    print(f"{'='*60}")
    if final_answer:
        print("\n" + "─" * 60)
        print("📋 최종 코드 리뷰 보고서")
        print("─" * 60)
        print(final_answer[:2000])
        if len(final_answer) > 2000:
            print(f"\n... (이하 {len(final_answer) - 2000}자 생략, 저장된 파일에서 전체 확인)")

    return {
        "answer": final_answer,
        "passed": final_passed,
        "iterations": iterations,
        "traces": traces,
    }


# ──────────────────────────────────────────────
# 8. 실행 예시 (직접 실행 시)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # 기본 예시: 베이스라인의 wikipedia_search 도구를 리뷰
    TARGET = "tools/wikipedia_search.py"
    SAVE_AS = "review_wikipedia_search.md"

    result = run_code_review_loop(
        target_file=TARGET,
        extra_context="특히 오류 처리와 API 타임아웃 패턴에 주목하세요.",
        save_as=SAVE_AS,
    )

    print(f"\n[DONE] passed={result['passed']} | iterations={result['iterations']}")
    if result["passed"]:
        print(f"[DONE] 리뷰 보고서: assignment/Changyub/reviews/{SAVE_AS}")
