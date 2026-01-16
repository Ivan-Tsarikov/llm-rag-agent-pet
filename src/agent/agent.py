# src/agent/agent.py

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.core.logging import get_logger
from src.agent.tools import ToolRegistry, ToolError

log = get_logger(__name__)


@dataclass
class AgentStep:
    step: int
    llm_raw: str
    action: str
    tool: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None


class AgentError(RuntimeError):
    pass


# -------------------------
# Helpers: parsing & routing
# -------------------------
def _extract_first_json_object(text: str) -> Dict[str, Any]:
    """
    Пытаемся вытащить ПЕРВЫЙ валидный JSON-объект (dict) из произвольного текста.
    Важно: это "best-effort" — в рантайме мы НЕ должны падать, если JSON нет.
    """
    s = (text or "").strip()
    if not s:
        raise AgentError("LLM returned empty output.")

    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(s) if ch == "{"]

    for i in starts:
        try:
            obj, _end = decoder.raw_decode(s[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    preview = s[:400].replace("\n", "\\n")
    raise AgentError(f"LLM did not return a valid JSON object. output_preview={preview}")


_MATH_HINT_RE = re.compile(r"(\d|\bпроцент\b|%|\+|\-|\*|/|сколько|посчитай|вычисли)", re.IGNORECASE)


def _looks_like_math(question: str) -> bool:
    q = (question or "").strip()
    if len(q) < 2:
        return False
    return bool(_MATH_HINT_RE.search(q))


def _extract_math_expr(question: str) -> str:
    """
    Преобразуем типичные русские формулировки в арифметическое выражение.
    Ключевой кейс: "3.5% от 12000" -> "(12000*3.5/100)".
    """
    q = (question or "").strip()

    # уберём "посчитай", "вычисли", "сколько будет" и т.п.
    q = re.sub(r"^\s*(посчитай|вычисли|сколько будет|сколько)\s*[:\-]?\s*", "", q, flags=re.IGNORECASE)

    # нормализуем десятичные запятые
    q_norm = q.replace(",", ".")

    # 3.5% от 12000  /  3.5 процентов от 12000  /  3.5 процента из 12000
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:%|процент(?:а|ов)?)\s*(?:от|из)\s*(\d+(?:\.\d+)?)",
        q_norm,
        flags=re.IGNORECASE,
    )
    if m:
        p = m.group(1)
        base = m.group(2)
        return f"({base}*{p}/100)"

    # если % прилеплен к числу и дальше есть "от/из" (вариант с пробелами/пунктуацией)
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*%\s*(?:от|из)\s*(\d+(?:\.\d+)?)",
        q_norm,
        flags=re.IGNORECASE,
    )
    if m:
        p = m.group(1)
        base = m.group(2)
        return f"({base}*{p}/100)"

    # fallback: вытащим допустимые символы
    allowed = re.findall(r"[0-9\.\+\-\*\/\(\)\s]+", q_norm)
    expr = "".join(allowed).strip()
    return expr if expr else q_norm


def _compact_hits(tool_result: Dict[str, Any], max_hits: int = 5, preview_chars: int = 400) -> Dict[str, Any]:
    """
    Чтобы не раздувать prompt, кладём в память компактный результат:
    - путь/чанк/скор/превью
    """
    hits = tool_result.get("hits")
    if not isinstance(hits, list):
        return tool_result

    compact = []
    for h in hits[:max_hits]:
        if not isinstance(h, dict):
            continue
        txt = h.get("text")
        compact.append(
            {
                "source_path": h.get("source_path"),
                "chunk_id": h.get("chunk_id"),
                "score": h.get("score"),
                "text_preview": (txt[:preview_chars] if isinstance(txt, str) else None),
            }
        )
    return {"hits": compact}


def _build_system_prompt_final_only() -> str:
    # ВАЖНО: не требуем JSON. Цель — стабильность.
    return (
        "Ты помощник службы поддержки маркетплейса.\n"
        "Отвечай на русском.\n"
        "Используй ТОЛЬКО данные из observation.\n"
        "Ничего не додумывай и не добавляй.\n"
        "Ответ должен быть коротким: максимум 6 пунктов или до 80 слов.\n"
        "Верни только финальный ответ ТЕКСТОМ.\n"
        "Не возвращай JSON и не используй фигурные скобки."
    )


def _build_user_prompt_final(question: str, memory: List[Tuple[str, str]]) -> str:
    parts = [f"Вопрос:\n{question}\n"]
    if memory:
        parts.append("Наблюдения (результаты инструментов):\n")
        for role, content in memory:
            parts.append(f"[{role}]\n{content}\n")
    parts.append("Сгенерируй финальный ответ.")
    return "\n".join(parts)


# --- best-effort extraction for broken JSON-in-string cases
_ANSWER_FIELD_RE = re.compile(r'"answer"\s*:\s*"(.*)"\s*}\s*$', re.DOTALL)


def _best_effort_extract_answer(text: str) -> Optional[str]:
    """
    На случай невалидного JSON (чаще всего из-за неэкранированных кавычек внутри answer).
    Достаём answer "как есть" и слегка разэкраниваем.
    """
    s = (text or "").strip()
    m = _ANSWER_FIELD_RE.search(s)
    if not m:
        return None
    ans = m.group(1)
    ans = ans.replace("\\n", "\n").replace('\\"', '"').strip()
    return ans or None


def _coerce_text_from_obj(obj: Dict[str, Any]) -> Optional[str]:
    """
    Если LLM вернул НЕ final-объект (или внутренний JSON), пытаемся вытащить хоть какой-то текст.
    """
    for k in ("answer", "text", "text_preview", "formatted"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    v = obj.get("value")
    if v is not None:
        return str(v)

    return None


def _fallback_from_search_hits(tool_result: Dict[str, Any]) -> str:
    """
    Фоллбек, если LLM недоступен/сломался: вернём top-1 хит как ответ.
    """
    hits = tool_result.get("hits")
    if not isinstance(hits, list) or not hits:
        return "В базе документов нет релевантного ответа по этому вопросу."

    h0 = hits[0] if isinstance(hits[0], dict) else {}
    src = h0.get("source_path") or "unknown_source"
    txt = h0.get("text") or h0.get("text_preview") or ""
    txt = txt.strip() if isinstance(txt, str) else ""

    if len(txt) > 600:
        txt = txt[:600].rstrip() + "…"

    if txt:
        return f"Источник: {src}\n{txt}"
    return f"Источник: {src}"


# -------------------------
# Main agent
# -------------------------
async def run_agent(
    *,
    llm_generate,  # async callable(prompt:str, timeout_s:float)->str
    question: str,
    tools: ToolRegistry,
    max_steps: int = 4,
    llm_timeout_s: float = 60.0,
    retry_once: bool = True,
) -> Tuple[str, List[AgentStep]]:
    """
    Надёжный MVP-агент:
      - Шаг 1 (авто): если математика -> calc, иначе -> search_docs
      - Шаг 2: LLM делает только финальный ответ по observation (или фоллбек)
    """
    _ = max_steps  # сейчас шагов всегда 2; оставляем параметр для совместимости
    steps: List[AgentStep] = []
    memory: List[Tuple[str, str]] = []

    async def call_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        spec = tools.get(tool_name)
        if not spec:
            raise ToolError(f"Tool '{tool_name}' is not allowed.")
        for _k, v in args.items():
            if isinstance(v, str) and len(v) > 2000:
                raise ToolError("Tool argument too large.")
        return await asyncio.wait_for(spec.handler(args), timeout=spec.timeout_s)

    # --- Step 1: auto tool routing (reliable) ---
    if _looks_like_math(question) and tools.get("calc"):
        expr = _extract_math_expr(question)
        tool_name = "calc"
        tool_args = {"expression": expr}
    else:
        tool_name = "search_docs"
        tool_args = {"query": question, "top_k": 5}

    step1 = AgentStep(step=1, llm_raw="<auto>", action="tool", tool=tool_name, tool_args=tool_args)
    steps.append(step1)

    try:
        tool_result = await call_tool(tool_name, tool_args)
    except Exception as e:
        tool_result = {"error": str(e)}

    step1.tool_result = tool_result

    # Если это calc и он успешен — не зовём LLM вообще (максимальная стабильность)
    if tool_name == "calc" and isinstance(tool_result, dict) and not tool_result.get("error"):
        formatted = tool_result.get("formatted")
        value = tool_result.get("value")
        out = formatted if isinstance(formatted, str) and formatted.strip() else (str(value) if value is not None else "")
        out = out.strip()
        steps.append(AgentStep(step=2, llm_raw="<skipped>", action="final_from_calc"))
        return out if out else "", steps

    # кладём в memory компактно
    obs_payload = {"tool": tool_name, "result": _compact_hits(tool_result)}
    memory.append(("observation", json.dumps(obs_payload, ensure_ascii=False)))

    # --- Step 2: LLM final only (best-effort, never crash demo) ---
    system = _build_system_prompt_final_only()
    user = _build_user_prompt_final(question, memory)
    prompt = f"{system}\n\n{user}"

    llm_raw = ""
    last_err: Optional[Exception] = None

    for attempt in (1, 2) if retry_once else (1,):
        try:
            llm_raw = await llm_generate(prompt, timeout_s=llm_timeout_s)
            llm_raw = (llm_raw or "").strip()
            if llm_raw:
                break
        except Exception as e:
            last_err = e
            log.warning("LLM failed%s: %s", " (retry once)" if attempt == 1 and retry_once else "", repr(e))

    if not llm_raw:
        steps.append(AgentStep(step=2, llm_raw=f"<llm_failed:{repr(last_err)}>", action="final_fallback_no_llm"))
        if tool_name == "search_docs" and isinstance(tool_result, dict):
            return _fallback_from_search_hits(tool_result), steps
        if isinstance(tool_result, dict) and tool_result.get("error"):
            return f"Ошибка инструмента: {tool_result['error']}", steps
        return "Не удалось получить ответ: LLM не вернул результат.", steps

    # Пытаемся разобрать JSON только как "бонус", но никогда не падаем из-за формата.
    try:
        obj = _extract_first_json_object(llm_raw)
        action = str(obj.get("action", "")).strip().lower()
        steps.append(AgentStep(step=2, llm_raw=llm_raw, action=action or "parsed_json"))

        # Если вдруг модель всё же вернула {"action":"final","answer":"..."} — возьмём answer.
        if action == "final":
            answer = obj.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer.strip(), steps

        # Иначе попробуем вытащить текст из объекта (например, твой кейс с text_preview)
        coerced = _coerce_text_from_obj(obj)
        if coerced:
            return coerced, steps

    except AgentError:
        # JSON не парсится — это нормально, возвращаем plain text.
        steps.append(AgentStep(step=2, llm_raw=llm_raw, action="final_plain_text"))

    # Если LLM пытался вернуть JSON с answer, но сломал экранирование — попробуем вытащить answer.
    extracted = _best_effort_extract_answer(llm_raw)
    if extracted:
        return extracted, steps

    # Иначе просто вернём как есть (plain text)
    return llm_raw.strip(), steps
