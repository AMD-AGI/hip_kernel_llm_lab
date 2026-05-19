from __future__ import annotations

import os
import re
import sys
import typing as T


def _normalize_reasoning_text(text: T.Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _split_reasoning_sentences(text: str) -> list[str]:
    normalized = _normalize_reasoning_text(text)
    if not normalized:
        return []
    sentences = [
        sentence.strip(" -")
        for sentence in re.split(r"(?<=[.!?;:])\s+", normalized)
        if sentence.strip()
    ]
    return sentences or [normalized]


def _trim_sentence(text: str, max_chars: int = 220) -> str:
    text = _normalize_reasoning_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _is_low_signal_opening_sentence(sentence: str) -> bool:
    lower = sentence.lower().strip()
    low_signal_phrases = (
        "okay",
        "let's see",
        "i need to optimize",
        "i need to analyze",
        "let me analyze",
        "the user wants",
        "the user provided",
        "i have the code",
        "i should optimize",
        "first, i need to",
    )
    return any(lower.startswith(phrase) for phrase in low_signal_phrases)


def _opening_sentence_score(sentence: str) -> int:
    lower = sentence.lower()
    keywords = (
        "bottleneck",
        "mi300",
        "kernel",
        "memory",
        "load",
        "store",
        "access",
        "coalesc",
        "shared memory",
        "lds",
        "warp",
        "wavefront",
        "thread",
        "block",
        "occup",
        "register",
        "launch",
        "reduce",
        "reduction",
        "vector",
        "tile",
        "unroll",
        "latency",
        "bandwidth",
        "throughput",
        "parallel",
        "fused",
        "matmul",
    )
    score = sum(1 for keyword in keywords if keyword in lower)
    if not _is_low_signal_opening_sentence(sentence):
        score += 1
    return score


def _build_opening_summary(sentences: list[str], max_items: int = 3) -> list[str]:
    early_sentences = sentences[:10]
    ranked = sorted(
        (
            (idx, sentence, _opening_sentence_score(sentence))
            for idx, sentence in enumerate(early_sentences)
        ),
        key=lambda item: (-item[2], item[0]),
    )

    selected: list[tuple[int, str]] = []
    for idx, sentence, score in ranked:
        if score <= 0 and _is_low_signal_opening_sentence(sentence):
            continue
        selected.append((idx, sentence))
        if len(selected) >= max_items:
            break

    if not selected:
        for idx, sentence in enumerate(early_sentences):
            if _is_low_signal_opening_sentence(sentence):
                continue
            selected.append((idx, sentence))
            if len(selected) >= max_items:
                break

    if not selected:
        selected = [(idx, sentence) for idx, sentence in enumerate(early_sentences[:max_items])]

    selected.sort(key=lambda item: item[0])
    return [_trim_sentence(sentence) for _, sentence in selected]


def _category_match_count(sentence: str, keywords: tuple[str, ...]) -> int:
    lower = sentence.lower()
    return sum(1 for keyword in keywords if keyword in lower)


def _build_optimization_signals(sentences: list[str], max_items: int = 6) -> list[str]:
    category_keywords = [
        (
            "memory_access",
            (
                "coalesc",
                "contiguous",
                "global memory",
                "memory access",
                "load",
                "store",
                "bandwidth",
                "vector",
            ),
        ),
        (
            "shared_memory",
            ("shared memory", "lds", "bank conflict", "tile", "tiling"),
        ),
        (
            "parallel_mapping",
            (
                "thread",
                "block",
                "warp",
                "wavefront",
                "grid-stride",
                "launch",
                "occup",
                "register",
            ),
        ),
        (
            "loop_compute",
            (
                "unroll",
                "prefetch",
                "reduce",
                "reduction",
                "fuse",
                "fusion",
                "latency",
                "throughput",
            ),
        ),
    ]
    causal_keywords = (
        "because",
        "so that",
        "to reduce",
        "to improve",
        "this avoids",
        "this reduces",
        "this improves",
        "this increases",
    )

    seen_sentences: set[str] = set()
    lines: list[str] = []

    for category, keywords in category_keywords:
        best_sentence = ""
        best_score = 0
        for sentence in sentences:
            sentence_norm = _normalize_reasoning_text(sentence)
            if not sentence_norm:
                continue
            category_hits = _category_match_count(sentence_norm, keywords)
            if category_hits <= 0:
                continue
            score = category_hits + _category_match_count(sentence_norm, causal_keywords)
            if score > best_score:
                best_score = score
                best_sentence = sentence_norm
        if best_sentence and best_sentence not in seen_sentences:
            lines.append(f"[{category}] {_trim_sentence(best_sentence)}")
            seen_sentences.add(best_sentence)
        if len(lines) >= max_items:
            return lines

    if lines:
        return lines[:max_items]

    fallback_keywords = (
        "optimiz",
        "shared memory",
        "coalesc",
        "vector",
        "unroll",
        "tile",
        "warp",
        "wavefront",
        "register",
        "occup",
        "latency",
        "throughput",
        "memory",
        "load",
        "store",
        "thread",
        "block",
        "launch",
        "mi300",
        "lds",
        "prefetch",
        "reduce",
        "fusion",
        "parallel",
    )
    scored = sorted(
        (
            (_category_match_count(sentence, fallback_keywords), idx, sentence)
            for idx, sentence in enumerate(sentences)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    for score, _, sentence in scored:
        sentence_norm = _normalize_reasoning_text(sentence)
        if score <= 0 or not sentence_norm or sentence_norm in seen_sentences:
            continue
        lines.append(f"[signal] {_trim_sentence(sentence_norm)}")
        seen_sentences.add(sentence_norm)
        if len(lines) >= max_items:
            break

    return lines


def _supports_reasoning_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _reasoning_color(text: T.Any, *codes: str) -> str:
    rendered = str(text)
    if not codes or not _supports_reasoning_color():
        return rendered
    return f"\033[{';'.join(codes)}m{rendered}\033[0m"


def _reasoning_bool(label: str, value: bool) -> str:
    tone = ("1", "32") if value else ("1", "31")
    return f"{_reasoning_color(label, '36')}={_reasoning_color(value, *tone)}"


def _reasoning_metric(label: str, value: T.Any, tone: T.Optional[tuple[str, ...]] = None) -> str:
    metric_tone = tone or ("1", "97")
    return f"{_reasoning_color(label, '36')}={_reasoning_color(value, *metric_tone)}"


def _format_reasoning_kv(label: str, parts: list[str]) -> str:
    return f"  {_reasoning_color(f'{label:<18}', '1', '94')}: {' '.join(parts)}"


def _style_signal_line(line: str) -> str:
    match = re.match(r"^\[([^\]]+)\]\s*(.*)$", line)
    if not match:
        return _reasoning_color(line, "37")

    tag, body = match.groups()
    tag_colors = {
        "memory_access": ("1", "96"),
        "shared_memory": ("1", "95"),
        "parallel_mapping": ("1", "94"),
        "loop_compute": ("1", "92"),
        "signal": ("1", "93"),
    }
    tag_tone = tag_colors.get(tag, ("1", "93"))
    return f"{_reasoning_color(f'[{tag}]', *tag_tone)} {_reasoning_color(body, '37')}"


def _format_reasoning_bullets(
    lines: list[str],
    empty_text: str = "<none>",
    signal_mode: bool = False,
) -> str:
    if not lines:
        return f"    {_reasoning_color('-', '90')} {_reasoning_color(empty_text, '2', '37')}"

    formatted_lines = []
    for line in lines:
        content = _style_signal_line(line) if signal_mode else _reasoning_color(line, "37")
        formatted_lines.append(f"    {_reasoning_color('-', '90')} {content}")
    return "\n".join(formatted_lines)


def summarize_think_blocks(response: T.Any) -> dict[str, T.Any]:
    """提取 `<think>...</think>` 调试信息，并结构化展示 CoT 内容。"""
    text = "" if response is None else str(response)
    xml_open_count = len(re.findall(r"<think>", text, flags=re.IGNORECASE))
    xml_close_count = len(re.findall(r"</think>", text, flags=re.IGNORECASE))

    blocks = []
    for block_idx, match in enumerate(
        re.finditer(r"<think>([\s\S]*?)</think>", text, flags=re.IGNORECASE)
    ):
        think_text = (match.group(1) or "").strip()
        sentences = _split_reasoning_sentences(think_text)
        blocks.append(
            {
                "style": "xml_think",
                "block_idx": block_idx,
                "has_cot": bool(think_text),
                "chars": len(think_text),
                "opening_summary": _build_opening_summary(sentences),
                "optimization_signals": _build_optimization_signals(sentences),
            }
        )

    return {
        "xml_open_count": xml_open_count,
        "xml_close_count": xml_close_count,
        "balanced_blocks": len(blocks),
        "has_any_cot": any(block["has_cot"] for block in blocks),
        "blocks": blocks,
    }


def format_think_inspection_log(index: int, think_summary: dict[str, T.Any], extracted_kernel: str) -> str:
    return "\n".join(
        [
            f"{_reasoning_color('[REWARD INFO]', '1', '97')} {_reasoning_color('kernel-agent-react think inspection', '1', '95')}",
            _format_reasoning_kv(
                "index",
                [_reasoning_color(index, "1", "97")],
            ),
            _format_reasoning_kv(
                "think_tokens",
                [
                    _reasoning_metric("open", think_summary["xml_open_count"]),
                    _reasoning_metric("close", think_summary["xml_close_count"]),
                    _reasoning_metric("balanced_blocks", think_summary["balanced_blocks"], ("1", "93")),
                ],
            ),
            _format_reasoning_kv(
                "parsed_result",
                [
                    _reasoning_bool("has_any_cot", think_summary["has_any_cot"]),
                    _reasoning_bool("extracted_kernel", bool(extracted_kernel)),
                    _reasoning_metric("extracted_len", len(extracted_kernel), ("1", "92")),
                ],
            ),
        ]
    )


def format_think_block_log(index: int, block: dict[str, T.Any]) -> str:
    return "\n".join(
        [
            f"{_reasoning_color('[REWARD INFO]', '1', '97')} {_reasoning_color('kernel-agent-react think block', '1', '93')}",
            _format_reasoning_kv(
                "meta",
                [
                    _reasoning_metric("index", index),
                    _reasoning_metric("block", block["block_idx"], ("1", "93")),
                    _reasoning_metric("style", block["style"], ("1", "95")),
                ],
            ),
            _format_reasoning_kv(
                "coverage",
                [
                    _reasoning_bool("has_cot", block["has_cot"]),
                    _reasoning_metric("chars", block["chars"], ("1", "92")),
                ],
            ),
            f"  {_reasoning_color('opening_summary', '1', '33')}:",
            _format_reasoning_bullets(block["opening_summary"]),
            f"  {_reasoning_color('optimization_signals', '1', '33')}:",
            _format_reasoning_bullets(block["optimization_signals"], signal_mode=True),
        ]
    )
