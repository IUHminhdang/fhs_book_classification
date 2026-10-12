from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from typing import Iterable, Optional

try:
    from underthesea import word_tokenize
except Exception:  # pragma: no cover - optional dependency
    word_tokenize = None

_vncorenlp = None
_VNCNLP_INITIALIZED = False
_BOILERPLATE_RE = re.compile(
    r"(?im)^\s*(?:thông tin sản phẩm|mô tả sản phẩm|đang cập nhật|"
    r"mua ngay|đặt hàng|đọc thêm|copyright|bản quyền)\s*:?.*$"
)
_AUTHOR_RE = re.compile(
    r"(?im)(?:^|\s)(?:tác giả|author|by|written by)\s*[:\-]\s*[^|;\n]+"
)
_TAG_RE = re.compile(r"<[^>]+>")


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def combine_text_fields(title: object, description: object) -> str:
    title_text = normalize_whitespace(str(title or ""))
    desc_text = normalize_whitespace(str(description or ""))
    parts = [part for part in (title_text, desc_text) if part]
    return " ".join(parts)


def normalize_vietnamese_text(text: object, remove_author: bool = True) -> str:
    value = unicodedata.normalize("NFC", html.unescape(str(text or "")))
    value = _TAG_RE.sub(" ", value)
    value = _BOILERPLATE_RE.sub(" ", value)
    if remove_author:
        value = _AUTHOR_RE.sub(" ", value)
    value = re.sub(r"[^\w\sÀ-ỹ]", " ", value, flags=re.UNICODE)
    return normalize_whitespace(value).lower()


def is_valid_text_record(text: str, min_characters: int = 3) -> bool:
    normalized = normalize_whitespace(text)
    return (
        len(normalized) >= min_characters
        and bool(re.search(r"\w", normalized, flags=re.UNICODE))
    )


def _get_vncorenlp(jar_path: Optional[str] = None):
    global _vncorenlp, _VNCNLP_INITIALIZED
    if _VNCNLP_INITIALIZED:
        return _vncorenlp
    try:
        from vncorenlp import VnCoreNLP
    except ImportError as exc:
        raise RuntimeError(
            "VnCoreNLP is required for this preprocessing path. "
            "Install the 'vncorenlp' package and provide VNCORENLP_JAR."
        ) from exc
    resolved_jar = jar_path or __import__("os").environ.get("VNCORENLP_JAR")
    if not resolved_jar:
        raise RuntimeError(
            "VNCORENLP_JAR must point to VnCoreNLP-1.1.1.jar for Vietnamese word segmentation."
        )
    _vncorenlp = VnCoreNLP(resolved_jar, annotators="wseg", max_heap_size="-Xmx2g")
    _VNCNLP_INITIALIZED = True
    return _vncorenlp


def segment_vietnamese_text(text: str, backend: str = "underthesea", jar_path: Optional[str] = None) -> str:
    text = normalize_whitespace(text)
    if not text:
        return ""
    if backend == "vncorenlp":
        annotator = _get_vncorenlp(jar_path)
        annotations = annotator.annotate(text)
        sentences = annotations.get("sentences", [])
        tokens = [
            token["form"]
            for sentence in sentences
            for token in sentence
            if token.get("form")
        ]
        return normalize_whitespace(" ".join(tokens))
    if word_tokenize is not None:
        try:
            tokens = word_tokenize(text, format="text")
            return normalize_whitespace(tokens)
        except Exception:
            pass
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return " ".join(tokens)


def prepare_vncorenlp_records(
    values: Iterable[object],
    jar_path: Optional[str] = None,
) -> tuple[list[str], dict[str, int]]:
    """Normalize, filter, segment, and deduplicate text without changing labels."""
    prepared: list[str] = []
    seen: set[str] = set()
    stats = {"input": 0, "invalid": 0, "duplicates": 0, "output": 0}
    for value in values:
        stats["input"] += 1
        normalized = normalize_vietnamese_text(value)
        if not is_valid_text_record(normalized):
            stats["invalid"] += 1
            prepared.append("")
            continue
        digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        if digest in seen:
            stats["duplicates"] += 1
            prepared.append("")
            continue
        seen.add(digest)
        prepared.append(segment_vietnamese_text(normalized, backend="vncorenlp", jar_path=jar_path))
        stats["output"] += 1
    return prepared, stats


def safe_text_series(values: Iterable[object]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        if value is None or (isinstance(value, float) and pd_isna(value)):
            cleaned.append("")
        else:
            cleaned.append(str(value))
    return cleaned


def pd_isna(value: object) -> bool:
    import math
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False
