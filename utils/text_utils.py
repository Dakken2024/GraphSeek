"""
文本分词工具（启发式相似度/覆盖率计算通用）。

中文按字符二元组（bigram）切分，英文按词切分，
避免 Python re 将连续中文切为单 token 导致的相似度失真。
"""
from __future__ import annotations

import re
from typing import Set


def term_tokens(text: str) -> Set[str]:
    """
    生成用于相似度/覆盖率比较的术语集合。

    - 英文/数字：整词
    - 中文：相邻字符二元组（如 "发布了" -> {发布, 布了}）

    Returns:
        长度 > 1 的术语集合
    """
    text = text.lower()
    tokens: Set[str] = set(re.findall(r"[a-z0-9_]+", text))
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    if cjk:
        tokens.update(cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1))
    return {t for t in tokens if len(t) > 1}


def split_sentences(text: str) -> list:
    """按中英文句末标点/换行切分句子（保留标点，兼容全角/半角）。"""
    parts = re.findall(r"[^。．.!?？；;！\n]+[。．.!?？；;！]?", text)
    return [p.strip() for p in parts if p.strip()]
