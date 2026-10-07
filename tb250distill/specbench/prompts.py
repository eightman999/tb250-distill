"""固定プロンプト集（base モデル向けの素の completion 文。中立的な内容のみ）。

ベンチは greedy（temperature 0）・ignore_eos で固定長を生成するので、続きが書きやすい文（コード・JSON・反復列挙）と
自由度の高い文（日本語/英語の散文）を混ぜて、ドラフトの受理率がカテゴリでどう変わるかを見る。
"""
from __future__ import annotations

PROMPTS: list[dict] = [
    {
        "id": "code_py",
        "category": "code_py",
        "prompt": (
            "def merge_sorted(a: list[int], b: list[int]) -> list[int]:\n"
            '    """Merge two ascending lists into one ascending list."""\n'
            "    result: list[int] = []\n"
            "    i = j = 0\n"
        ),
    },
    {
        "id": "code_c",
        "category": "code_c",
        "prompt": (
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n"
            "\n"
            "typedef struct Node {\n"
            "    int value;\n"
            "    struct Node *next;\n"
            "} Node;\n"
            "\n"
            "/* Insert a new element at the head of a singly linked list. */\n"
            "Node *push_front(Node *head, int value) {\n"
        ),
    },
    {
        "id": "json",
        "category": "json",
        "prompt": (
            "[\n"
            '  {"id": 1, "name": "Alice", "age": 30, "city": "Tokyo", "tags": ["admin", "dev"]},\n'
            '  {"id": 2, "name": "Bob", "age": 25, "city": "Osaka", "tags": ["dev"]},\n'
            '  {"id": 3, "name": "Carol", "age": 41, "city": "Nagoya", "tags": ["ops", "dev"]},\n'
            '  {"id": 4,'
        ),
    },
    {
        "id": "list",
        "category": "list",
        "prompt": (
            "1 x 7 = 7\n"
            "2 x 7 = 14\n"
            "3 x 7 = 21\n"
            "4 x 7 = 28\n"
            "5 x 7 = 35\n"
            "6 x 7 = 42\n"
            "7 x 7 ="
        ),
    },
    {
        "id": "ja_explain",
        "category": "ja_explain",
        "prompt": (
            "TCP と UDP の違いについて説明します。\n\n"
            "TCP（Transmission Control Protocol）は、"
        ),
    },
    {
        "id": "ja_prose",
        "category": "ja_prose",
        "prompt": (
            "夏の朝、駅へ向かう道には蝉の声が響いていた。"
            "坂の途中にある古い商店の前で、"
        ),
    },
    {
        "id": "en_explain",
        "category": "en_explain",
        "prompt": (
            "A hash table is a data structure that maps keys to values. "
            "It works by"
        ),
    },
    {
        "id": "en_story",
        "category": "en_story",
        "prompt": (
            "The old lighthouse keeper climbed the stairs one last time. "
            "At the top, he"
        ),
    },
]


def prompt_ids() -> list[str]:
    return [p["id"] for p in PROMPTS]
