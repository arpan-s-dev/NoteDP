"""Word-level tokenizer for the internship demo.

The paper uses ClinicalBERT WordPiece (§V). That checkpoint is not bundled.
This tokenizer exists so generated tokens can be decoded back to readable
synthetic notes. It is [UNSPECIFIED] relative to the paper.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

TOKEN_RE = re.compile(r"<[^>]+>|[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[.]|,|;|:|/|%")

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


class WordTokenizer:
    def __init__(self, token_to_id: dict[str, int] | None = None) -> None:
        if token_to_id is None:
            self.token_to_id = {tok: i for i, tok in enumerate(SPECIAL)}
        else:
            self.token_to_id = dict(token_to_id)
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    def add(self, token: str) -> None:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def tokenize(self, text: str) -> list[str]:
        return [m.group(0).lower() if not m.group(0).startswith("<") else m.group(0) for m in TOKEN_RE.finditer(text)]

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        pieces = self.tokenize(text)
        ids = [self.token_to_id.get(p, self.unk_id) for p in pieces]
        if add_special:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        words: list[str] = []
        special = {self.pad_id, self.bos_id, self.eos_id}
        for i in ids:
            if skip_special and i in special:
                continue
            tok = self.id_to_token.get(int(i), "<unk>")
            if tok in {".", ",", ";", ":"}:
                if words:
                    words[-1] = words[-1] + tok
                else:
                    words.append(tok)
            else:
                words.append(tok)
        return " ".join(words)

    def to_dict(self) -> dict[str, int]:
        return dict(self.token_to_id)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.token_to_id, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data)

    @classmethod
    def from_texts(cls, texts: list[str]) -> "WordTokenizer":
        tok = cls()
        extras = [
            "patient", "year-old", "years", "old", "male", "female",
            "history", "of", "rare", "genetic", "disorder", "town",
            "occupation", "surgeon", "nurse", "date", "march", "clinic",
            "summary", "the", "a", "and", "with", "for", "on", "in",
        ]
        for text in list(texts) + extras:
            for piece in tok.tokenize(text):
                tok.add(piece)
        return tok
