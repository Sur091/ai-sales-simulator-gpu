"""Create a small train/valid split for MLX-LM LoRA fine-tuning.

Reads data/train_chat.jsonl (chat-format conversations produced by formatData.py),
filters out conversations that are too long for a 16 GB M2, shuffles, and writes:
  mlx_data/train.jsonl
  mlx_data/valid.jsonl
"""

import json
import random
from pathlib import Path

SRC = Path("data/train_chat.jsonl")
OUT_DIR = Path("mlx_data")
OUT_DIR.mkdir(exist_ok=True)

TOTAL = 2000
VALID_FRAC = 0.1
MAX_CHARS = 6000  # ~1.5k tokens, keeps training fast and memory-safe
SEED = 42


def approx_chars(example: dict) -> int:
    return sum(len(m.get("content", "")) for m in example["messages"])


def normalize_for_gemma(msgs: list[dict]) -> list[dict] | None:
    """Gemma 3 chat template has no `system` role and requires strict
    user/assistant alternation. Fold system into the first user turn and
    merge consecutive same-role turns."""
    if not msgs:
        return None

    system_text = ""
    rest = []
    for m in msgs:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "system":
            system_text = content
        elif role in ("user", "assistant") and content:
            rest.append({"role": role, "content": content})

    # drop leading assistant turns so conversation starts with user
    while rest and rest[0]["role"] == "assistant":
        rest.pop(0)
    if not rest:
        return None

    # merge consecutive same-role turns
    merged = [rest[0]]
    for m in rest[1:]:
        if m["role"] == merged[-1]["role"]:
            merged[-1]["content"] = merged[-1]["content"] + "\n\n" + m["content"]
        else:
            merged.append(m)

    # fold system prompt into the first user message (Gemma convention)
    if system_text:
        merged[0]["content"] = f"{system_text}\n\n{merged[0]['content']}"

    # need at least one user + one assistant turn to learn anything
    if len(merged) < 2 or not any(m["role"] == "assistant" for m in merged):
        return None

    return merged


def main():
    kept = []
    with SRC.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            msgs = normalize_for_gemma(ex.get("messages", []))
            if msgs is None:
                continue

            if approx_chars({"messages": msgs}) > MAX_CHARS:
                continue

            kept.append({"messages": msgs})

    print(f"eligible conversations: {len(kept)}")

    random.seed(SEED)
    random.shuffle(kept)
    sample = kept[:TOTAL]

    n_valid = int(len(sample) * VALID_FRAC)
    valid = sample[:n_valid]
    train = sample[n_valid:]

    def write(path: Path, rows):
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write(OUT_DIR / "train.jsonl", train)
    write(OUT_DIR / "valid.jsonl", valid)

    print(f"wrote {len(train)} to {OUT_DIR/'train.jsonl'}")
    print(f"wrote {len(valid)} to {OUT_DIR/'valid.jsonl'}")


if __name__ == "__main__":
    main()
