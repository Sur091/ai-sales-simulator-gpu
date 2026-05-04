import json
from pathlib import Path

input_path = Path("data/train.jsonl")
output_path = Path("data/train_chat.jsonl")

SPEAKER_TO_ROLE = {
    "sales_rep": "user",
    "salesperson": "user",
    "rep": "user",
    "seller": "user",
    "agent": "user",
    "customer": "assistant",
    "buyer": "assistant",
    "prospect": "assistant",
}


def build_persona(scenario_obj: dict) -> str:
    cp = scenario_obj.get("customer_persona", {}) if isinstance(scenario_obj, dict) else {}
    name = cp.get("name", "a customer")
    role = cp.get("role", "decision-maker")
    company = cp.get("company", "their company")
    industry = cp.get("industry", "their industry")
    pains = "; ".join(cp.get("pain_points", [])[:3])
    needs = "; ".join(cp.get("needs", [])[:3])
    return (
        f"You are {name}, a {role} at {company} in {industry}. "
        f"Respond realistically as this customer in a sales conversation. "
        f"Pain points: {pains}. Needs: {needs}."
    )


kept = 0
skipped = 0

with input_path.open("r", encoding="utf-8") as fin, \
        output_path.open("w", encoding="utf-8") as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            ex = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue

        raw_conv = ex.get("conversation")
        raw_scenario = ex.get("scenario")
        if not raw_conv:
            skipped += 1
            continue

        try:
            turns = json.loads(raw_conv) if isinstance(raw_conv, str) else raw_conv
            scenario_obj = (
                json.loads(raw_scenario) if isinstance(raw_scenario, str) else (raw_scenario or {})
            )
        except json.JSONDecodeError:
            skipped += 1
            continue

        messages = [{"role": "system", "content": build_persona(scenario_obj)}]
        valid = True
        for t in turns:
            speaker = (t.get("speaker") or "").lower().strip()
            content = (t.get("message") or "").strip()
            role = SPEAKER_TO_ROLE.get(speaker)
            if role is None or not content:
                valid = False
                break
            messages.append({"role": role, "content": content})

        if not valid or len(messages) < 3:
            skipped += 1
            continue

        fout.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        kept += 1

        if (kept + skipped) % 10000 == 0:
            print(f"processed={kept + skipped}  kept={kept}  skipped={skipped}")

print(f"Done. kept={kept}  skipped={skipped}  output={output_path}")
