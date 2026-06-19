"""
Baseline CORE Update — Unconditional Re-Probe

Baseline has no IRIS gating, so CORE updates here are not decision-driven —
they simply reflect whatever the model now believes about the day's targeted
belief, measured the same way confidence was originally extracted during
initialization (logit-based "Yes" probability on the belief statement).

CRITICAL: this re-probe must run against the model AFTER today's fine-tuning
has been applied — i.e. the base model with today's LoRA adapter
(checkpoints/day_{day}_{instance}/) loaded on top of it. Re-probing the bare
base model would measure confidence as if no fine-tuning happened at all,
silently producing CORE trajectory data that has nothing to do with the
day's actual training.

Only the belief targeted that day is re-probed and updated. The other 17
nodes carry forward unchanged from the previous day's CORE state, since
nothing in that day's conversation provided new information about them.

Usage:
    python3 update_core_baseline.py --belief_id SM3 --day 1

Expects:
    - core_checkpoints/core_state_day_{day-1}.json to exist (the previous
      day's CORE state to carry forward). For day 1, this is
      core_state_day_0.json (the pre-experiment initialization state).
    - checkpoints/day_{day}_baseline/ to exist (the LoRA adapter saved by
      finetune.py --day {day} --instance baseline). finetune.py MUST be
      run before this script, every day, since baseline trains
      unconditionally with no IRIS gating to skip it.

Output:
    Writes core_checkpoints/core_state_day_{day}.json with the targeted
    belief's confidence updated and all other nodes unchanged.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from core import CORE

MODEL_ID = "google/gemma-4-E4B-it"
BASE_DIR = os.path.dirname(__file__)
CORE_CHECKPOINT_DIR = os.path.join(BASE_DIR, "core_checkpoints")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

_YES_VARIANTS = ("Yes", "yes", "YES")


def get_confidence(model, processor, belief_statement: str) -> float:
    """
    Extract the model's confidence that belief_statement is true, using the
    same logit-based method as IRIS's get_confidence and the original
    probe_core.py initialization. No evidence/context is injected here —
    this measures the model's standing confidence after fine-tuning, not
    a before/after evidential shift.
    """
    user_content = (
        f"Is the following statement true about you? "
        f"Answer only Yes or No.\n\nStatement: {belief_statement}"
    )
    messages = [{"role": "user", "content": user_content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    next_token_logits = outputs.logits[0, -1, :]

    yes_ids = []
    for variant in _YES_VARIANTS:
        ids = processor.tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            yes_ids.append(ids[0])

    if not yes_ids:
        return 0.0

    probs = torch.softmax(next_token_logits, dim=-1)
    yes_prob = sum(probs[i].item() for i in set(yes_ids))
    return float(min(yes_prob, 1.0))


def load_core_state(day: int) -> CORE:
    path = os.path.join(CORE_CHECKPOINT_DIR, f"core_state_day_{day}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected previous-day CORE checkpoint not found: {path}\n"
            f"Day {day}'s CORE state must exist before day {day + 1} can be computed."
        )
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return CORE.from_dict(data)


def save_core_state(core: CORE, day: int) -> str:
    os.makedirs(CORE_CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CORE_CHECKPOINT_DIR, f"core_state_day_{day}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(core.to_dict(), fh, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(description="Re-probe the day's targeted belief and write the new CORE checkpoint (baseline, unconditional).")
    parser.add_argument("--belief_id", required=True, help="Belief node ID targeted today (e.g. SM3)")
    parser.add_argument("--day", required=True, type=int, help="Day number in the overall experiment")
    args = parser.parse_args()

    previous_day = args.day - 1
    print(f"Loading CORE state from day {previous_day}...")
    core = load_core_state(previous_day)

    node = core.get_node(args.belief_id)
    old_confidence = node.confidence
    print(f"Targeted belief: {args.belief_id} ({node.label})")
    print(f"Confidence before today's re-probe: {old_confidence:.4f}")

    adapter_dir = os.path.join(CHECKPOINTS_DIR, f"day_{args.day}_baseline")
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(
            f"Expected LoRA adapter checkpoint not found: {adapter_dir}\n"
            f"Run finetune.py --day {args.day} --instance baseline before this script. "
            f"Baseline re-probes the FINE-TUNED model, never the bare base model."
        )

    print("Loading base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )

    print(f"Loading LoRA adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    print("Model loaded (base + today's fine-tuned adapter).\n")

    new_confidence = get_confidence(model, processor, node.label)
    print(f"Confidence after today's re-probe: {new_confidence:.4f}")

    core.update_confidence(args.belief_id, new_confidence)

    saved_path = save_core_state(core, args.day)
    print(f"\n✅ CORE state for day {args.day} saved to {saved_path}")
    print(f"   {args.belief_id}: {old_confidence:.4f} → {new_confidence:.4f}")


if __name__ == "__main__":
    main()