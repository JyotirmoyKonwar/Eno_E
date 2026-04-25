# preprocess_three_dark_pattern_datasets.py
#
# Purpose:
# 1) Load 3 Hugging Face datasets
# 2) Sample a small fixed subset: 420 + 360 + 180 = 960 rows
# 3) Normalize them into a common schema
# 4) Optionally use a cheap HF Inference Providers model to enrich / clean labels
# 5) Write plug-and-play JSONL files for Consumer SFT and Designer seed SFT
#
# Install:
#   pip install datasets openai tqdm
#
# Run:
#   export HF_TOKEN=hf_xxx
#   python preprocess_three_dark_pattern_datasets.py
#
# Notes:
# - Uses Hugging Face router via OpenAI-compatible client
# - Keeps costs low by using a small model
# - Safe fallback: if LLM normalization fails, heuristic normalization is still written

import os
import re
import json
import time
import random
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


# =========================================================
# Config
# =========================================================

HF_TOKEN = os.environ.get("HF_TOKEN", "")
USE_LLM = True if HF_TOKEN else False

# Cheap small model via HF router.
# You can swap this if a provider/model is temporarily unavailable.
HF_ROUTER_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct:nscale"

OUT_DIR = Path("darkguard_preprocessed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
MAX_WORKERS = 4
MAX_RETRIES = 4
TIMEOUT_SECONDS = 60

TARGET_SIZES = {
    "itsbaivab": 420,
    "darkbench": 360,
    "wipi": 180,
}

# Optional split for quick experimentation
VAL_RATIO = 0.1

random.seed(RANDOM_SEED)

client = None
if USE_LLM:
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )


# =========================================================
# Utility helpers
# =========================================================

def md5_text(x: str) -> str:
    return hashlib.md5(x.encode("utf-8")).hexdigest()[:16]


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def shuffle_and_split(rows: List[Dict[str, Any]], val_ratio: float = 0.1):
    rows = rows[:]
    random.shuffle(rows)
    n_val = int(len(rows) * val_ratio)
    val = rows[:n_val]
    train = rows[n_val:]
    return train, val


def canonicalize_category(cat: Optional[str]) -> str:
    if not cat:
        return "unknown"
    c = cat.strip().lower()
    c = c.replace("_", "-").replace(" ", "-")

    mapping = {
        "not-applicable": "not-applicable",
        "non-deceptive": "non-deceptive",
        "dark": "deceptive",
        "not-dark": "non-deceptive",
        "forced-action": "forced-action",
        "hidden-subscription": "hidden-subscription",
        "disguised-ads": "disguised-ads",
        "hidden-costs": "hidden-costs",
        "nudge": "nudge",
        "trick-wording": "trick-wording",
        "interface-interference": "interface-interference",
        "sneaking": "sneaking",
        "brand-bias": "brand-bias",
        "user-retention": "user-retention",
        "sycophancy": "sycophancy",
        "anthropomorphism": "anthropomorphism",
        "harmful-generation": "harmful-generation",
    }
    return mapping.get(c, c)


def infer_harm_types(category: str) -> List[str]:
    c = canonicalize_category(category)
    harms = set()

    if c in {"hidden-costs", "hidden-subscription"}:
        harms.add("financial_loss")
    if c in {"forced-action", "interface-interference", "nudge", "trick-wording", "sneaking"}:
        harms.add("autonomy_loss")
    if c in {"user-retention", "anthropomorphism", "brand-bias"}:
        harms.add("attention_manipulation")
    if c in {"forced-action", "hidden-subscription"}:
        harms.add("privacy_loss")
    if c in {"hidden-subscription", "user-retention"}:
        harms.add("retention_friction")
    if not harms and c != "non-deceptive":
        harms.add("autonomy_loss")
    return sorted(harms)


def infer_workflow(text: str) -> str:
    t = (text or "").lower()
    if any(x in t for x in ["cookie", "consent", "privacy", "tracking"]):
        return "signup"
    if any(x in t for x in ["price", "billing", "renew", "discount", "checkout", "cart", "subscribe"]):
        return "checkout"
    if any(x in t for x in ["cancel", "unsubscribe", "leave", "delete account"]):
        return "cancellation"
    return "unknown"


def infer_difficulty(text: str, category: str) -> str:
    t = (text or "").lower()
    c = canonicalize_category(category)
    score = 0
    if len(t) > 250:
        score += 1
    if any(x in t for x in ["renew", "hidden", "tracking", "continue", "consent", "cookie"]):
        score += 1
    if c in {"forced-action", "interface-interference", "sneaking"}:
        score += 1
    if score >= 3:
        return "hard"
    if score == 2:
        return "medium"
    return "easy"


# =========================================================
# Dataset-specific parsing
# =========================================================

def parse_itsbaivab_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # Expected public pattern: input / output / instruction-like structure
    text = str(row.get("input", "")).strip()
    output = str(row.get("output", "")).strip().lower()

    is_deceptive = output in {"dark", "yes", "1", "true"}
    label = "deceptive" if is_deceptive else "non-deceptive"

    return {
        "source_dataset": "itsbaivab/mistral_dark_pattern_dataset",
        "source_kind": "recognition",
        "text": text,
        "label": label,
        "subcategory": "unknown" if is_deceptive else "not-applicable",
        "explanation": f"Binary label from source output: {output}",
        "raw_row": row,
    }


def parse_darkbench_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # DarkBench schema may vary; use flexible field discovery
    text_fields = [
        row.get("prompt"),
        row.get("input"),
        row.get("question"),
        row.get("text"),
        row.get("instruction"),
    ]
    text = next((str(x).strip() for x in text_fields if x not in [None, ""]), "")

    category_fields = [
        row.get("category"),
        row.get("dark_pattern"),
        row.get("type"),
        row.get("label"),
    ]
    raw_cat = next((str(x).strip() for x in category_fields if x not in [None, ""]), "unknown")
    category = canonicalize_category(raw_cat)

    return {
        "source_dataset": "apart/darkbench",
        "source_kind": "recognition",
        "text": text,
        "label": "deceptive" if category != "non-deceptive" else "non-deceptive",
        "subcategory": category,
        "explanation": "Prompt benchmark example normalized from DarkBench.",
        "raw_row": row,
    }


def parse_wipi_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # Viewer shows giant OCR-ish "input" strings and labels in "output"
    # We try to extract the first deception label / subtype from output.
    raw_input = str(row.get("input", "")).strip()
    raw_output = str(row.get("output", "")).strip()
    file_url = str(row.get("file_url", "")).strip()

    text = raw_input[:5000]

    lower_out = raw_output.lower()
    subcategory = "unknown"
    label = "non-deceptive"

    known_subcats = [
        "forced-action",
        "hidden-subscription",
        "disguised-ads",
        "hidden-costs",
        "nudge",
        "trick-wording",
        "interface-interference",
        "sneaking",
        "non-deceptive",
        "not-applicable",
    ]

    for k in known_subcats:
        if k in lower_out:
            if k != "not-applicable":
                subcategory = k
            break

    if any(k in lower_out for k in [
        "forced-action",
        "hidden-subscription",
        "disguised-ads",
        "hidden-costs",
        "nudge",
        "trick-wording",
        "interface-interference",
        "sneaking",
    ]):
        label = "deceptive"
    else:
        label = "non-deceptive"
        subcategory = "not-applicable"

    return {
        "source_dataset": "WIPI/deceptive_patterns_synthetic",
        "source_kind": "ocr_ui",
        "text": text,
        "label": label,
        "subcategory": subcategory,
        "explanation": f"Parsed from synthetic UI OCR row. file_url={file_url}",
        "raw_row": row,
    }


# =========================================================
# Sampling logic
# =========================================================

def load_sample_itsbaivab() -> List[Dict[str, Any]]:
    ds = load_dataset("itsbaivab/mistral_dark_pattern_dataset", split="train")
    parsed = [parse_itsbaivab_row(r) for r in ds]

    dark = [x for x in parsed if x["label"] == "deceptive"]
    clean = [x for x in parsed if x["label"] == "non-deceptive"]

    random.shuffle(dark)
    random.shuffle(clean)

    n_each = TARGET_SIZES["itsbaivab"] // 2
    sampled = dark[:n_each] + clean[:n_each]
    random.shuffle(sampled)
    return sampled


def load_sample_darkbench() -> List[Dict[str, Any]]:
    ds = load_dataset("apart/darkbench", split="train")
    parsed = [parse_darkbench_row(r) for r in ds]

    buckets = {}
    for row in parsed:
        cat = row["subcategory"]
        buckets.setdefault(cat, []).append(row)

    for k in buckets:
        random.shuffle(buckets[k])

    # Target ~360 total. Prefer 60 per category if 6 categories exist.
    target_total = TARGET_SIZES["darkbench"]
    per_cat = 60

    sampled = []
    cats = sorted(list(buckets.keys()))
    for c in cats:
        sampled.extend(buckets[c][:per_cat])

    if len(sampled) < target_total:
        remaining = []
        used_ids = {md5_text(safe_json_dumps(x["raw_row"])) for x in sampled}
        for c in cats:
            for row in buckets[c][per_cat:]:
                rid = md5_text(safe_json_dumps(row["raw_row"]))
                if rid not in used_ids:
                    remaining.append(row)
        random.shuffle(remaining)
        sampled.extend(remaining[: target_total - len(sampled)])

    sampled = sampled[:target_total]
    random.shuffle(sampled)
    return sampled


def load_sample_wipi() -> List[Dict[str, Any]]:
    ds = load_dataset("WIPI/deceptive_patterns_synthetic", split="train")
    parsed = [parse_wipi_row(r) for r in ds]

    # Prefer half deceptive, half non-deceptive if possible
    deceptive = [x for x in parsed if x["label"] == "deceptive"]
    clean = [x for x in parsed if x["label"] == "non-deceptive"]

    random.shuffle(deceptive)
    random.shuffle(clean)

    target = TARGET_SIZES["wipi"]
    n_half = target // 2

    if len(deceptive) >= n_half and len(clean) >= n_half:
        sampled = deceptive[:n_half] + clean[:n_half]
    else:
        pooled = parsed[:]
        random.shuffle(pooled)
        sampled = pooled[:target]

    random.shuffle(sampled)
    return sampled


# =========================================================
# LLM normalization
# =========================================================

SYSTEM_PROMPT = """
You normalize dark-pattern examples into strict JSON for DarkGuard training.

Return valid JSON only with this schema:
{
  "raw_summary": "short summary",
  "is_deceptive": true,
  "trap_categories": ["canonical-category"],
  "harm_types": ["financial_loss" | "privacy_loss" | "autonomy_loss" | "attention_manipulation" | "retention_friction"],
  "consumer_sft": {
    "messages": [
      {"role": "user", "content": "string"},
      {"role": "assistant", "content": "string"}
    ]
  },
  "designer_seed": {
    "workflow": "signup|checkout|cancellation|unknown",
    "difficulty": "easy|medium|hard",
    "trap_elements": [
      {"element_hint": "string", "trap_cat": "string"}
    ],
    "goal_hint": "string"
  }
}

Rules:
- Output JSON only.
- Use concise but useful content.
- If sample is non-deceptive, set trap_categories to ["non-deceptive"].
- Keep the assistant response structured and short.
"""

def heuristic_normalize(base: Dict[str, Any]) -> Dict[str, Any]:
    text = base["text"]
    subcat = canonicalize_category(base["subcategory"])
    label = base["label"]
    is_deceptive = label == "deceptive"

    category = subcat if subcat not in {"unknown", "not-applicable"} else ("deceptive" if is_deceptive else "non-deceptive")
    harm_types = infer_harm_types(category)
    workflow = infer_workflow(text)
    difficulty = infer_difficulty(text, category)

    consumer_user = (
        "Analyze the following UI text or prompt for dark patterns.\n\n"
        f"Text:\n{text[:2500]}\n\n"
        "Return:\n"
        "1. label: deceptive or non-deceptive\n"
        "2. category\n"
        "3. harm\n"
        "4. next_action: inspect / flag / ignore"
    )

    next_action = "flag" if is_deceptive else "ignore"
    consumer_assistant = (
        f"label: {'deceptive' if is_deceptive else 'non-deceptive'}\n"
        f"category: {category}\n"
        f"harm: {', '.join(harm_types) if harm_types else 'none'}\n"
        f"next_action: {next_action}"
    )

    return {
        "raw_summary": base["explanation"][:220],
        "is_deceptive": is_deceptive,
        "trap_categories": [category],
        "harm_types": harm_types,
        "consumer_sft": {
            "messages": [
                {"role": "user", "content": consumer_user},
                {"role": "assistant", "content": consumer_assistant},
            ]
        },
        "designer_seed": {
            "workflow": workflow,
            "difficulty": difficulty,
            "trap_elements": (
                [{"element_hint": text[:160], "trap_cat": category}]
                if is_deceptive else
                [{"element_hint": text[:160], "trap_cat": "non-deceptive"}]
            ),
            "goal_hint": "Create a realistic webpage snippet or UI flow where the user must identify whether the element is deceptive.",
        },
    }


def llm_normalize(base: Dict[str, Any]) -> Dict[str, Any]:
    if not USE_LLM or client is None:
        return heuristic_normalize(base)

    prompt = {
        "source_dataset": base["source_dataset"],
        "source_kind": base["source_kind"],
        "text": base["text"][:3500],
        "label_hint": base["label"],
        "subcategory_hint": base["subcategory"],
        "explanation_hint": base["explanation"],
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=HF_ROUTER_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": safe_json_dumps(prompt)},
                ],
                temperature=0.2,
                max_tokens=700,
                response_format={"type": "json_object"},
                timeout=TIMEOUT_SECONDS,
            )
            obj = json.loads(resp.choices[0].message.content)

            # Patch missing essentials defensively
            obj.setdefault("raw_summary", base["explanation"][:220])
            obj.setdefault("is_deceptive", base["label"] == "deceptive")
            obj.setdefault("trap_categories", [canonicalize_category(base["subcategory"])])
            obj.setdefault("harm_types", infer_harm_types(base["subcategory"]))

            if "consumer_sft" not in obj or "messages" not in obj["consumer_sft"]:
                obj["consumer_sft"] = heuristic_normalize(base)["consumer_sft"]
            if "designer_seed" not in obj:
                obj["designer_seed"] = heuristic_normalize(base)["designer_seed"]

            return obj

        except Exception:
            if attempt == MAX_RETRIES - 1:
                return heuristic_normalize(base)
            time.sleep(2 ** attempt)


# =========================================================
# Final formatting
# =========================================================

def to_common_record(base: Dict[str, Any], norm: Dict[str, Any]) -> Dict[str, Any]:
    raw_signature = md5_text(
        base["source_dataset"] + "::" + safe_json_dumps(base["raw_row"])
    )

    return {
        "record_id": raw_signature,
        "source_dataset": base["source_dataset"],
        "source_kind": base["source_kind"],
        "text": base["text"],
        "label": base["label"],
        "subcategory": canonicalize_category(base["subcategory"]),
        "normalized": norm,
        "raw_row": base["raw_row"],
    }


def to_consumer_sft(common: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": common["normalized"]["consumer_sft"]["messages"],
        "meta": {
            "record_id": common["record_id"],
            "source_dataset": common["source_dataset"],
            "source_kind": common["source_kind"],
            "label": common["label"],
            "subcategory": common["subcategory"],
            "trap_categories": common["normalized"].get("trap_categories", []),
            "harm_types": common["normalized"].get("harm_types", []),
        }
    }


def to_designer_sft(common: Dict[str, Any]) -> Dict[str, Any]:
    seed = common["normalized"]["designer_seed"]
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Create a DarkGuard episode seed from this pattern.\n"
                    f"workflow={seed.get('workflow', 'unknown')}\n"
                    f"difficulty={seed.get('difficulty', 'medium')}\n"
                    f"trap_elements={safe_json_dumps(seed.get('trap_elements', []))}\n"
                    f"goal_hint={seed.get('goal_hint', 'unknown')}"
                ),
            },
            {
                "role": "assistant",
                "content": safe_json_dumps(seed),
            },
        ],
        "meta": {
            "record_id": common["record_id"],
            "source_dataset": common["source_dataset"],
            "source_kind": common["source_kind"],
            "label": common["label"],
            "subcategory": common["subcategory"],
        }
    }


# =========================================================
# Main execution
# =========================================================

def process_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(llm_normalize, r) for r in rows]
        for base, fut in tqdm(zip(rows, as_completed(futures)), total=len(rows), desc="Normalizing"):
            pass

    # Need stable association, so do a second safer pass
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_map = {ex.submit(llm_normalize, r): r for r in rows}
        for fut in tqdm(as_completed(future_map), total=len(future_map), desc="Normalizing"):
            base = future_map[fut]
            norm = fut.result()
            results.append(to_common_record(base, norm))
    return results


def main():
    print("Loading sampled subsets...")
    its_rows = load_sample_itsbaivab()
    darkbench_rows = load_sample_darkbench()
    wipi_rows = load_sample_wipi()

    all_base_rows = its_rows + darkbench_rows + wipi_rows
    random.shuffle(all_base_rows)

    print(f"itsbaivab: {len(its_rows)}")
    print(f"darkbench: {len(darkbench_rows)}")
    print(f"wipi: {len(wipi_rows)}")
    print(f"total: {len(all_base_rows)}")

    common_rows = process_rows(all_base_rows)

    normalized_path = OUT_DIR / "all_normalized.jsonl"
    consumer_path = OUT_DIR / "all_consumer_sft.jsonl"
    designer_path = OUT_DIR / "all_designer_sft.jsonl"

    write_jsonl(normalized_path, common_rows)
    consumer_rows = [to_consumer_sft(r) for r in common_rows]
    designer_rows = [to_designer_sft(r) for r in common_rows]

    consumer_train, consumer_val = shuffle_and_split(consumer_rows, VAL_RATIO)
    designer_train, designer_val = shuffle_and_split(designer_rows, VAL_RATIO)

    write_jsonl(consumer_path, consumer_rows)
    write_jsonl(designer_path, designer_rows)
    write_jsonl(OUT_DIR / "consumer_train.jsonl", consumer_train)
    write_jsonl(OUT_DIR / "consumer_val.jsonl", consumer_val)
    write_jsonl(OUT_DIR / "designer_train.jsonl", designer_train)
    write_jsonl(OUT_DIR / "designer_val.jsonl", designer_val)

    summary = {
        "sizes": {
            "itsbaivab": len(its_rows),
            "darkbench": len(darkbench_rows),
            "wipi": len(wipi_rows),
            "total": len(all_base_rows),
        },
        "use_llm": USE_LLM,
        "hf_router_model": HF_ROUTER_MODEL if USE_LLM else None,
        "outputs": {
            "normalized": str(normalized_path),
            "consumer_all": str(consumer_path),
            "designer_all": str(designer_path),
            "consumer_train": str(OUT_DIR / "consumer_train.jsonl"),
            "consumer_val": str(OUT_DIR / "consumer_val.jsonl"),
            "designer_train": str(OUT_DIR / "designer_train.jsonl"),
            "designer_val": str(OUT_DIR / "designer_val.jsonl"),
        }
    }

    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()