
#!/usr/bin/env python3
"""
CLI grader for Agentic AI Assignment 2

Usage example:
  python msds24037_02_grader.py \
    --images quiz_pages/page1.jpg quiz_pages/page2.jpg \
    --out results.json --logs logs.jsonl \
    --model openai:gpt-4o-mini --temperature 0.0

Responsibilities:
  1) Read 1–4 images (order = pages 1..N)
  2) Downscale + JPEG-encode (max side ≈ 1600 px)
  3) Use a vision model to extract:
     - Part A: MCQ map "1".."8" -> "A|B|C|D" (sanitize to first A–D only)
     - Part B: two sequences (A..H) for B1/B2 (trim to 8, valid letters only)
     - Part C: student's prompt spec text
  4) Grade deterministically using the gold key
  5) Write results.json and logs.jsonl (JSON Lines)

Environment variables (loadable via .env):
  OPENAI_API_KEY=<your key>

Notes:
  - Supports --model openai:<name> or local:<name> (stub hook for local models)
"""

import argparse
import base64
import io
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # optional

try:
    from PIL import Image  # type: ignore
except Exception as e:
    print("install Pillow (PIL).", file=sys.stderr)
    raise

# OpenAI SDK is optional; only used for openai:* models
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


# ----------------------------- Logging Utilities -----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class JsonlLogger:
    def __init__(self, logs_path: str):
        self.logs_path = logs_path
        self._fh = open(self.logs_path, "a", encoding="utf-8")

    def log(self, event: str, level: str = "INFO", **fields):
        rec = {
            "ts": utc_now_iso(),
            "level": level,
            "event": event,
        }
        rec.update(fields)
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


# ----------------------------- Image Processing -----------------------------

def load_and_preprocess_images(paths: List[str], max_side: int = 1600) -> Tuple[List[bytes], List[int]]:
    """Load images (any PIL-supported format), handle alpha/CMYK, downscale, JPEG re-encode.
    Returns (jpeg_bytes_list, image_bytes_in_list).
    """
    processed: List[bytes] = []
    input_sizes: List[int] = []

    for path in paths:
        with open(path, "rb") as fh:
            raw = fh.read()
            input_sizes.append(len(raw))
        img = Image.open(io.BytesIO(raw))
        # Normalize color space and transparency
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            # Composite over white background to avoid black halos
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img.convert("RGBA")).convert("RGB")
        elif img.mode == "CMYK":
            img = img.convert("RGB")
        else:
            img = img.convert("RGB")
        w, h = img.size
        scale = 1.0
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)

        out = io.BytesIO()
        img.save(out, format="JPEG", quality=85, optimize=True)
        processed.append(out.getvalue())
    return processed, input_sizes


# ----------------------------- Vision Parsing -----------------------------

MCQ_KEYS = ["1", "2", "3", "4", "5", "6", "7", "8"]
VALID_ABCD = set(list("ABCD"))
VALID_AH = set(list("ABCDEFGH"))


@dataclass
class VisionParse:
    part_a_map: Dict[str, str]
    b1_seq: List[str]
    b2_seq: List[str]
    prompt_text: str


def sanitize_part_a_map(mcq_map: Dict[str, str]) -> Dict[str, str]:
    clean: Dict[str, str] = {}
    for k in MCQ_KEYS:
        v = mcq_map.get(k, "")
        v = (v or "").strip().upper()
        # keep first A–D only
        vv = "".join(ch for ch in v if ch in VALID_ABCD)
        clean[k] = vv[0] if vv else ""
    return clean


def sanitize_flow(seq: List[str]) -> List[str]:
    cleaned = [c.strip().upper() for c in seq if c and c.strip().upper() in VALID_AH]
    return cleaned[:8]


def parse_with_openai(
    client: "OpenAI",
    model_name: str,
    jpeg_pages_bytes: List[bytes],
    temperature: float,
) -> Tuple[VisionParse, Dict[str, int]]:
    """Call OpenAI vision model and parse outputs. Returns (VisionParse, api_usage).
    api_usage contains prompt_tokens, completion_tokens, total_tokens if available.
    """
    # Build the prompt requesting structured JSON
    system_msg = (
        "You are a precise grader-extraction assistant. Extract strictly structured JSON."
    )
    user_instructions = (
        "From the following scanned quiz pages, extract: \n"
        "Part A: MCQ answers for questions '1'..'8' as letters A/B/C/D. Return as an object map.\n"
        "Part B: Two sequences for B1 and B2, each exactly 8 letters A..H in order.\n"
        "Part C: The student's prompt specification text (verbatim).\n\n"
        "Return ONLY valid JSON with this schema: {\n"
        "  \"part_a\": {\"1\": \"A|B|C|D\", ... up to \"8\"},\n"
        "  \"part_b\": {\"b1\": [\"A..H\" x8], \"b2\": [\"A..H\" x8]},\n"
        "  \"part_c\": {\"prompt\": \"string\"}\n"
        "} without markdown fencing."
    )

    # Convert each page to data URI image object
    content_blocks = [
        {"type": "input_text", "text": user_instructions}
    ]
    image_bytes_out: List[int] = []
    for b in jpeg_pages_bytes:
        b64 = base64.b64encode(b).decode("utf-8")
        image_bytes_out.append(len(b))
        content_blocks.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{b64}",
        })

    # Use the new Responses API (OpenAI>=1.0)
    usage: Dict[str, int] = {}
    resp = client.responses.create(
        model=model_name,
        temperature=temperature,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_msg}]},
            {"role": "user", "content": content_blocks},
        ],
    )

    # Extract text
    text_out = ""
    # Preferred helper if available
    if hasattr(resp, "output_text") and resp.output_text:
        text_out = resp.output_text
    elif resp.output and hasattr(resp.output, "text"):
        text_out = resp.output.text
    else:
        # Fallback to concatenating content blocks
        try:
            parts = []
            for item in getattr(resp, "output", []) or []:
                t = getattr(item, "content", None)
                if t and isinstance(t, list):
                    for sub in t:
                        if sub.get("type") == "output_text":
                            parts.append(sub.get("text", ""))
            text_out = "".join(parts)
        except Exception:
            text_out = ""

    # Usage (best-effort; keys may vary)
    try:
        u = getattr(resp, "usage", None)
        if u:
            usage = {
                "prompt_tokens": int(getattr(u, "input_tokens", 0) or 0),
                "completion_tokens": int(getattr(u, "output_tokens", 0) or 0),
                "total_tokens": int(getattr(u, "total_tokens", 0) or 0),
            }
    except Exception:
        usage = {}

    # Parse JSON
    parsed: Dict[str, object] = {}
    try:
        # Some models may wrap JSON in markdown; try to strip fences
        txt = text_out.strip()
        if txt.startswith("```"):
            txt = txt.strip('`')
            # if language tag present, attempt to find inner JSON braces
            l = txt.find("{")
            r = txt.rfind("}")
            if l != -1 and r != -1:
                txt = txt[l : r + 1]
        parsed = json.loads(txt)
    except Exception:
        parsed = {}

    part_a_map = sanitize_part_a_map(
        dict(parsed.get("part_a", {})) if isinstance(parsed.get("part_a"), dict) else {}
    )
    pb = parsed.get("part_b", {}) if isinstance(parsed.get("part_b"), dict) else {}
    b1 = sanitize_flow(list(pb.get("b1", [])) if isinstance(pb.get("b1"), list) else [])
    b2 = sanitize_flow(list(pb.get("b2", [])) if isinstance(pb.get("b2"), list) else [])
    pc = parsed.get("part_c", {}) if isinstance(parsed.get("part_c"), dict) else {}
    prompt_text = str(pc.get("prompt", "")) if pc else ""

    return VisionParse(part_a_map, b1, b2, prompt_text), usage | {"image_bytes_out": image_bytes_out}


def parse_with_local(model_name: str, jpeg_pages_bytes: List[bytes], temperature: float) -> Tuple[VisionParse, Dict[str, int]]:
    """Hook for local VLMs (e.g., LLaVA, Qwen2-VL). Not implemented by default.
    Implement your local model call here and return the same structure as parse_with_openai.
    """
    raise NotImplementedError(f"Local model '{model_name}' not implemented. See README for guidance.")


# ----------------------------- Grading -----------------------------

GOLD_PART_A: Dict[str, str] = {
    "1": "C", "2": "B", "3": "A", "4": "B", "5": "A", "6": "A", "7": "C", "8": "D"
}
GOLD_B1: List[str] = list("CEAGBDFH")
GOLD_B2: List[str] = list("DAHBGFEC")


def grade_part_a(student: Dict[str, str]) -> Tuple[float, Dict[str, int]]:
    raw_correct = 0
    for k in MCQ_KEYS:
        if student.get(k, "") == GOLD_PART_A.get(k, ""):
            raw_correct += 1
    raw_max = len(MCQ_KEYS)
    # 4 MCQ scaled by % correct
    score = 4.0 * (raw_correct / float(raw_max))
    return score, {"raw_correct": raw_correct, "raw_max": raw_max}


def positional_match_fraction(student_seq: List[str], gold_seq: List[str]) -> float:
    n = min(len(student_seq), len(gold_seq), 8)
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if student_seq[i] == gold_seq[i])
    return matches / float(8)


def grade_part_b(b1: List[str], b2: List[str]) -> Tuple[float, Dict[str, float]]:
    def match_count(student_seq, gold_seq):
        return sum(1 for i in range(8) if i < len(student_seq) and student_seq[i] == gold_seq[i])

    b1_matches = match_count(b1, GOLD_B1)
    b2_matches = match_count(b2, GOLD_B2)

    b1_match = b1_matches / 8
    b2_match = b2_matches / 8
    score = 4.0 * ((b1_match + b2_match) / 2.0)

    return score, {"b1": b1_match, "b2": b2_match, "b1_matches": b1_matches, "b2_matches": b2_matches}


def grade_part_c(prompt_text: str, deterministic_award: int = 2) -> float:
    """Deterministic prompt grading. By default: 2 if non-empty text, else 0.
    Adjust deterministic_award via CLI if needed.
    """
    if prompt_text and prompt_text.strip():
        return float(deterministic_award)
    return 0.0


# ----------------------------- CLI -----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agentic AI Assignment 2 - CLI grader")
    p.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Paths to 1..4 image files (order page1..N). Any common format supported by PIL (jpg, jpeg, png, webp, bmp, tiff).",
    )
    p.add_argument("--out", required=True, help="Path to results.json output")
    p.add_argument("--logs", required=True, help="Path to logs.jsonl output")
    p.add_argument("--model", required=True, help="Model spec, e.g., openai:gpt-4o-mini or local:<name>")
    p.add_argument("--temperature", type=float, default=0.0, help="Model temperature (default 0.0)")
    p.add_argument(
        "--prompt-award",
        type=int,
        default=2,
        help="Deterministic award for non-empty prompt text in Part C (0..2)",
    )
    p.add_argument("--max-side", type=int, default=1600, help="Max side for downscale (default 1600)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    if load_dotenv:
        load_dotenv()  # load .env if present

    args = parse_args(argv)

    # Ensure images 1..4
    if not (1 <= len(args.images) <= 4):
        print("Provide between 1 and 4 image paths in order (page1..N).", file=sys.stderr)
        return 2

    cid = str(uuid.uuid4())
    logger = JsonlLogger(args.logs)
    files = list(args.images)
    model = args.model
    temperature = args.temperature

    # grade.start
    logger.log(
        "grade.start",
        cid=cid,
        pages=len(files),
        files=files,
        model=model,
    )

    try:
        # Load & preprocess
        jpeg_pages, image_bytes_in = load_and_preprocess_images(files, max_side=args.max_side)

        # vision.parse.start
        logger.log("vision.parse.start", cid=cid, pages=len(jpeg_pages))

        # Decide backend
        api_usage: Dict[str, int] = {}
        image_bytes_out = [len(b) for b in jpeg_pages]
        if model.startswith("openai:"):
            if OpenAI is None:
                raise RuntimeError("OpenAI SDK not installed. Add to requirements.txt and install.")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set. Put it in .env or environment.")
            client = OpenAI(api_key=api_key)
            vp, usage = parse_with_openai(client, model.split(":", 1)[1], jpeg_pages, temperature)
            api_usage = {k: v for k, v in usage.items() if isinstance(v, int)}
            image_bytes_out = usage.get("image_bytes_out", image_bytes_out)  # type: ignore
            if api_usage:
                logger.log("vision.api.usage", cid=cid, **api_usage)
        elif model.startswith("local:"):
            name = model.split(":", 1)[1]
            vp, usage = parse_with_local(name, jpeg_pages, temperature)
            image_bytes_out = usage.get("image_bytes_out", image_bytes_out)  # type: ignore
        else:
            raise ValueError("Unsupported --model. Use openai:<name> or local:<name>.")

        # vision.parse.cleanA (log kept/dropped sample for MCQs)
        kept = {k: v for k, v in vp.part_a_map.items() if v in VALID_ABCD}
        dropped = [k for k, v in vp.part_a_map.items() if not v]
        logger.log("vision.parse.cleanA", cid=cid, kept=list(kept.keys()), dropped=dropped)

        # Grade
        part_a_score, part_a_meta = grade_part_a(vp.part_a_map)
        logger.log("grade.part_a", cid=cid)

        part_b_score, part_b_meta = grade_part_b(vp.b1_seq, vp.b2_seq)
        logger.log("grade.part_b", cid=cid)

        part_c_score = grade_part_c(vp.prompt_text, deterministic_award=args.prompt_award)
        logger.log("grade.part_c", cid=cid)

        totals = {
            "part_a": {"score": round(part_a_score, 2), "max": 4.0, "raw_correct": part_a_meta["raw_correct"], "raw_max": part_a_meta["raw_max"]},
            "part_b": {"score": round(part_b_score, 2), "max": 4.0, "b1": part_b_meta["b1"], "b2": part_b_meta["b2"]},
            "part_c": {"score": round(part_c_score, 2), "max": 2.0},
        }
        overall = round(totals["part_a"]["score"] + totals["part_b"]["score"] + totals["part_c"]["score"], 2)
        totals["overall"] = {"score": overall, "max": 10.0}

        # MCQ per-question breakdown
        part_a_breakdown = []
        for q in MCQ_KEYS:
            student_ans = (vp.part_a_map.get(q) or "").upper()
            correct_ans = GOLD_PART_A.get(q, "")
            part_a_breakdown.append({
                "question": q,
                "student": student_ans,
                "correct": correct_ans,
                "is_correct": bool(student_ans == correct_ans),
            })

        # Write outputs
        results = {
            "totals": totals,
            "extracted": {
                "part_a": vp.part_a_map,
                "part_b": {"b1": vp.b1_seq, "b2": vp.b2_seq},
                "part_c": {"prompt": vp.prompt_text},
            },
            "part_a": part_a_breakdown,
            "part_b": [
                {"question": "B1", "score": part_b_meta["b1"], "explanation": f"{int(part_b_meta['b1']*8)}/8 positions match."},
                {"question": "B2", "score": part_b_meta["b2"], "explanation": f"{int(part_b_meta['b2']*8)}/8 positions match."},
            ],
        }

        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)

        # finalize logs
        logger.log(
            "grade.done",
            cid=cid,
            totals=totals,
            image_bytes_in=image_bytes_in,
            image_bytes_out=image_bytes_out,
        )

        return 0

    except Exception as e:
        logger.log("error", level="ERROR", cid=cid, message=str(e))
        logger.log("grade.done", level="ERROR", cid=cid)
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    sys.exit(main())



