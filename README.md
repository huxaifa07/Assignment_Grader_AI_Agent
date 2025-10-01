## Agentic AI Assignment 2 – CLI Grader

This repository contains a command-line grader that reads 1–4 scanned quiz pages, downscales and re-encodes them to JPEG, uses a vision model to extract answers, grades deterministically against the provided gold key, and writes `results.json` and structured logs in `logs.jsonl`.

### Setup
1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. If using OpenAI models, copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

### Usage
```bash
python rollNumber_02_grader.py \
  --images quiz_pages/page1.jpg quiz_pages/page2.jpg \
  --out results.json --logs logs.jsonl \
  --model openai:gpt-4o-mini --temperature 0.0
```

Arguments:
- `--images`: 1–4 image paths in order (page1..N)
- `--out`: output JSON file path
- `--logs`: JSON Lines logs file path
- `--model`: `openai:<name>` or `local:<name>`
- `--temperature`: model temperature (default 0.0)
- `--prompt-award`: deterministic score (0..2) if prompt text is non-empty (default 2)
- `--max-side`: max side for downscale (default 1600)

### What the grader does
1. Reads 1–4 images (order = pages 1..N)
2. Downscales + JPEG-encodes (max side ≈ 1600 px)
3. Uses a vision model to extract:
   - Part A: MCQ map "1".."8" → "A|B|C|D" (sanitized to first A–D only)
   - Part B: two sequences (A..H) for B1/B2 (trimmed to 8, valid letters only)
   - Part C: student's prompt spec text
4. Grades deterministically using the gold key:
   - /10 total → 4 MCQ (scaled by % correct) + 4 flows (avg positional match) + 2 prompt
5. Writes `results.json` and structured logs `logs.jsonl`.

### Gold Key
- Part A: 1 C, 2 B, 3 A, 4 B, 5 A, 6 A, 7 C, 8 D
- Flows: B1 = C E A G B D F H • B2 = D A H B G F E C

### Required Logs (JSONL)
- `grade.start`
- `vision.parse.start` / `vision.api.usage`
- `vision.parse.cleanA`
- `grade.part_a` / `grade.part_b` / `grade.part_c`
- `llm.explain.usage` (if you add short explanations) – optional
- `grade.done` `error`

### Local model support 
You can implement `--model local:<name>` using an open-source VLM (e.g., LLaVA, Qwen2-VL, Qwen3-omni). The file already contains a `parse_with_local` hook to implement. Suggested steps:
1. Download/serve your chosen local VLM.
2. Implement `parse_with_local(name, jpeg_pages_bytes, temperature)` to call the model and return the same structure as the OpenAI path.
3. Document the setup commands and usage here.

### Deliverables
- `rollNumber_02_grader.py`, `requirements.txt`, `README.md`
- `results.json`, `logs.jsonl`
- `quiz_pages/` with your scanned pages


