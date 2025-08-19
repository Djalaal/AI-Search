#!/usr/bin/env python3
import os
import time
import json
import re
from typing import List, Dict

# Reuse your existing config + loaders so paths/models match the main script
from summarize_emails import (
    OLLAMA_MODEL,
    USE_EXTERNAL_SERVER,
    SUMMARY_OUTPUT_FILE,
    CHUNKS_OUTPUT_FILE,
    load_chunks,
)
from llm import query_llm

# === Customize your temporary keypoints prompt here ===
NEW_DETAIL2_PROMPT = (
    "Summarize the text below concisely for the purpose of semantic search using bullet points.\n"
    "Avoid reiterating the entire email or metadata.\n"
    "Return only '...' if you don't understand the text, or no text is given.\n"
    "Don't add commentary, feedback or verbose statements.\n"
    "Text:\n\n"
    "{TEXT}"
)

# === Options ===
OUTPUT_SUFFIX = ".detail2_refresh"    # new file: email_summaries.jsonl.detail2_refresh
OVERWRITE_IN_PLACE = False            # set True to overwrite SUMMARY_OUTPUT_FILE (backed up first)
DRY_RUN_LIMIT = 0                     # set >0 to only process first N threads for testing
RESUME_IF_EXISTS = True  

def recompute_detail2(text: str, model_name: str) -> str:
    prompt = NEW_DETAIL2_PROMPT.replace("{TEXT}", text)
    if USE_EXTERNAL_SERVER:
        out = query_llm(prompt, model_name=model_name, model_loc="external_api")
    else:
        out = query_llm(prompt, model_name=model_name)
    return out

def iter_jsonl(path: str):
    """Yield parsed objects from a JSONL file; skip malformed lines."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"⚠️ Skipping malformed line {i} in {path}: {e}")

def join_chunks(chunks: List[str]) -> str:
    # Mimic how you summarize whole threads today: join chunks into one text
    return "\n\n".join(chunks or [])

def backup_file(path: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    bak = f"{path}.bak.{ts}"
    with open(path, "rb") as fsrc, open(bak, "wb") as fdst:
        fdst.write(fsrc.read())
    return bak

def main():
    os.makedirs(os.path.dirname(SUMMARY_OUTPUT_FILE), exist_ok=True)

    # Load chunk text once; we’ll look up by thread_id fast
    chunks_list = load_chunks(CHUNKS_OUTPUT_FILE)
    chunks_by_id: Dict[str, List[str]] = {c["thread_id"]: (c.get("chunks") or []) for c in (chunks_list or [])}
    print(f"📥 Loaded chunks for {len(chunks_by_id)} thread(s).")

    # Decide output path
    if OVERWRITE_IN_PLACE:
        if os.path.exists(SUMMARY_OUTPUT_FILE):
            bak = backup_file(SUMMARY_OUTPUT_FILE)
            print(f"🗃️  Backed up original to: {bak}")
        out_path = SUMMARY_OUTPUT_FILE
        mode = "w"
        processed_ids = set()
    else:
        out_path = SUMMARY_OUTPUT_FILE + OUTPUT_SUFFIX
        # If resuming, collect already-written ids from prior partial runs
        processed_ids = set()
        if RESUME_IF_EXISTS and os.path.exists(out_path):
            for obj in iter_jsonl(out_path):
                tid = obj.get("thread_id")
                if tid:
                    processed_ids.add(tid)
            mode = "a"
            print(f"⏩ Resuming: {len(processed_ids)} thread(s) already written in {out_path}")
        else:
            mode = "w"

    # Open output for streaming writes
    out_f = open(out_path, mode, encoding="utf-8")

    # Stream input summaries line-by-line
    in_count = 0
    out_count = 0
    for s in iter_jsonl(SUMMARY_OUTPUT_FILE):
        in_count += 1
        tid = s.get("thread_id")
        subject = s.get("subject", "N/A")

        # Resume skip
        if tid and tid in processed_ids:
            continue

        # Build the text to summarize from existing chunks; if none, keep old detail2
        text = join_chunks(chunks_by_id.get(tid, []))
        if text.strip():
            new_detail2 = recompute_detail2(text, OLLAMA_MODEL)
        else:
            new_detail2 = s.get("detail2", "")

        # Emit a new JSONL line immediately
        s_out = dict(s)
        s_out["detail2"] = new_detail2
        json.dump(s_out, out_f, ensure_ascii=False)
        out_f.write("\n")
        out_f.flush()
        try:
            os.fsync(out_f.fileno())  # ensure data hits disk so you can tail it
        except Exception:
            pass

        out_count += 1
        print(f"✅ [{out_count}] {tid}  {subject[:70]}")

        if DRY_RUN_LIMIT and out_count >= DRY_RUN_LIMIT:
            break

    out_f.close()
    print(f"🎉 Wrote {out_count} updated summaries to {out_path} (scanned {in_count}).")

if __name__ == "__main__":
    main()
