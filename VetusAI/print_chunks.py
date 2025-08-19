import os
import json
from chunking import estimate_tokens # Custom python script

# === Configuration ===
CHUNKS_INPUT_FILE    = "./processed/email_chunks.jsonl"
CHUNKS_OUTPUT_FILE   = "./processed/texts/email_chunks.txt"

def write_chunks(
    input_path: str = CHUNKS_INPUT_FILE,
    output_path: str = CHUNKS_OUTPUT_FILE
):
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return

    # Load and group chunks by thread_id
    chunks_by_thread: dict[str, list[str]] = {}
    with open(input_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
                thread_id   = obj.get("thread_id", "N/A")
                chunks_by_thread[thread_id] = obj.get("chunks", [])

            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping malformed JSON on line {lineno}: {e}")

    # Write out the grouped chunks with bullets
    with open(output_path, "w", encoding="utf-8") as out:
        for idx, (thread_id, chunks) in enumerate(chunks_by_thread.items(), start=1):
            out.write(f"\n\n\n==================== Thread {idx} ====================\n")
            out.write(f"Thread ID: {thread_id}\n")
            out.write("Chunks:\n")
            for chunk in chunks:
                tokens = estimate_tokens(chunk)
                if tokens > 4000:
                    print(f"❗ Warning, thread {thread_id} has chunks bigger than 4000 tokens")
                out.write(f"#################### {tokens} ####################\n")
                out.write(f"{chunk}\n")
    print(f"✅ Chunks written to: {output_path}")

if __name__ == "__main__":
    write_chunks()
