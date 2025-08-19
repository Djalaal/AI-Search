import json
import os

SUMMARY_FILE = "./processed/email_summaries.jsonl"
OUTPUT_FILE = "./processed/texts/email_summaries.txt"

def write_summary_overview(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for i, line in enumerate(infile, start=1):
            try:
                data = json.loads(line)
                thread_id = data.get("thread_id", "N/A")
                subject = data.get("subject", "").strip()
                topic = data.get("topic", "").strip()
                detail1 = data.get("detail1", "").strip()
                detail2 = data.get("detail2", "").strip()

                outfile.write(f"\n\n\n==================== Thread {i} ====================\n")
                outfile.write(f"Thread ID: {thread_id}\n")
                outfile.write(f"Subject: {subject}\n")
                outfile.write(f"Topic:\n{topic}\n")
                outfile.write(f"Detail1:\n{detail1}\n")
                outfile.write(f"Detail2:\n{detail2}\n")

            except json.JSONDecodeError as e:
                outfile.write(f"\n\n\nError parsing line {i}: {e}\n")

    print(f"Summaries written to: {output_path}")


if __name__ == "__main__":
    write_summary_overview(SUMMARY_FILE, OUTPUT_FILE)
