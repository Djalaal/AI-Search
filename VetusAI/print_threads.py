import json
import os

THREADS_INPUT_FILE = "./processed/threads.jsonl"
EMAILS_INPUT_FILE = "./processed/emails.jsonl"
FULL_OUTPUT_FILE = "./processed/texts/threads_full.txt"
OVERVIEW_OUTPUT_FILE = "./processed/texts/threads_overview.txt"

def write_thread_overview(
    threads_path: str = THREADS_INPUT_FILE,
    emails_path: str = EMAILS_INPUT_FILE,
    overview_path: str = OVERVIEW_OUTPUT_FILE,
    full_path: str = FULL_OUTPUT_FILE):
    
    if not os.path.exists(threads_path) or not os.path.exists(emails_path):
        print(f"❌ One or more input files not found.")
        return
    

    # Load threads from JSONL
    threads: dict[str, list[str]] = {}
    with open(threads_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                thread_id = obj["thread_id"]
                email_ids = obj["email_ids"]
                threads[thread_id] = email_ids
            except Exception as e:
                print(f"⚠️ Skipping malformed line {line_number} in threads file: {e}")


    # Load emails from JSONL
    emails: dict[str, dict] = {}
    with open(emails_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                email_id = obj["email_id"]
                email_data = obj["data"]
                emails[email_id] = email_data
            except Exception as e:
                print(f"⚠️ Skipping malformed line {line_number} in emails file: {e}")

    with open(overview_path, "w", encoding="utf-8") as overview_outfile, \
         open(full_path, "w", encoding="utf-8") as full_outfile:

        for i, (thread_id, email_ids) in enumerate(threads.items(), start=1):
            if not email_ids:
                continue

            # Use first available email to get subject
            first_valid_email = next((emails[eid] for eid in email_ids if eid in emails), {})
            subject = first_valid_email.get("subject", "N/A")

            overview_outfile.write(f"\nThread {i}\n")
            overview_outfile.write(f"Thread ID: {thread_id}\n")
            overview_outfile.write(f"Subject: {subject}\n")
            overview_outfile.write(f"{len(email_ids)} Email IDs:\n")

            full_outfile.write(f"\n===== Thread {i} =====\n")
            full_outfile.write(f"Thread ID: {thread_id}\n")
            full_outfile.write(f"Subject: {subject}\n")

            for eid in email_ids:
                email = emails.get(eid)
                if not email:
                    continue
                sender = email.get("sender", "N/A")
                date = email.get("date", "N/A")
                body = email.get("body", "").strip()

                overview_outfile.write(f"  - {eid}\t\t{date}\n")

                full_outfile.write("\n--- Email ---\n")
                full_outfile.write(f"* From: {sender}\n")
                full_outfile.write(f"* Date: {date}\n")
                full_outfile.write(f"* Email ID: {eid}\n\n")
                full_outfile.write(body + "\n")

    print(f"✅ Overview written to: {overview_path}")
    print(f"✅ Full email bodies written to: {full_path}")

if __name__ == "__main__":
    write_thread_overview()
