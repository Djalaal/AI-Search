import os
import json
import hashlib
import dateparser
import re
import extract_msg
from llm import query_llm
from chunking import estimate_tokens

REFRESH_ALL = False
USE_EXTERNAL_SERVER = True
DOCUMENTS_DIR = "./documents"
PROCESSED_DIR = "./processed"
THREADS_FILE = os.path.join(PROCESSED_DIR, "threads.jsonl")
EMAILS_FILE = os.path.join(PROCESSED_DIR, "emails.jsonl")
EMAIL_IDS_FILE = os.path.join(PROCESSED_DIR, "email_ids.json")
MSG_IDS_FILE = os.path.join(PROCESSED_DIR, "msg_ids.json")
VALID_HEADERS = {"from", "sent", "to", "cc", "subject"}
HEADER_MAP = {
    "Van":        "From",
    "Verzonden":  "Sent",
    "Aan":        "To",
    "Onderwerp":  "Subject",
}


os.makedirs(PROCESSED_DIR, exist_ok=True)

# Helper functions for loading and saving email IDs, email threads, and msg IDs
def load_email_ids(file: str=EMAIL_IDS_FILE) -> dict:
    try:
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)  # Returns a dict
    except (FileNotFoundError, json.JSONDecodeError):
        return {}  # Empty dict if file missing or malformed

def save_email_ids(email_ids: dict, file: str=EMAIL_IDS_FILE):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(email_ids, f, indent=2)
        
def load_threads(file: str = THREADS_FILE) -> dict:
    threads = {}
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                    threads[obj["thread_id"]] = obj["email_ids"]
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping malformed structure JSON on line {line_number}: {e}")
    return threads

def save_threads(threads: dict, changes: list = [], file: str = THREADS_FILE):
    with open("changed_threads.tmp", "w", encoding="utf-8") as f:
        json.dump(changes, f)
    with open(file, "w", encoding="utf-8") as f:
        for thread_id, email_ids in threads.items():
            json.dump({"thread_id": thread_id, "email_ids": email_ids}, f)
            f.write("\n")

            
def load_emails(file: str = EMAILS_FILE) -> dict:
    emails = {}
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                    emails[obj["email_id"]] = obj["data"]
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping malformed email JSON on line {line_number}: {e}")
    return emails

def save_emails(emails: dict, file: str = EMAILS_FILE):
    with open(file, "w", encoding="utf-8") as f:
        for email_id, email_data in emails.items():
            json.dump({"email_id": email_id, "data": email_data}, f)
            f.write("\n")

def load_msg_ids(file: str=MSG_IDS_FILE) -> set:
    try:
        with open(file, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()

def save_msg_ids(msg_ids: set, file: str=MSG_IDS_FILE):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(list(msg_ids), f, indent=2)





# Helper function to convert (any) dates to the iso format standard
def convert_date(date_raw):
    # Fix malformed AM/PM issues like "9:11 A"
    date_raw = re.sub(r"\b(\d{1,2}:\d{2})\s*A\b", r"\1 AM", date_raw)
    date_raw = re.sub(r"\b(\d{1,2}:\d{2})\s*P\b", r"\1 PM", date_raw)
    
    parsed_date = dateparser.parse(date_raw)
    if parsed_date:
        return parsed_date.isoformat()
    else:
        print(f"Warning: Could not parse date '{date_raw}'")
        return date_raw

def convert_metadata_language(email, model_name: str = "deepseek-r1-12k:8b"):
    prompt = (
        "For the following email, replace the non-english email metadata with the following appropriate English equivalents if applicable:\n"
        "From:\n"
        "Sent:\n"
        "To:\n"
        "CC:\n"
        "Subject:\n"
        "Don't make any other changes to the email. Don't give me any commentary, verbose politeness, feedback or questions. Just be silent if you don't know something.\n\n"
        f"{email}"
    )
    if USE_EXTERNAL_SERVER:
        result = query_llm(prompt, model_name=model_name, model_loc="external_api", use_chat=False).strip()
    else:
        result = query_llm(prompt, model_name=model_name, use_chat=False).strip()
    
    # strip out any <think>…</think>
    if "</think>" in result:
        #result_think = result.split("</think>", 1)[0].replace("<think>", "").strip()
        result = result.split("</think>", 1)[-1].strip()
        
    return result



# Helper function to extract relevant data from a .msg file
def extract_email_data(msg, msg_path) -> tuple[dict[str, dict], set, set]:
    # The .msg files contain not only the latest email, but also the email history. These must be separated.
    # The metadata is added to the beginning to match the formatting of emails found in the email history.
    # Replace dutch headers with english ones
    
    header = ""
    header += f"From: {msg.sender}\n"
    header += f"Sent: {msg.date}\n"
    header += f"To: {msg.to}\n"
    if msg.cc:
        header += f"CC: {msg.cc}\n"
    header += f"Subject: {msg.subject}\n\n"
    
    email = header + msg.body
    email = email.replace('\r\n', '\n')
    
    # Translate all Dutch metadata to English
    for dutch, eng in HEADER_MAP.items():
        pattern     = rf"\n\s*{re.escape(dutch)}:"
        replacement = f"\n{eng}:"
        email = re.sub(pattern, replacement, email, flags=re.IGNORECASE)
    
    # Split the whole email thread into separate emails
    split_pattern = r'''(?<=\n)                     # only split after a newline
    (?=                                             # lookahead to mark split point
    (?:^ (?=.*:)(?=.*@) .*\r?\n)                    # line 1: has ':' & '@' (from)
    (?:^ (?=.*:) .*\r?\n)                           # line 2: has ':' (date)
    (?:^ (?=.*:)(?=.*@) .*\r?\n)                    # line 3: has ':' & '@' (from)
    )
    '''
    
    raw_parts = re.split(split_pattern, email, flags=re.VERBOSE|re.MULTILINE)
    parts     = [p.lstrip(' \t') for p in raw_parts if p.strip()]   # filter out any empty strings
    thread_id = hashlib.sha256(parts[-1].encode("utf-8")).hexdigest() # A placeholder
        
    # Assigning metadata for each email
    new_emails = {}
    tokens = []
    abnormal_mail_ids = set()
    invalid_mail_threads = set()
    invalid_mail = False
    
    for part in parts:
        abnormal_mail = False
        invalid_mail = False
        #1 Check for abnormal email headers
        if not part.lower().startswith("from:"):
            abnormal_mail = True
            
            # Remove bullet points from header if applicable
            if part.startswith("*\t"): 
                lines = part.splitlines()
                for idx, line in enumerate(lines):
                    if idx < 5 and line.startswith("*\t"):
                        lines[idx] = line.replace("*\t","")
                    elif idx >= 5: break
                part = "\n".join(lines)
                if part.lower().startswith("from:"): # Check for English header after removing bullet points
                    break
            
            # Normalize to English metadata (with AI) if applicable
            elif part[0].lower().isalpha():
                part = convert_metadata_language(part)
                print(f"✅ Converted non-english metadata.")

            # If all the above fails, skip the email to avoid errors later on
            else:
                invalid_mail = True
                print("Skipping email. Misformatted metadata.")
                continue
        
        #2 Split into non‑empty lines
        lines = [line.lstrip(' \t') for line in part.splitlines() if line.strip()]
        
        #3 Initialize
        meta = {"sender": "", "date": "", "receiver": "", "cc": "", "subject": ""}
        body_lines = []

        #4 Filter out metadata from body text
        proceed = False 
        for line in lines:
            if line.lower().startswith("from:"):
                meta["sender"] = line[5:].strip()
                proceed = True
            elif not proceed: 
                continue # skipping any garbage AI might have added before the actual email when translating it
            elif line.lower().startswith("sent:"):
                date_raw = line[5:].strip()
                meta["date"] = convert_date(date_raw)
            elif line.lower().startswith("to:"):
                meta["receiver"] = line[3:].strip()
            elif line.lower().startswith("cc:"):
                meta["cc"] = line[3:].strip()
            elif line.lower().startswith("subject:"):
                subject_raw = line[8:].strip()
                subject = re.sub(r"^(re:|fw:|fwd:)\s*", "", subject_raw, flags=re.IGNORECASE).strip()
                meta["subject"] = subject
            else:
                body_lines.append(line)
        
        email_body = "\n".join(body_lines).strip()
        if not email_body:
            print(f"Empty email found in thread {thread_id}.")
        tokens.append(estimate_tokens(email_body))
        
        id_text = meta["sender"][:5] + meta["date"][:10] + email_body
        email_id = hashlib.sha256(id_text.encode("utf-8")).hexdigest()
        if abnormal_mail:
            abnormal_mail_ids.add(email_id)
        if invalid_mail:
            invalid_mail_threads.add(thread_id)
        
        new_emails[email_id] = {
            "thread_id": thread_id,
            "subject": meta["subject"],
            "sender": meta["sender"],
            "receiver": meta["receiver"],
            "cc": meta["cc"],
            "date": meta["date"],
            "file_path": msg_path,
            "body": email_body
        }

    print(f"ℹ️  Extracted {len(new_emails)} emails from {msg_path}, with the number of tokens being {tokens}.")
    if invalid_mail:
        print(f"❗ Thread {thread_id} contains invalid mails.")

    return new_emails, abnormal_mail_ids, invalid_mail_threads





# Main function that processes all emails
def process_msg_files(doc_path: str = DOCUMENTS_DIR, only_new: bool = not REFRESH_ALL):
    all_msg_ids: set[str] = load_msg_ids()
    all_emails: dict[str, dict] = load_emails()
    all_threads: dict[str, list[str]] = load_threads()
    changed_threads = []
    email_count = 0
    thread_count = 0
    msg_count = 0
    abnormal_email_ids = set()
    invalid_email_threads = set()

    # Processing all .msg files and populating the new_* variables
    print("📥 Scanning for new .msg files...")
    for root, _, files in os.walk(doc_path):
        for file in files:
            if not file.endswith(".msg"):
                continue
            filepath = os.path.join(root, file)
            
            try:
                msg = extract_msg.Message(filepath)
                msg_count += 1
                if only_new and msg.messageId in all_msg_ids:
                    continue
                
                if not msg.messageId:
                    continue
                
                # Extract emails from the .msg file
                new_emails, abnormal_ids, invalid_ids = extract_email_data(msg, filepath)
                if abnormal_ids:
                    abnormal_email_ids.update(abnormal_ids)
                if invalid_ids:
                    invalid_email_threads.update(invalid_ids)
                
                # Step 1: Make a set of all emails that belong to the same thread, removing duplicate threads
                new_email_ids = set(new_emails)
                for email_id in new_emails:
                    if email_id in all_emails:
                        existing_thread_id = all_emails[email_id]["thread_id"]   
                        new_email_ids.update(all_threads.get(existing_thread_id, []))
                        all_threads.pop(existing_thread_id, None)
                    else:
                        all_emails[email_id] = new_emails[email_id]
                        email_count += 1
                        
                # Step 2: Sort the emails according to date (according to the "date" metadata)
                try:
                    sorted_email_ids = sorted(
                        new_email_ids,
                        key=lambda email_id: all_emails[email_id]["date"]
                        if email_id in all_emails else new_emails[email_id]["date"],
                        reverse = True
                    )
                except Exception as e:
                    print(f"⚠️ Failed to sort emails by date: {e}")
                    sorted_email_ids = list(new_email_ids)
                
                # Step 3: Add the new emails and update the thread ID of existing emails in the Emails document, and add the thread to the Threads document 
                new_thread_id = all_emails[sorted_email_ids[-1]]["thread_id"]
                changed_threads.append(new_thread_id)
                for email_id in sorted_email_ids:                  
                    all_emails[email_id]["thread_id"] = new_thread_id
                        
                all_threads[new_thread_id] = sorted_email_ids
                thread_count += 1
                all_msg_ids.add(msg.messageId)
                
                # 💾 Save everything
                save_threads(all_threads, changes=changed_threads)
                save_emails(all_emails)
                save_msg_ids(all_msg_ids)
                        
            except Exception as e:
                print(f"⚠️ Error processing {file}: {e}")

    print(f"✅ Saved {email_count} new emails across {thread_count} threads by scanning {msg_count} .msg files.")
    if abnormal_email_ids:
        print(f"❗ There are {len(abnormal_email_ids)} abnormal mails:")
        for mail_id in abnormal_email_ids:
            print(mail_id)
    if invalid_email_threads:
        print(f"❗ There are {len(invalid_email_threads)} invalid mails in threads:")
        for thread_id in invalid_email_threads:
            print(thread_id)
        

if __name__ == "__main__":
    process_msg_files()
