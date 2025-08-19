import os
import json
import time
from llm import query_llm
import chunking # Custom python script


# === Configuration ===
OLLAMA_MODEL = "deepseek-r1-12k:32b"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_EXTERNAL_URL = "http://api.lontsolutions.com/ask"
PROCESSED_DIR = "./processed"
THREADS_FILE = os.path.join(PROCESSED_DIR, "threads.jsonl")
EMAILS_FILE = os.path.join(PROCESSED_DIR, "emails.jsonl")
SUMMARY_OUTPUT_FILE = os.path.join(PROCESSED_DIR, "email_summaries.jsonl")
CHUNKS_OUTPUT_FILE = os.path.join(PROCESSED_DIR, "email_chunks.jsonl")
REFRESH_ALL = False
USE_EXTERNAL_SERVER = True
CHUNKS_ONLY = False # Set to True to only chunk emails without summarizing
SKIP_CHUNKING = False  # Set to True to skip chunking and only summarize
CHUNK_TYPE = "email" # default is email. Otherwise choose 'paragraph' or 'semantic'
IDEAL_TOKENS = 1000
IDEAL_OVERLAP = 100


def load_threads_with_emails(threads_path: str = THREADS_FILE, emails_path: str = EMAILS_FILE) -> list[dict]:
    threads = []
    email_data_map = {}

    # Load all emails into a dictionary
    if os.path.exists(emails_path):
        with open(emails_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                    email_id = obj.get("email_id")
                    data = obj.get("data")
                    email_data_map[email_id] = data
                except Exception as e:
                    print(f"⚠️ Skipping malformed line {line_number} in emails file: {e}")

    # Load threads and attach emails
    if os.path.exists(threads_path):
        with open(threads_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    thread_obj = json.loads(line)
                    thread_id = thread_obj.get("thread_id")
                    email_ids = thread_obj.get("email_ids", [])
                    emails = [email_data_map[eid] for eid in email_ids if eid in email_data_map]
                    threads.append({
                        "thread_id": thread_id,
                        "email_ids": email_ids,
                        "emails": emails,
                        "subject": emails[0].get("subject", "N/A") if emails else "N/A",
                        "topic": "",
                        "detail1" : "",
                        "detail2": "",
                    })
                except Exception as e:
                    print(f"⚠️ Skipping malformed line {line_number} in threads file: {e}")
    
    return threads



def load_summarized_threads(filepath: str = SUMMARY_OUTPUT_FILE) -> list[dict]:
    summarized_threads = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                    thread_id = obj.get("thread_id")
                    subject = obj.get("subject", "N/A")
                    topic = obj.get("topic", "N/A")
                    detail1 = obj.get("detail1", "N/A")
                    detail2 = obj.get("keypoints", "N/A")
                    summarized_threads.append({
                        "thread_id": thread_id,
                        "subject": subject,
                        "topic": topic,
                        "detail1": detail1,
                        "detail2": detail2
                    })
                except Exception as e:
                    print(f"⚠️ Skipping malformed line {line_number} in summaries file: {e}")
    return summarized_threads



def load_chunks(filepath: str = CHUNKS_OUTPUT_FILE) -> list[dict]:
    chunks = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                    thread_id = obj.get("thread_id")
                    subject = obj.get("subject", "N/A")
                    chunk_data = obj.get("chunks", [])
                    chunks.append({
                        "thread_id": thread_id,
                        "subject": subject,
                        "chunks": chunk_data
                        })
                except Exception as e:
                    print(f"⚠️ Skipping malformed line {line_number} in chunks file: {e}")
    return chunks



def load_summarized_thread_ids(filepath: str = SUMMARY_OUTPUT_FILE) -> set:
    summarized_ids = set()
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                    thread_id = obj.get("thread_id")
                    if thread_id:
                        summarized_ids.add(thread_id)
                except Exception as e:
                    print(f"⚠️ Skipping malformed line {line_number} in summaries file: {e}")
    return summarized_ids



def load_chunked_thread_ids(filepath: str = CHUNKS_OUTPUT_FILE) -> set:
    chunk_ids = set()
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                    thread_id = obj.get("thread_id")
                    if thread_id:
                        chunk_ids.add(thread_id)
                except Exception as e:
                    print(f"⚠️ Skipping malformed line {line_number} in summaries file: {e}")
    return chunk_ids



def load_changed_threads(threads_path: str = "changed_threads.tmp") -> list[str]:
    if not os.path.exists(threads_path):
        return []
    with open(threads_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️ Error loading changed threads: {e}")
            return []



def clean_thread(full_thread, model_name: str=OLLAMA_MODEL,):
    thread_id = full_thread["thread_id"]
    for email in full_thread["emails"]:
        content= email['body']
        
        tokens = chunking.estimate_tokens(content)
        print(f"Email '{email['subject']}' in thread {thread_id} has approximately {tokens} tokens.")
        
        if tokens < 3:
            continue
        
        chunk_warning = 12000
        if tokens > chunk_warning:
            print(f"❗ Warning, thread {thread_id} has paragraph chunks bigger than {chunk_warning} tokens")  
        
        prompt = (
            "For the text below, clean it up for summarization - removing signatures, links, and unnecessary information.\n"
            "If there is no text given, just return 'end_of_body'. If you don't have enough information, just return the text as-is.\n"
            "Don't add commentary, feedback or verbose statements.\n"
            "Text:\n\n"
            f"{content}"
        )
        
        if USE_EXTERNAL_SERVER:
            result = query_llm(prompt, model_name=model_name, model_loc="external_api", use_chat=False).strip()
        else:
            result = query_llm(prompt, model_name=model_name, use_chat=False).strip()
            
        # strip out any <think>…</think>
        if "</think>" in result:
            result = result.split("</think>", 1)[-1].strip()
            
        if "end_of_body" in result:
            result = result.split("end_of_body", 1)[0].strip()
        
        # join this email’s chunk summaries into one email‐level summary
        email["body"] = result
    
    return full_thread
        



def chunk_thread(full_thread, type: str = CHUNK_TYPE) -> list[str]:
    thread_id = full_thread["thread_id"]
    all_chunks = []
    for email in full_thread["emails"]:
        chunks = []
        metadata = (
            f"From: {email['sender']}\n"
            f"Subject: {email['subject']}\n"
            f"Date: {email['date']}\n\n"
        )
        header = f"Email subject: {email['subject']}\nEmail sender: {email['sender']}\n"
        content= f"{metadata}\n\n{email['body']}"
        
        if type == "paragraph":
            chunks = chunking.chunk_by_paragraph(content, header=header, size=IDEAL_TOKENS)
            for chunk in chunks:
                tokens = chunking.estimate_tokens(chunk)
                if tokens > 4000:
                    print(f"❗ Warning, thread {thread_id} has paragraph chunks bigger than 4000 tokens")  
            all_chunks.extend(chunks)
            
        elif type == "semantic":
            chunks = chunking.chunk_by_semantics(content, header=header)
            for chunk in chunks:
                tokens = chunking.estimate_tokens(chunk)
                if tokens > 4000:
                    print(f"❗ Warning, thread {thread_id} has semantic chunks bigger than 4000 tokens")  
            all_chunks.extend(chunks)
            
        else:
            tokens = chunking.estimate_tokens(content)
            if tokens > 4000:
                print(f"❗ Warning, thread {thread_id} has emails bigger than 4000 tokens")  
            all_chunks.append(content)

    return all_chunks



def summarize_text(text, model_name):
    topic_prompt = (
        "State concisely the topic or project discussed in the text below.\n"
        "Try to keep it to one sentence if reasonable.\n"
        "Return only '...' if you don't have sufficient information, don't understand, or no text is given.\n"
        "Don't add commentary, feedback or verbose statements.\n"
        "Text:\n\n"
        f"{text}"
    )
    detail1_prompt = (
        "Summarize concisely the text below for the purpose of semantic search by engineers. Put each sentence on a new line.\n"
        "Focus on including information like useful facts, decisions made and important names.\n"
        "Avoid reiterating the entire text and metadata like subject, sender and date.\n"
        "Return only '...' if you don't understand the text, or no text is given.\n"
        "Don't add commentary, feedback or verbose statements.\n"
        "Text:\n\n"
        f"{text}"
    )
    detail2_prompt = (
        "For the text below, \n"
        "Return only '...' if you don't have sufficient information, don't understand, or no text is given.\n"
        "Don't add commentary, feedback or verbose statements.\n"
        "Text:\n\n"
        f"{text}"
    )

    if USE_EXTERNAL_SERVER:
        topic = query_llm(topic_prompt, model_name=model_name, model_loc="external_api", use_chat=False).strip()
        detail1 = query_llm(detail1_prompt, model_name=model_name, model_loc="external_api", use_chat=False).strip()
        #detail2 = query_llm(detail2_prompt, model_name=model_name, model_loc="external_api", use_chat=False).strip()
        detail2 = "..."  # No detail2 for external API to avoid extra cost
    else:
        topic = query_llm(topic_prompt, model_name=model_name, use_chat=False).strip()
        detail1 = query_llm(detail1_prompt, model_name=model_name, use_chat=False).strip()
        #detail2 = query_llm(detail2_prompt, model_name=model_name, use_chat=False).strip()
        detail2 = "..."  # No detail2 for local model to avoid extra cost

    # Strip out any <think>…</think>
    if "</think>" in topic:
        topic = topic.split("</think>", 1)[-1].strip()
    if "</think>" in detail1:
        detail1 = detail1.split("</think>", 1)[-1].strip()
    if "</think>" in detail2:
        detail2 = detail2.split("</think>", 1)[-1].strip()

    return topic, detail1, detail2



def summarize_thread(thread_id, chunks, per_chunk=False, model_name=OLLAMA_MODEL):
    print(f"🧠 Summarizing thread {thread_id} with {len(chunks)} chunk(s)...")

    if per_chunk:
        print("🔍 Summarizing each chunk individually...")
        chunk_topics = []
        chunk_detail1 = []
        chunk_detail2 = []

        for i, chunk in enumerate(chunks, start=1):
            print(f"   📄 Chunk {i}/{len(chunks)}")
            topic, detail1, detail2 = summarize_text(chunk, model_name)
            chunk_topics.append(topic)
            chunk_detail1.append(detail1)
            chunk_detail2.append(detail2)

        topic = "\n\n".join(chunk_topics)
        detail1 = "\n\n".join(chunk_detail1)
        detail2 = "\n\n".join(chunk_detail2)        

    else:
        print("🔍 Summarizing each thread as a whole...")
        full_text = "\n\n".join(chunks)
        tokens = chunking.estimate_tokens(full_text)
        print(f"The whole thread {thread_id} has approximately {tokens} tokens.")

        topic, detail1, detail2 = summarize_text(full_text, model_name)

    return topic, detail1, detail2



def save_summary(
    summaries: list[dict],
    filepath: str = SUMMARY_OUTPUT_FILE):
    with open(filepath, "w", encoding="utf-8") as f:
        for summary in summaries:
            if not summary.get("thread_id"):
                continue
            json.dump(summary, f)
            f.write("\n")
        
def save_chunks(
    chunks: list[dict],
    filepath: str = CHUNKS_OUTPUT_FILE):
    with open(filepath, "w", encoding="utf-8") as f:
        for chunk in chunks:
            if not chunk.get("thread_id") or not chunk.get("chunks"):
                continue
            json.dump(chunk, f)
            f.write("\n")





def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    changed_threads = load_changed_threads()
    existing_chunks = []
    existing_summaries = []
    time_start = time.time()
    
    if not SKIP_CHUNKING:

        print("📥 Loading threads...")
        all_data = load_threads_with_emails()
        print(f"📄 Found {len(all_data)} threads.")
        
        if REFRESH_ALL:
            print("📥 Refreshing all chunks.")
            existing_chunks = []
            chunked_ids = set()
        else:
            print("📥 Loading existing chunks.")
            existing_chunks = load_chunks()
            chunked_ids = load_chunked_thread_ids()
            total_chunks = sum(len(thread.get("chunks") or []) for thread in existing_chunks)
            print(f"📄 Found {total_chunks} existing chunks from {len(chunked_ids)} IDs.")

        for i, full_thread in enumerate(all_data):
            thread_id = full_thread.get("thread_id", "N/A")
            subject = full_thread.get("subject", "N/A")

            if not thread_id in chunked_ids or thread_id in changed_threads:
                try:
                    cleaned_thread = clean_thread(full_thread)
                    chunks = chunk_thread(cleaned_thread)
                    existing_chunks.append({
                        "thread_id": thread_id,
                        "subject": subject,
                        "chunks": chunks
                    })
                    elapsed = time.time() - time_start
                    print(f"✅ [{elapsed:.2f}s]: ({i+1}/{len(all_data)}) cleaned thread {thread_id}")
                    
                    save_chunks(existing_chunks)
                        
                except Exception as e:
                    print(f"❌ Failed to chunk thread {thread_id}: {e}\n")
                    continue
        
    if not CHUNKS_ONLY:
        if SKIP_CHUNKING:
            print("📥 Loading existing chunks...")
            existing_chunks = load_chunks()
        
        if REFRESH_ALL:
            print("📖 Refreshing all summarized threads...")
            existing_summaries = []
            summarized_ids = set()
        else:
            print("📖 Loading already summarized threads...")
            existing_summaries = load_summarized_threads()
            summarized_ids = load_summarized_thread_ids()
            print(f"📄 Found {len(existing_summaries)} existing summaries from {len(summarized_ids)} IDs.")
        
        for i, full_chunk in enumerate(existing_chunks):
            thread_id = full_chunk.get("thread_id", "N/A")
            subject = full_chunk.get("subject", "N/A")
            chunks = full_chunk.get("chunks", "N/A")

            if not thread_id in summarized_ids or thread_id in changed_threads:
                try:
                    topic, detail1, detail2 = summarize_thread(thread_id, chunks)
                    existing_summaries.append({
                        "thread_id": thread_id,
                        "subject": subject,
                        "topic": topic,
                        "detail1": detail1,
                        "detail2": detail2
                    })
                    elapsed = time.time() - time_start
                    print(f"✅ [{elapsed:.2f}s]: ({i+1}/{len(existing_chunks)}) Summarized thread {thread_id}\n")
                    
                    save_summary(existing_summaries)
                    
                except Exception as e:
                    print(f"❌ Failed to summarize thread {thread_id}: {e}\n")
                    continue
                    




if __name__ == "__main__":
    main()



