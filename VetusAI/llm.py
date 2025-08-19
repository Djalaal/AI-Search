import re
import requests
from typing import Tuple
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage

# Configuration
OLLAMA_MODEL = "llama3.1-12k:8b"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_EXTERNAL_URL = "http://api.lontsolutions.com/ask"
SYSTEM_PROMPT = ""

def strip_think(s: str) -> str:
    if "</think>" in s:
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
        s = re.sub(r"^\s*[\s\S]*?\b\*{0,2}Answer:\*{0,2}\s*", "", s, flags=re.IGNORECASE)
    return s.strip()

def separate_thinking(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    m = re.search(r"<think>([\s\S]*?)</think>", text, flags=re.IGNORECASE)
    thinking = m.group(1).strip() if m else ""
    answer = strip_think(text)
    return answer, thinking

def query_llm(
    user_input: str,
    model_name: str = OLLAMA_MODEL,
    model_loc: str = OLLAMA_URL,
    system_prompt: str = SYSTEM_PROMPT,
    use_chat: bool = True,
    return_meta: bool = False,
) -> str:
    if model_loc == "external_api":
        payload = {
            "user_input": user_input,
            "model_name": model_name,
            "system_prompt": system_prompt,
            "model_loc": "",  # remote server decides
            "use_chat": use_chat,
            "return_meta": return_meta,
        }
        try:
            resp = requests.post(OLLAMA_EXTERNAL_URL, json=payload, timeout=300, verify=False)
            resp.raise_for_status()
            answer, thinking = separate_thinking(resp.json().get("answer", ""))
            return answer.strip()
        except Exception as e:
            raise RuntimeError(f"LLM query failed (external): {e}")

    # --- Local Ollama ---
    if use_chat:
        # Chat endpoint: feed messages; result is an AIMessage with `.content`
        llm = ChatOllama(model=model_name, base_url=model_loc)
        messages = [
            SystemMessage(content=system_prompt or ""),
            HumanMessage(content=user_input),
        ]
        response = llm.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
    else:
        # Generate endpoint: feed a single string; result is a plain string
        llm = OllamaLLM(model=model_name, base_url=model_loc)
        prompt = f"{(system_prompt or '').strip()}\n\n{user_input}" if system_prompt else user_input
        response = llm.invoke(prompt)
        text = response if isinstance(response, str) else str(response)
        
    answer, thinking = separate_thinking(text or "")
    return answer.strip()
