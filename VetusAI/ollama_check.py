import requests

# === Configuration ===
OLLAMA_MODELS = {
    "llama": "llama3.1", # As of July 2025, latest pulls 8B model
    "llama8b": "llama3.1:8b",
    "gemma" : "gemma3", # As of July 2025, latest pulls 4B model
    "gemma27b": "gemma3:27b",
    "gemma12b": "gemma3:12b",
    "dolphin": "dolphin3",
    "deepseek": "deepseek-r1", # As of July 2025, latest pulls 8B model
    "deepseek14b": "deepseek-r1:14b",
    "deepseek32b": "deepseek-r1:32b",
    "nomic": "nomic-embed-text",
}
OLLAMA_URL = "http://localhost:11434"


# === Utility: List pulled models ===
def list_ollama_models(loc: str) -> list:
    try:
        response = requests.get(f"{loc}/api/tags")
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to get the list of Ollama models: {e}")
        return []


# === Main Check & Pull Function ===
def check_ollama(model_name: str = OLLAMA_MODELS["llama"], model_loc: str = OLLAMA_URL) -> bool:
    # Check if Ollama server is running
    try:
        requests.get(model_loc)
        print("✅ Connected to Ollama")
    except requests.ConnectionError as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return False

    # Check available models
    models = list_ollama_models(model_loc)
    if model_name in models:
        print(f"✅ Model '{model_name}' is available.")
        return True
    else:
        print(f"⬇️ Pulling model '{model_name}'...")
        try:
            response = requests.post(f"{model_loc}/api/pull", json={"name": model_name})
            response.raise_for_status()
            print(f"✅ Successfully pulled model '{model_name}'.")
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to pull model: {e}")
            return False
