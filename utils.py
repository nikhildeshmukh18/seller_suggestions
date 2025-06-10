import requests

GROQ_API_KEY = "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"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def call_groq_llm(prompt, model="llama-3.3-70b-versatile"):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for e-commerce tasks."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    res = requests.post(GROQ_URL, headers=HEADERS, json=payload)
    res.raise_for_status()
    return res.json()['choices'][0]['message']['content']
