"""
Quick test script to verify Ollama is working
"""

import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3-4b-fast"

print("="*60)
print("Testing Ollama Connection")
print("="*60)
print()

# First check if Ollama is running
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    if response.status_code == 200:
        print("✅ Ollama is running!")
        models = response.json().get('models', [])
        print(f"📚 Found {len(models)} models")

        # Check if our model exists
        model_names = [m['name'] for m in models]
        print(f"\n🔍 Looking for: {MODEL}")
        if MODEL in model_names:
            print(f"✅ Found {MODEL}")
        else:
            print(f"❌ {MODEL} not found!")
            print(f"\nAvailable models starting with 'qwen':")
            for name in model_names:
                if 'qwen' in name.lower():
                    print(f"   - {name}")
    else:
        print(f"❌ Ollama returned status: {response.status_code}")
except Exception as e:
    print(f"❌ Cannot connect to Ollama: {e}")
    print("\nMake sure Ollama is running:")
    print("   ollama serve")
    exit(1)

print()
print("="*60)
print("Testing Text Generation")
print("="*60)
print()

# Test generation
payload = {
    "model": MODEL,
    "prompt": "Write one sentence about artificial intelligence.",
    "stream": False,
    "options": {
        "num_predict": 50,
        "temperature": 0.8,
    }
}

try:
    print(f"📝 Sending prompt to {MODEL}...")
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()

    result = response.json()
    generated = result.get('response', '')

    if generated:
        print(f"✅ Generation successful!")
        print(f"\nGenerated text:")
        print(f"   {generated}")
    else:
        print(f"❌ No text generated!")
        print(f"Response: {result}")

except Exception as e:
    print(f"❌ Generation failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*60)
