# list_models_fixed.py
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("Set GROQ_API_KEY in your env or .env")

client = Groq(api_key=api_key)

try:
    response = client.models.list()
    # response.data is a list of Model objects
    print(f"✅  You have access to {len(response.data)} model(s).")
    for i, model in enumerate(response.data, start=1):
        # Each `model` has attributes: id, object, created, etc.
        print(f"{i:02d}. {model.id}")
except Exception as exc:
    print(f"❌  Error listing models: {exc}")