"""Print the model names your key can use. Usage: python list_models.py gemini"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()
from llm import LLM  # noqa: E402

provider = sys.argv[1] if len(sys.argv) > 1 else os.getenv("PROVIDER", "gemini")
for m in LLM(provider).client.models.list().data:
    print(m.id)