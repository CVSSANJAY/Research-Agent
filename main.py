import argparse
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()
from agent import ResearchAgent  # noqa: E402
from pdf_report import markdown_to_pdf  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Research agent: topic -> validated, cited brief (PDF)")
    p.add_argument("topic")
    p.add_argument("--provider", default=os.getenv("PROVIDER", "gemini"),
                   choices=["gemini", "groq", "openrouter", "ollama"])
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--max-sources", type=int, default=5)
    p.add_argument("--no-validate", action="store_true", help="skip stage 5 to save API calls")
    p.add_argument("--save-json", action="store_true", help="also save the raw data as JSON (for debugging)")
    args = p.parse_args()

    agent = ResearchAgent(args.provider, args.max_steps, args.max_sources)
    result = agent.run(args.topic, validate=not args.no_validate)
    print("\n" + "=" * 60 + "\n" + result["brief"])

    os.makedirs("briefs", exist_ok=True)
    slug = re.sub(r"\W+", "_", args.topic.lower())[:50]
    base = os.path.join("briefs", slug)

    markdown_to_pdf(args.topic, result["brief"], base + ".pdf")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(f"# {args.topic}\n\n{result['brief']}\n")
    saved = f"{base}.pdf and {base}.md"
    if args.save_json:
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in result.items() if k != "brief"}, f, indent=2, default=str)
        saved += f" and {base}.json"
    print(f"\nSaved {saved}")