# Research Agent

Topic in, validated cited brief out.

## Pipeline
1. **Gather**: agent loop with `search_web` and `read_url`
2. **Analyze**: each page is turned into compact notes (claims, credibility, forecast or fact)
3. **Compare**: finds agreements, conflicts, single-source claims, gaps
4. **Report**: fixed structure with a citation after every factual sentence
5. **Validate**: every claim is checked against the text of the source it cites

## Setup
```
pip install -r requirements.txt
copy .env.example .env      (Windows)   # then add your key
python main.py "impact of AI on Indian agriculture"
```

Options: `--provider gemini|groq|openrouter|ollama`, `--max-sources 5`, `--no-validate`.
Run `python list_models.py gemini` to see valid model names.

Pages and searches are cached in `.cache/`, so re-running the same topic is cheap.
Results are saved to `briefs/` as .md and .json.