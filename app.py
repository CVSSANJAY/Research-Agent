"""Streamlit web app for the research agent.  Run locally: streamlit run app.py"""
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("CALL_DELAY", "4")        # stay under free-tier per-minute limits

from agent import ResearchAgent          # noqa: E402
from llm import PROVIDERS                # noqa: E402
from pdf_report import markdown_to_pdf   # noqa: E402

FREE_RUNS = int(os.getenv("FREE_RUNS", "2"))    # runs per visitor session on the owner's key

st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="wide")
st.session_state.setdefault("runs", 0)


def default_key(provider: str) -> str:
    """The owner's key from Streamlit secrets or the environment (never shown to visitors)."""
    name = PROVIDERS[provider][1]
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, "")


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Provider", ["gemini", "groq"])
    user_key = st.text_input(f"Your {provider} API key (optional)", type="password",
                             help="Used only for this session and never stored. "
                                  "Without it, a small number of demo runs is allowed.")
    max_sources = st.slider("Max sources", 2, 6, 4)
    validate = st.checkbox("Validate claims", value=True,
                           help="Checks every claim against its cited source. Slower, uses more API calls.")
    st.caption("Pipeline: search and read, analyze, compare, write, validate.")

# ---------------- main page ----------------
st.title("🔎 Research Agent")
st.write("Enter a topic. The agent searches the web, reads sources, compares them, writes a cited "
         "brief, and checks each claim against the source it cites.")

topic = st.text_input("Topic", max_chars=200,
                      placeholder="e.g. how CRISPR gene editing works and its main applications")
go = st.button("Research", type="primary")

if go:
    key = user_key.strip() or default_key(provider)
    using_owner_key = not user_key.strip()
    topic_clean = " ".join(topic.split())

    if len(topic_clean) < 3:
        st.error("Please enter a topic (at least 3 characters).")
    elif not key:
        st.error("Enter your API key in the sidebar to run the agent.")
    elif using_owner_key and st.session_state.runs >= FREE_RUNS:
        st.error(f"Demo limit of {FREE_RUNS} runs reached for this session. "
                 "Add your own API key in the sidebar to continue.")
    else:
        if using_owner_key:
            st.session_state.runs += 1
        with st.status("Researching... this takes 1-3 minutes.", expanded=True) as status:
            box, lines = st.empty(), []

            def on_log(msg: str):
                lines.append(msg)
                box.code("\n".join(lines[-14:]), language=None)

            try:
                agent = ResearchAgent(provider, max_steps=10, max_sources=max_sources,
                                      verbose=False, api_key=key, on_log=on_log)
                result = agent.run(topic_clean, validate=validate)
                with tempfile.TemporaryDirectory() as d:
                    path = os.path.join(d, "brief.pdf")
                    markdown_to_pdf(topic_clean, result["brief"], path)
                    with open(path, "rb") as f:
                        pdf_bytes = f.read()
                st.session_state["result"] = {"topic": topic_clean, "data": result, "pdf": pdf_bytes}
                status.update(label="Done", state="complete", expanded=False)
            except Exception as e:                       # show a short message, never the key
                status.update(label="Failed", state="error")
                st.error(f"Something went wrong: {str(e)[:300]}")

# ---------------- results (kept in session so downloads don't wipe them) ----------------
res = st.session_state.get("result")
if res:
    data, claims = res["data"], res["data"]["claims"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Sources read", len(data["sources"]))
    if claims is not None:
        flagged = sum(1 for c in claims if c["status"] != "supported")
        c2.metric("Claims checked", len(claims))
        c3.metric("Claims flagged", flagged)
    d1, d2, _ = st.columns([1, 1, 4])
    d1.download_button("Download PDF", res["pdf"], file_name="research_brief.pdf", mime="application/pdf")
    d2.download_button("Download Markdown", f"# {res['topic']}\n\n{data['brief']}\n",
                       file_name="research_brief.md", mime="text/markdown")
    st.divider()
    st.markdown(data["brief"])