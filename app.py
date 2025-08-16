import os
import streamlit as st
import pandas as pd
import tempfile
from typing import Optional
from agno.tools import tool
from agno.models.groq import Groq
from agno.agent import Agent
from agno.tools.googlesearch import GoogleSearchTools as SearchToolClass
from agno.tools.hackernews import HackerNewsTools as HackerNewsToolClass
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@tool(
    name="analyze_air_quality_csv",
    show_result=True,
    stop_after_tool_call=True,
)



def analyze_air_quality_csv(file_path: str) -> str:
    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    date_col = None
    for cand in ["date", "timestamp", "day", "datetime"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(by=date_col)

    pollutants = [c for c in ["pm25", "pm2_5", "pm10", "no2", "so2", "co", "o3", "aqi"] if c in df.columns]
    desc = [f"Rows: {len(df)}"]
    if date_col and df[date_col].notna().any():
        start = df[date_col].min()
        end = df[date_col].max()
        desc.append(f"Date range: {start.date() if pd.notna(start) else 'N/A'} → {end.date() if pd.notna(end) else 'N/A'}")

    if pollutants:
        means = df[pollutants].mean(numeric_only=True).to_dict()
        means_str = "; ".join([f"{k}={v:.2f}" for k, v in means.items()])
        desc.append(f"Means: {means_str}")

        if date_col and df[date_col].notna().any() and len(df) >= 8:
            q = len(df) // 4
            head = df.head(q)
            tail = df.tail(q)
            trend_msgs = []
            for pol in pollutants:
                try:
                    h, t = head[pol].mean(), tail[pol].mean()
                    if pd.notna(h) and pd.notna(t):
                        delta = t - h
                        if abs(delta) < 1e-6:
                            continue
                        direction = "decreasing" if delta < 0 else "increasing"
                        trend_msgs.append(f"{pol}: {direction} (~{delta:.2f})")
                except Exception:
                    pass
            if trend_msgs:
                desc.append("Simple trends: " + "; ".join(trend_msgs))
    else:
        desc.append("No standard pollutant columns found. Showing dataframe head:")
        desc.append(df.head(5).to_string(index=False))

    return "\n".join(desc)


def groq_model(model_id: str = "qwen/qwen3-32b") -> Groq:
    return Groq(id=model_id)


def make_news_analyst() -> Agent:
    return Agent(
        name="News Analyst",
        role="Finds recent city-level sustainability initiatives and green projects (past year).",
        model=groq_model(),
        tools=[SearchToolClass()],
        instructions="Cite sources and prefer official city pages or reputable news.",
        markdown=True,
    )


def make_policy_reviewer() -> Agent:
    return Agent(
        name="Policy Reviewer",
        role="Summarizes municipal and regional sustainability policies and recent updates.",
        model=groq_model(),
        tools=[SearchToolClass()],
        instructions=(
            "Prefer .gov, city council, and official policy PDFs/pages; summarize key actions, dates, and status. "
            "Cite sources."
        ),
        markdown=True,
    )


def make_innovations_scout() -> Agent:
    tools = []
    if HackerNewsToolClass:
        tools.append(HackerNewsToolClass())
    tools.append(SearchToolClass()) 
    return Agent(
        name="Innovations Scout",
        role="Finds innovative urban sustainability tech and pilots relevant to cities.",
        model=groq_model(),
        tools=tools,
        instructions="Return concrete, recent examples and include links.",
        markdown=True,
    )


def make_data_analyst() -> Agent:
    return Agent(
        name="Data Analyst",
        role="Analyzes air-quality CSVs and summarizes trends using the custom pandas tool.",
        model=groq_model(),
        tools=[analyze_air_quality_csv],
        instructions=(
            "When analyzing, be concise. Report rows, date range, means and obvious trends. "
            "If columns are unknown, describe what you can infer."
        ),
        markdown=True,
    )


def make_synthesizer() -> Agent:
    return Agent(
        name="Proposal Synthesizer",
        role="Combines agent outputs into a unified sustainability proposal for a Smart City Council.",
        model=groq_model(),
        instructions=(
            "Produce a clean, skimmable proposal with sections: Executive Summary, Recent Initiatives, "
            "Policy Landscape, Data Insights, Innovation Opportunities, and Next Steps. Use bullets and keep it actionable."
        ),
        markdown=True,
    )


def run_news(city: str) -> str:
    agent = make_news_analyst()
    prompt = (
        f"Find city-level sustainability projects in the past 12 months for {city}. "
        "Focus on official announcements, pilots, or deployments; include 3–6 examples and cite sources."
    )
    resp = agent.run(prompt)
    return str(resp.content) if hasattr(resp, "content") else str(resp)


def run_policy(city: str) -> str:
    agent = make_policy_reviewer()
    prompt = (
        f"Summarize recent government/city council sustainability policies for {city}. "
        "Include dates, status, and links; keep it to 6–10 bullet points."
    )
    resp = agent.run(prompt)
    return str(resp.content) if hasattr(resp, "content") else str(resp)



def run_innovations(city: str) -> str:
    agent = make_innovations_scout()
    prompt = (
        f"Find innovative urban sustainability technologies relevant to {city} (or similar cities). "
        "Include pilots, startups, and academic demos from the last 18 months with source links."
    )
    resp = agent.run(prompt)
    return str(resp.content) if hasattr(resp, "content") else str(resp)


def run_data(csv_path: str) -> str:
    agent = make_data_analyst()
    # The tool is stop_after_tool_call=True, so just call it explicitly:
    resp = agent.run(f"analyze the CSV at path '{csv_path}' using analyze_air_quality_csv")
    return str(resp.content) if hasattr(resp, "content") else str(resp)


def synthesize(news: str, policy: str, data: str, innovation: str, city: str) -> str:
    agent = make_synthesizer()
    prompt = (
        f"City: {city}\n\n"
        f"=== News Analyst ===\n{news}\n\n"
        f"=== Policy Reviewer ===\n{policy}\n\n"
        f"=== Data Analyst ===\n{data}\n\n"
        f"=== Innovations Scout ===\n{innovation}\n\n"
        "Combine the above into a single proposal with the required sections and actionable next steps. "
        "Avoid duplicating links; keep citations inline."
    )
    resp = agent.run(prompt)
    return str(resp.content) if hasattr(resp, "content") else str(resp)


st.set_page_config(page_title="Mission Sustainability – Agno Agents", page_icon="🌍", layout="wide")
st.title("🌍 Multi-Agent Task Force: Mission Sustainability")

with st.sidebar:
    st.header("Run Mode")
    mode = st.radio("Choose mode:", ["Single Agent", "Full Task Force"])
    city = st.text_input("City / Region focus", value="Lahore, Pakistan")
    st.caption("Tip: Use a specific city for better results.")

    st.markdown("---")
    st.subheader("Data Analyst")
    uploaded = st.file_uploader("Upload air-quality CSV (optional but recommended)", type=["csv"])
    sample_note = st.checkbox("Use a tiny in-memory sample if no file uploaded", value=True)

col_l, col_r = st.columns([1, 1])

if mode == "Single Agent":
    with col_l:
        agent_choice = st.selectbox("Select an agent:", ["News Analyst", "Policy Reviewer", "Innovations Scout", "Data Analyst"])
    with col_r:
        go = st.button("Run Agent")
    if go:
        if agent_choice == "News Analyst":
            st.subheader("🗞️ News Analyst")
            st.write(run_news(city))
        elif agent_choice == "Policy Reviewer":
            st.subheader("📜 Policy Reviewer")
            st.write(run_policy(city))
        elif agent_choice == "Innovations Scout":
            st.subheader("💡 Innovations Scout")
            st.write(run_innovations(city))
        else:
            st.subheader("📊 Data Analyst")
            if uploaded is None:
                if sample_note:
                    # Create a tiny temp CSV for demo
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                    tmp.write(
                        b"date,pm25,pm10,no2\n"
                        b"2025-01-01,65,118,40\n"
                        b"2025-03-01,58,100,38\n"
                        b"2025-05-01,50,92,35\n"
                        b"2025-07-01,44,85,32\n"
                    )
                    tmp.flush()
                    csv_path = tmp.name
                else:
                    st.warning("Please upload a CSV or check the sample option.")
                    st.stop()
            else:
                # Save the uploaded file
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                tmp.write(uploaded.read())
                tmp.flush()
                csv_path = tmp.name
            st.write(run_data(csv_path))

else:
    go_all = st.button("🚀 Run Full Task Force")
    if go_all:
        st.subheader("🗞️ News Analyst")
        news = run_news(city)
        st.write(news)

        st.subheader("📜 Policy Reviewer")
        policy = run_policy(city)
        st.write(policy)

        st.subheader("💡 Innovations Scout")
        innovations = run_innovations(city)
        st.write(innovations)

        st.subheader("📊 Data Analyst")
        if uploaded is None:
            # Use a small in-memory sample if needed
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.write(
                b"date,pm25,pm10,no2\n"
                b"2025-01-01,65,118,40\n"
                b"2025-03-01,58,100,38\n"
                b"2025-05-01,50,92,35\n"
                b"2025-07-01,44,85,32\n"
            )
            tmp.flush()
            csv_path = tmp.name
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.write(uploaded.read())
            tmp.flush()
            csv_path = tmp.name
        data_summary = run_data(csv_path)
        st.write(data_summary)

        st.subheader("🧩 Combined Sustainability Proposal")
        proposal = synthesize(news, policy, data_summary, innovations, city)
        st.write(proposal)
