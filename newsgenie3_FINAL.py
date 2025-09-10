#environment setup
import os, json, re, requests, datetime as dt, urllib.parse as ul, feedparser
from pathlib import Path
from dotenv import load_dotenv

#dependency for the Tavily tool
from langchain_tavily import TavilySearch

#dependencies to build Alpha Vantage tool
from typing import List, Optional, Annotated
from langchain.tools import tool

#setup for chatbot
from langchain.chat_models import init_chat_model

#for building a StateGraph for an agent
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

#for tools creation
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
NEWSAPI_BASE = "https://newsapi.org/v2"

#for streamlit UI
import streamlit as st

st.set_page_config(page_title="NewsGenie", page_icon="📰", layout="centered")

ENV_DIR = Path(__file__).resolve().parent
load_dotenv(ENV_DIR / ".env")

#setup for chatbot
llm = init_chat_model(
     model="gpt-4o-mini",
     model_provider="openai",
     temperature=0,
)

#build a class
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

# create a function to run the tools
class BasicToolNode:
    """Executes any tool_calls in the last AIMessage and returns ToolMessage(s)."""

    def __init__(self, tools: list) -> None:
        # Ensure we have objects with .name (LangChain Tool or @tool-decorated)
        self.tools_by_name = {t.name: t for t in tools}

    def __call__(self, inputs: dict):
        messages = inputs.get("messages", [])
        if not messages:
            raise ValueError("No messages found in input state.")
        last = messages[-1]

        # If no tool calls, return no update
        tcalls = getattr(last, "tool_calls", None) or []
        if not tcalls:
            return {"messages": []}

        out_msgs = []
        for call in tcalls:
            # Support both dict-style and attr-style tool calls
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
            call_id = call.get("id")   if isinstance(call, dict) else getattr(call, "id", None)

            if not name:
                continue
            tool = self.tools_by_name.get(name)
            if tool is None:
                out_msgs.append(ToolMessage(content=f"Tool '{name}' not found.", name=name, tool_call_id=call_id or ""))
                continue

            # Normalize args to a string
            if isinstance(args, dict):
                arg_str = args.get("query") or args.get("input") or json.dumps(args)
            elif args is None:
                arg_str = ""
            else:
                arg_str = str(args)

            # Invoke tool
            result = tool.invoke(arg_str)
            if not isinstance(result, str):
                try:
                    result = json.dumps(result, ensure_ascii=False)
                except Exception:
                    result = str(result)

            out_msgs.append(ToolMessage(content=result, name=name, tool_call_id=call_id or ""))

        # With add_messages, return ONLY new messages to append
        return {"messages": out_msgs}

# Debug log for tests 
DEBUG_EVENTS = []

def log_event(kind, **payload):
    DEBUG_EVENTS.append({"kind": kind, **payload})

#special functions to define Alpha Vantage Tool
def _av_news_call(query: str, limit: int = 5) -> List[dict]:
    """
    Call Alpha Vantage NEWS_SENTIMENT.
    If the query looks like a ticker (e.g., AAPL), use 'tickers'.
    Otherwise, use 'topics' (e.g., 'electric vehicles', 'semiconductors').
    """
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key or api_key == "YOUR_KEY_HERE":
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set in environment.")

    is_ticker = bool(re.fullmatch(r"[A-Z\.]{1,7}", query.strip()))
    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": api_key,
        "sort": "LATEST",
        "limit": str(limit),
    }
    if is_ticker:
        params["tickers"] = query.strip().upper()
    else:
        # Alpha Vantage supports predefined topics (like 'technology', 'financial_markets', etc.).
        # Free-form topics also work reasonably for broader themes.
        params["topics"] = query.strip()

    r = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if "feed" not in data:
        # Helpful error surface
        raise RuntimeError(f"Alpha Vantage response did not include 'feed'. Payload: {data}")
    return data["feed"]

def _format_news(feed: List[dict], max_items: int = 5) -> str:
    lines = []
    for item in feed[:max_items]:
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        ts = item.get("time_published", "")
        # time_published is like '20250907T123000'
        try:
            ts_fmt = dt.datetime.strptime(ts, "%Y%m%dT%H%M%S").isoformat(sep=" ")
        except Exception:
            ts_fmt = ts
        # Optional: brief sentiment on primary ticker, if present
        tickers = item.get("ticker_sentiment", [])
        sent_note = ""
        if tickers:
            # pick the strongest magnitude for a quick glance
            best = max(tickers, key=lambda t: abs(float(t.get("relevance_score", 0.0))))
            sent_note = f" [{best.get('ticker','')}: {best.get('ticker_sentiment_label','N/A')}]"
        lines.append(f"- {title}{sent_note} ({ts_fmt})\n  {url}")
    return "\n".join(lines) if lines else "No headlines returned."

@tool("alpha_vantage_news", return_direct=False)
def alpha_vantage_news_tool(query: str) -> str:
    """Finance news via Alpha Vantage NEWS_SENTIMENT. Input: a ticker (e.g., 'AAPL') or a topic (e.g., 'electric vehicles'). Returns latest headlines."""
    feed = _av_news_call(query, limit=5)
    return _format_news(feed, max_items=5)

def _format_articles(articles: List[dict], max_items: int = 8) -> str:
    lines = []
    for a in articles[:max_items]:
        title = a.get("title") or a.get("name") or "Untitled"
        src   = (a.get("source") or {}).get("name") or a.get("source") or "Unknown source"
        url   = a.get("url") or a.get("link") or ""
        ts    = a.get("publishedAt") or a.get("pubDate") or a.get("published") or ""
        lines.append(f"- {title} — {src} ({ts})\n  {url}")
    return "\n".join(lines) if lines else "No headlines returned."

def _newsapi_top_headlines(category: Optional[str]=None, q: Optional[str]=None,
                           page_size: int=10, country: Optional[str]="us") -> List[dict]:
    """NewsAPI top-headlines; category is optional, and we don’t force it if it hurts recall."""
    api_key = os.environ.get("NEWSAPI_API_KEY")
    if not api_key:
        return []
    params = {"apiKey": api_key, "pageSize": page_size}
    if q: params["q"] = q
    if category: params["category"] = category
    # If you find 'country' too restrictive, set to None above
    if country: params["country"] = country
    r = requests.get(f"{NEWSAPI_BASE}/top-headlines", params=params, timeout=20)
    try:
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "ok":
            return data.get("articles", []) or []
    except Exception:
        pass
    return []

def _google_news_rss(query: str, n: int = 8) -> List[dict]:
    """Google News RSS fallback; broad coverage, no key required."""
    url = f"https://news.google.com/rss/search?q={ul.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    out = []
    for e in feed.entries[:n]:
        out.append({
            "title": e.get("title"),
            "url": e.get("link"),
            "source": {"name": getattr(e, "source", {}).get("title", "Google News")},
            "publishedAt": e.get("published", e.get("updated", "")),
        })
    return out

def _ddg_news(query: str, max_results: int = 8) -> List[dict]:
    """DuckDuckGo News fallback; no key required."""
    try:
        from duckduckgo_search import DDGS
    except Exception:
        return []
    out = []
    with DDGS() as ddgs:
        for item in ddgs.news(query, max_results=max_results):
            out.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "source": {"name": item.get("source") or "DuckDuckGo"},
                "publishedAt": item.get("date", ""),
            })
    return out

def _sports_query_boost(q: str) -> str:
    # Boost recall with common sports words; harmless if already present
    hints = ["NFL","NBA","MLB","NHL","MLS","game","score","schedule","trades","injury","coach","team"]
    maybe = " ".join(hints[:5])
    return f"{q} {maybe}"

def _tech_query_boost(q: str) -> str:
    hints = ["technology","AI","software","hardware","chip","semiconductor","cloud","cybersecurity"]
    maybe = " ".join(hints[:5])
    return f"{q} {maybe}"

def _three_stage_search(primary_fn, fallback_queries: List[str]) -> List[dict]:
    """Run a primary function first, then RSS, then DDG with given queries until we get hits."""
    # Primary (NewsAPI)
    articles = primary_fn()
    if articles:
        return articles
    # Google News RSS (try variants)
    for q in fallback_queries:
        rss = _google_news_rss(q)
        if rss:
            return rss
    # DuckDuckGo News (try variants)
    for q in fallback_queries:
        ddg = _ddg_news(q)
        if ddg:
            return ddg
    return []

#define tools based on newsAPI
@tool("sports_news", return_direct=False)
def sports_news_tool(query: str) -> str:
    """Sports news: tries NewsAPI sports top-headlines, then Google News RSS, then DuckDuckGo News."""
    q = query.strip()
    boosted = _sports_query_boost(q)
    articles = _three_stage_search(
        primary_fn=lambda: _newsapi_top_headlines(category="sports", q=q, page_size=10, country="us"),
        fallback_queries=[q, boosted]
    )
    return _format_articles(articles, max_items=8)

@tool("tech_news", return_direct=False)
def tech_news_tool(query: str) -> str:
    """Tech news: tries NewsAPI tech top-headlines, then Google News RSS, then DuckDuckGo News."""
    q = query.strip()
    boosted = _tech_query_boost(q)
    # Tip: removing the category here can sometimes improve recall on NewsAPI
    articles = _three_stage_search(
        primary_fn=lambda: _newsapi_top_headlines(category="technology", q=q, page_size=10, country="us"),
        fallback_queries=[q, boosted]
    )
    return _format_articles(articles, max_items=8)

#define the tools
webtool = TavilySearch(max_results=3)
tools = [webtool, alpha_vantage_news_tool, sports_news_tool, tech_news_tool]
llm_with_tools = llm.bind_tools(tools)

# --- Category routing + category-specific chatbots ---

# Coarse keyword router (fast + deterministic).
FIN_WORDS    = r"(stock|stocks|ticker|earnings|ipo|sec|fed|dividend|market|finance|financial|nasdaq|nyse|dow|s&p|bond|alpha vantage|ticker:[A-Z]{1,6})"
SPORTS_WORDS = r"(nfl|nba|mlb|nhl|mls|premier league|broncos|yankees|lakers|espn|odds|bet|draft|fantasy|game|score|schedule|player|coach|team)"
TECH_WORDS   = r"(ai|gpt|openai|nvidia|apple|microsoft|google|meta|semiconductor|chip|gpu|cloud|saas|startup|cyber|security|software|hardware|tech|technology)"

def _last_user_text(state: State) -> str:
    msgs = state["messages"]
    for m in reversed(msgs):
        role = getattr(m, "type", None) or getattr(m, "role", None)
        if role in ("human", "user"):
            return (m.content or "").lower()
    return (msgs[-1].content or "").lower()

def category_router(state: State) -> str:
    text = _last_user_text(state)
    if re.search(FIN_WORDS, text):
        return "finance"
    if re.search(SPORTS_WORDS, text):
        return "sports"
    if re.search(TECH_WORDS, text):
        return "tech"
    return "general"

# Each category chatbot is an LLM bound only to its relevant tools
finance_agent = llm.bind_tools([alpha_vantage_news_tool])
sports_agent  = llm.bind_tools([sports_news_tool])
tech_agent    = llm.bind_tools([tech_news_tool])
general_agent = llm.bind_tools([webtool])

def make_chatbot(bound_agent):
    def _chat(state: State):
        ai_msg = bound_agent.invoke(state["messages"])
        return {"messages": [ai_msg]}
    return _chat

chatbot_finance = make_chatbot(finance_agent)
chatbot_sports  = make_chatbot(sports_agent)
chatbot_tech    = make_chatbot(tech_agent)
chatbot_general = make_chatbot(general_agent)

# tool routing: return 'tools' if the last AI message requested tools, else END

def route_tools(state: State):
    # Be defensive about state shape
    msgs = state.get("messages", []) if isinstance(state, dict) else state
    if not msgs:
        return END
    ai = msgs[-1]

    # 1) Standard tool_calls attribute
    if getattr(ai, "tool_calls", None):
        return "tools"

    # 2) Providers that stash in additional_kwargs
    ak = getattr(ai, "additional_kwargs", {}) or {}
    if isinstance(ak, dict) and (ak.get("tool_calls") or ak.get("function_call")):
        return "tools"

    # 3) Some providers return structured chunks in .content
    content = getattr(ai, "content", None)
    if isinstance(content, list) and any(
        isinstance(x, dict) and x.get("type") in {"tool_use", "tool_call"} for x in content
    ):
        return "tools"

    return END

# Build the graph with combined subject routing

graph_builder = StateGraph(State)

# Nodes
graph_builder.add_node("router", lambda state: {})      # returns a dict
graph_builder.add_node("chatbot_finance", chatbot_finance)
graph_builder.add_node("chatbot_sports",  chatbot_sports)
graph_builder.add_node("chatbot_tech",    chatbot_tech)
graph_builder.add_node("chatbot_general", chatbot_general)

# Single Tools node that knows about all tools (executes only what's requested)
tool_node = BasicToolNode(tools=[webtool, alpha_vantage_news_tool, sports_news_tool, tech_news_tool])
graph_builder.add_node("tools", tool_node)

# Edges
graph_builder.add_edge(START, "router")

graph_builder.add_conditional_edges(
    "router",
    category_router,
    {
        "finance": "chatbot_finance",
        "sports": "chatbot_sports",
        "tech": "chatbot_tech",
        "general": "chatbot_general",
    },
)

# From each chatbot, go to tools (if requested) or END
graph_builder.add_conditional_edges("chatbot_finance", route_tools, {"tools": "tools", END: END})
graph_builder.add_conditional_edges("chatbot_sports",  route_tools, {"tools": "tools", END: END})
graph_builder.add_conditional_edges("chatbot_tech",    route_tools, {"tools": "tools", END: END})
graph_builder.add_conditional_edges("chatbot_general", route_tools, {"tools": "tools", END: END})

# After tools execute, loop back to router (multi-turn)
graph_builder.add_edge("tools", "router")

graph = graph_builder.compile()
graph

# --- Streamlit UI (single-input, auto-router) ---

st.title("📰 NewsGenie — Finance • Sports • Tech • General News")

# Initialize chat history once
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Clear chat
with st.sidebar:
    if st.button("🧹 Clear chat"):
        st.session_state["messages"] = []
        st.rerun()

# Render history (hide tool messages)
for m in st.session_state["messages"]:
    if isinstance(m, HumanMessage):
        with st.chat_message("user"):
            st.write(m.content)
    elif isinstance(m, ToolMessage):
        continue  # hide raw tool output
    else:  # AIMessage or other assistant types
        with st.chat_message("assistant"):
            st.write(getattr(m, "content", str(m)))

# Single input box
prompt = st.chat_input("Ask for headlines (e.g., 'Latest headlines for AAPL', 'Denver Broncos news', 'NVIDIA GPUs')")

if prompt:
    try:
        # Add user turn
        st.session_state["messages"].append(HumanMessage(content=prompt))

        # One agent turn (router -> chatbot_* -> tools? -> router)
        with st.spinner("Thinking..."):
            out = graph.invoke({"messages": st.session_state["messages"]})

        # Append only the new messages (State uses add_messages)
        new_msgs = out["messages"][len(st.session_state["messages"]):]
        st.session_state["messages"].extend(new_msgs)

        # Render the newly added assistant/tool messages
        for m in new_msgs:
            if isinstance(m, ToolMessage):
                continue  # hide raw tool output
            with st.chat_message("assistant"):
                st.write(getattr(m, "content", str(m)))

    except Exception as e:
        st.error(f"Error: {e}")





