This project focused on building NewsGenie, an AI-powered assistant that unifies conversational Q&A 
with real-time news across finance, sports, and technology. The goal was to provide timely, 
trustworthy updates and quick general answers—within a single experience—addressing today’s 
fragmented and overwhelming news landscape. The solution implements category-aware routing, 
tool-use with fallbacks, and a clean Streamlit UI for interactive use. 

The program starts with a input text box for the user. The input goes to a ChatGPT chatbot who 
decides whether the query can be answered with the base LLM, or if the chatbot should call tools to 
find a suitable response. The chatbot has access to a finance, a sports, and a technology tool node 
which the chatbot can call. All of these nodes then feed to an output node, which provides a response 
to the user.
________________________________________
Project Notes

Understanding the Prompt
The CEP problem statement defined a unified assistant that 
(1) distinguishes general queries from news requests, 
(2) integrates a real-time news API plus web search, 
(3) manages workflow and error handling, and 
(4) delivers an intuitive UI with session management and category support. 
My solution maps directly to those requirements with LangGraph-based routing, multiple news tools, robust fallbacks, and a Streamlit front end. 

Tech Stack Design

•	Python for development (Jupyter for prototyping; VS Code for cleaning and editing the operable .py file, and Streamlit for the final app)
•	LangChain + LangGraph for conversation state, tool use, and routing
•	OpenAI (init_chat_model) as the base LLM - OpenAI's ChatGPT 4o Mini model
•	Tools (4 total):
o	TavilySearch for general web lookups
o	Alpha Vantage News for finance headlines
o	Sports news tool (NewsAPI → fallback to Google News RSS → fallback to DuckDuckGo News)
o	Tech news tool (same staged fallback strategy as sports)
•	python-dotenv for secret API management via .env (needs to be added to run)
•	Streamlit for the interactive UI and session history
________________________________________
Build Process

1. Environment & Dependency Setup

•	Consolidated dependencies into a clean requirements.txt with relaxed version ranges to avoid resolver conflicts (kept langchain and langchain-community in the same 0.3.x band).
•	Centralized secrets in .env (OPENAI_API_KEY, TAVILY_API_KEY, ALPHAVANTAGE_API_KEY, NEWSAPI_API_KEY) and loaded them with load_dotenv. [API secrets need to be loaded into a .env file]
•	Removed notebook-only code (e.g., get_ipython, %pip) from the rough Jupyter Notebook file using VS Code to ensure clean app runs.

2. Category-Aware Routing (LangGraph)

•	Implemented a coarse keyword router that routes user text to finance, sports, tech, or general buckets.
•	Created category-specific chatbots by binding the LLM to only the relevant tools:
o	Finance → Alpha Vantage News
o	Sports → Sports News tool
o	Tech → Tech News tool
o	General → TavilySearch
•	Compiled a graph with:
o	router → routes to the correct chatbot
o	chatbot_* → may request tools
o	tools → executes requested tool calls; returns ToolMessage(s)
o	A loop back to router for multi-turn interactions
•	Built a BasicToolNode that executes any tool_calls from the last AI message and returns only new ToolMessage objects (compatible with add_messages state updates).

3. Tools & Fallbacks
•	Alpha Vantage News for finance.
•	NewsAPI for sports/tech “top-headlines” first; if no hits, fallback to:
o	Google News RSS, then
o	DuckDuckGo News.
•	Normalized and formatted results into concise bullet summaries; added query “boosts” (domain hints, topical terms) to improve recall when fallbacks are used.

4. Streamlit UI
•	Single-input chat interface; the graph’s router decides the category automatically.
•	Session state stores the running history; only assistant replies are displayed (raw tool payloads are hidden for a clean UX).
•	Sidebar “Clear chat” resets the conversation.
•	Enforced Streamlit rules:
o	st.set_page_config(...) is the first st.* call
o	No widget/session errors (guarded initialization)
•	Mirrors the structure and clarity of the prior project’s UI section while adapting to a chat/news experience. 
________________________________________
Testing and Debugging
•	Key issues and fixes:
o	Package conflicts between langchain and langchain-community: moved to relaxed pins with aligned minor versions (≥0.3.23,<0.4).
o	Missing/incorrect extras (Alpha Vantage, NewsAPI): avoided tool-package extras by using requests directly; staged RSS/DDG fallbacks improved recall.
o	Persisting Jupyter code in the Streamlit script: removed %pip, get_ipython, and Jupyter shims to prevent runtime errors.
o	Streamlit set_page_config order: guaranteed it’s the first Streamlit call to eliminate API exceptions.
o	Tool output leak: initially showed raw ToolMessage JSON; updated UI renderers to skip tool messages and display only assistant responses.
o	.env usage: standardized .env var names (e.g., NEWSAPI_API_KEY) and verified early with friendly error messaging.
•	Functional tests:
o	Upon many early failures, I restarted the project in Jupyter Notebook and added test blocks at each stage. I tested to make sure the dependencies loaded properly, 
    I tested the router to make sure the requests were valid and going to the correct places, and I tested the outputs of each content node:
	“Latest headlines for AAPL/MSFT” → Finance path; Alpha Vantage results
	“Denver Broncos news” → Sports path; NewsAPI → RSS/DDG fallbacks
	“NVIDIA GPUs news” → Tech path; staged fallbacks if top-headlines sparse
	“What’s a node in LangGraph?” → General path; Tavily web lookup.
