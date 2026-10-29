"""
Totoro — Intermediate Conversational Mental Health RAG Companion
================================================================
A Streamlit-powered conversational AI companion designed for individuals seeking
mental health support, psychoeducation, and evidence-grounded coping strategies.

Key Architectural Capabilities:
  1. Conversational Reasoning & Inquiry: Does not just dump static summaries.
     Engages in genuine dialogue, asks thoughtful clarifying questions, and reasons
     through the user's emotional experience.
  2. Grounded in the Knowledge Base: Directly references specific chapters, concepts,
     and tools from the mental health database (CBT, 4-7-8, grounding, behavioral activation).
  3. Context-Aware Query Reformulation: History-aware search query extraction so follow-up
     pronouns ("it", "that technique", "what if that happens at night") retrieve accurate context.
  4. Real-time Safety Guardrails: Immediate priority intervention protocols for crisis / self-harm.
  5. Transparent Provenance: Every response links back to retrieved database chapters with inspectable excerpts.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
import streamlit as st

# ── LangChain Core & OpenAI ──────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

# ── Configuration & Paths ────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent
ENV_PATH = APP_DIR / ".env"
GUIDE_PATH = APP_DIR / "mental_health_guide.txt"

load_dotenv(dotenv_path=ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "langchain-vectordb")
NAMESPACE = "mental-health"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536

# ── Crisis Detection Patterns ────────────────────────────────────────────────
CRISIS_PATTERNS = [
    r"\b(suicid(e|al)|kill myself|end my life|want to die|don't want to live)\b",
    r"\b(self[- ]?harm|cutting myself|hurt myself|overdose|take all my pills)\b",
    r"\b(can't go on anymore|better off dead|no reason to live)\b",
    r"\b(jump off|hang myself|shoot myself)\b",
]

CRISIS_BANNER_HTML = """
<div style="background: linear-gradient(135deg, #7f1d1d, #991b1b); border-radius: 12px; padding: 1.25rem 1.5rem; color: #ffffff; margin-bottom: 1.5rem; border: 1px solid #f87171; box-shadow: 0 4px 12px rgba(239,68,68,0.2);">
  <h3 style="margin-top:0; color:#ffffff; font-size:1.15rem; display:flex; align-items:center; gap:8px;">
    🚨 Immediate Crisis & Safety Support (24/7, Free & Confidential)
  </h3>
  <p style="font-size:0.92rem; margin-bottom:0.8rem; line-height:1.5;">
    If you are in severe distress, feeling hopeless, or having thoughts of self-harm, please connect with someone who can support you right now. You are not alone:
  </p>
  <ul style="font-size:0.9rem; margin-bottom:0.8rem; line-height:1.6; padding-left:1.2rem;">
    <li><b>988 Suicide & Crisis Lifeline:</b> Call or text <b>988</b> (US & Canada, 24/7)</li>
    <li><b>Crisis Text Line:</b> Text <b>HOME</b> to <b>741741</b> (24/7 text support)</li>
    <li><b>The Trevor Project (LGBTQ youth):</b> Call <b>1-866-488-7386</b> or text <b>START to 678-678</b></li>
    <li><b>International Help:</b> Visit <a href="https://findahelpline.com/" target="_blank" style="color:#fef08a; text-decoration:underline;">findahelpline.com</a> for confidential support worldwide</li>
  </ul>
  <p style="font-size:0.85rem; margin:0; opacity:0.9;">
    <i>Please reach out to a professional or trusted loved one. Your life and well-being matter deeply.</i>
  </p>
</div>
"""

# ── System Prompts ───────────────────────────────────────────────────────────
REWRITE_PROMPT = """You are an intelligent query reformulation engine for a mental health support RAG application.
Given a multi-turn conversation history and the latest user query:
1. Identify if the user's query refers back to previous turns (using pronouns like "it", "that", "the second one", "what about at night", etc.).
2. Reformulate the query into a clear, standalone search query containing the core mental health, psychoeducational, or coping keywords.
3. If the user query is already self-contained, return it as is.
4. Return ONLY the search phrase without conversational filler, explanations, or quotes.
"""

TOTORO_COUNSELOR_PROMPT = """You are "Totoro", a deeply compassionate, empathetic, and evidence-grounded Mental Health Support Companion.
You are conversing with someone experiencing emotional distress, mental health challenges, or daily life overwhelm.

YOUR CORE OBJECTIVES:
1. Actively Conversate & Reason: Do not treat this as a one-way search engine. Listen deeply, validate the user's emotional state with warmth (e.g., "I can feel how exhausting this weight must be for you"), and help them reason through what their body/mind is experiencing.
2. Ground Strongly in the Knowledge Base: You MUST explicitly refer back to the evidence-based strategies, chapters, and concepts in the mental health database provided below. Explicitly mention the techniques by name (e.g., "In the guide's chapter on anxiety, there is an evidence-based method called the 4-7-8 breathing technique...", "In the material on depression, it emphasizes behavioral activation—taking small, low-effort steps...").
3. Ask Clarifying & Exploratory Questions: If the user's input is brief, ambiguous, or purely emotional, ask 1 or 2 gentle, thoughtful questions to understand their situation better (e.g., "When this wave of anxiety hits, do you feel it more physically in your chest or stomach, or is it racing catastrophic thoughts keeping you awake?"). Never sound clinical or interrogation-like—ask with genuine curiosity and warmth.
4. Help Mitigate Problems Step-by-Step: Break down coping mechanisms into 1 or 2 gentle, bite-sized actions they can try right now. Avoid overwhelming them with huge walls of text.
5. Strict Non-Diagnostic Boundary: You are an AI companion providing psychoeducation and supportive coping tools. You are NOT a medical doctor or licensed therapist. Never diagnose psychiatric conditions or recommend medications.
6. Safety First: If the user hints at self-harm or despair, remind them with care about 24/7 confidential help like the 988 Lifeline.

KNOWLEDGE BASE RETRIEVED CONTEXT:
==================================================
{context}
==================================================
"""

# ── Pinecone & Knowledge Base Sync ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_vector_store() -> Tuple[PineconeVectorStore, int]:
    """Connect to Pinecone index and ensure the 'mental-health' namespace is populated."""
    if not PINECONE_API_KEY:
        st.error("PINECONE_API_KEY is missing from .env. Please configure it.")
        st.stop()

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(INDEX_NAME)
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=NAMESPACE,
    )

    stats = index.describe_index_stats()
    namespaces = stats.get("namespaces", {})
    existing_count = namespaces.get(NAMESPACE, {}).get("vector_count", 0)

    # If already indexed, reuse immediately
    if existing_count > 0:
        return vector_store, existing_count

    # If not indexed yet, load and split
    if not GUIDE_PATH.exists():
        st.error(f"Knowledge base document not found at {GUIDE_PATH}")
        st.stop()

    with open(GUIDE_PATH, "r", encoding="utf-8") as f:
        full_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    raw_chunks = splitter.split_text(full_text)
    documents = []

    current_chapter = "General Mental Health Support"
    for i, chunk in enumerate(raw_chunks):
        match = re.search(r"##\s+(Chapter\s+\d+:\s+[^\n]+)", chunk)
        if match:
            current_chapter = match.group(1).strip()
        doc = Document(
            page_content=chunk,
            metadata={
                "id": f"mh-chunk-{i}",
                "source": "Mental Health Support Guide",
                "chapter": current_chapter,
                "chunk_index": i,
            },
        )
        documents.append(doc)

    vector_store.add_documents(
        documents=documents,
        ids=[d.metadata["id"] for d in documents],
    )
    return vector_store, len(documents)


def check_crisis(text: str) -> bool:
    """Pre-retrieval check for acute self-harm or suicidal ideation."""
    for pattern in CRISIS_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    """Instantiate ChatOpenAI client."""
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is missing from .env.")
        st.stop()
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        openai_api_key=OPENAI_API_KEY,
    )


def contextualize_user_query(
    user_input: str,
    chat_history: List[Dict[str, str]],
    llm: ChatOpenAI
) -> str:
    """Rewrite follow-up queries using conversation history for precision vector search."""
    if not chat_history:
        return user_input

    # Convert last 4 messages to BaseMessage
    history_messages: List[BaseMessage] = []
    for msg in chat_history[-6:]:
        if msg["role"] == "user":
            history_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_messages.append(AIMessage(content=msg["content"]))

    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITE_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()

    try:
        rewritten = chain.invoke({
            "chat_history": history_messages,
            "input": user_input,
        }).strip()
        return rewritten if rewritten else user_input
    except Exception:
        return user_input


# ── Streamlit UI Setup ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Totoro",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styles for Calming Aesthetic: Pure White Background & Green Buttons
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,700;1,9..40,400&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Pure White Background ── */
    [data-testid="stAppViewContainer"],
    [data-testid="stApp"],
    .main,
    .block-container,
    section[data-testid="stMain"],
    header[data-testid="stHeader"],
    footer {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }

    /* Sidebar Background: Crisp White with Subtle Border */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    /* ── Forest Green Buttons ── */
    .stButton > button,
    div[data-testid="stButton"] > button,
    button[kind="primary"],
    button[kind="secondary"] {
        background-color: #2d6a4f !important;
        color: #ffffff !important;
        border: 1px solid #2d6a4f !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 2px 4px rgba(45, 106, 79, 0.15) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stButton > button:hover,
    div[data-testid="stButton"] > button:hover {
        background-color: #1b4332 !important;
        border-color: #1b4332 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(27, 67, 50, 0.28) !important;
        transform: translateY(-1px) !important;
    }

    .stButton > button:active,
    div[data-testid="stButton"] > button:active {
        background-color: #081c15 !important;
        transform: translateY(0px) !important;
    }

    /* Green Chat Input Submit Button */
    [data-testid="stChatInputSubmitButton"] > button {
        background-color: #2d6a4f !important;
        color: #ffffff !important;
        border-radius: 50% !important;
        border: none !important;
    }

    [data-testid="stChatInputSubmitButton"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    /* Chat Input Focus Ring */
    [data-testid="stChatInput"] textarea:focus {
        border-color: #2d6a4f !important;
        box-shadow: 0 0 0 1px #2d6a4f !important;
    }

    /* Header Container */
    .totoro-hero {
        background: linear-gradient(135deg, rgba(45, 106, 79, 0.08), rgba(82, 183, 136, 0.04));
        border: 1px solid rgba(82, 183, 136, 0.3);
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.5rem;
    }
    .totoro-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2d6a4f;
        margin: 0 0 0.4rem 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .totoro-subtitle {
        font-size: 0.95rem;
        color: #475569;
        margin: 0;
        line-height: 1.5;
    }
    .totoro-disclaimer {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.6rem;
        border-top: 1px dashed rgba(82, 183, 136, 0.3);
        padding-top: 0.5rem;
    }
    
    /* Source Pill Badge */
    .source-tag {
        display: inline-block;
        background: rgba(45, 106, 79, 0.1);
        color: #2d6a4f;
        border: 1px solid rgba(45, 106, 79, 0.2);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }

    /* Chat Message Bubbles on White Background */
    [data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    }

    /* Assistant Message with Calming Sage Accent */
    div:has(> [data-testid="stChatMessageAvatarAssistant"]) {
        border-left: 3px solid #2d6a4f !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 Totoro")
    st.caption("A compassionate, evidence-grounded mental health conversation partner.")

    # Knowledge Base Health Badge
    vector_store, total_vectors = init_vector_store()
    st.markdown(f"""
    <div style="background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.25); border-radius: 8px; padding: 0.8rem; margin-bottom: 1rem;">
        <div style="font-size: 0.78rem; font-weight: 600; color: #16a34a; display: flex; align-items: center; gap: 6px;">
            <span>🟢</span> RAG Database Active
        </div>
        <div style="font-size: 0.75rem; color: #666; margin-top: 4px;">
            Index: <b>{INDEX_NAME}</b><br/>
            Namespace: <b>{NAMESPACE}</b><br/>
            Loaded Vectors: <b>{total_vectors} chunks</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 💬 Helpful Conversation Starters")
    st.caption("Click any topic below to explore with Totoro:")

    starters = [
        ("🌬️ Panic / Shallow Breathing", "I feel a panic attack coming on and my chest feels tight. What can I do right now?"),
        ("💭 Spiraling Thoughts (CBT)", "I can't stop catastrophizing about my future. How do I challenge these thoughts?"),
        ("🛌 Nighttime Racing Mind", "My brain won't turn off at night when I try to sleep. What helps?"),
        ("🔋 Low Energy & Heavy Mood", "I feel completely unmotivated, tired, and low today. Everything feels like too much."),
        ("🫂 Explaining to Loved Ones", "How do I explain to my family or friends that I'm struggling with mental health?"),
    ]

    for label, prompt_text in starters:
        if st.button(label, use_container_width=True):
            st.session_state["queued_prompt"] = prompt_text

    st.markdown("---")
    st.markdown("#### 🚨 24/7 Crisis Resources")
    with st.expander("Emergency Helplines", expanded=False):
        st.markdown("""
        - **Call/Text 988** (Suicide & Crisis Lifeline, 24/7 Free)
        - **Text HOME to 741741** (Crisis Text Line)
        - **1-866-488-7386** (The Trevor Project)
        - **[findahelpline.com](https://findahelpline.com)** (Global Directory)
        """)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Reload DB", use_container_width=True):
            st.cache_resource.clear()
            st.success("Vector cache cleared!")
            st.rerun()

# ── Main Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="totoro-hero">
    <div class="totoro-title">🌿 Totoro</div>
    <div class="totoro-subtitle">
        A thoughtful, conversational partner grounded in evidence-based mental health resources.
        Totoro explores what you are going through, asks gentle questions to understand, and guides you through practical coping exercises.
    </div>
    <div class="totoro-disclaimer">
        ⚠️ <b>Educational & Support Companion:</b> Totoro is not a doctor, licensed therapist, or crisis emergency service. If you are in immediate danger, please dial <b>988</b>.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Chat Session State ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello, I am **Totoro**. I am here to sit with you, listen without judgment, "
                "and share grounded coping practices from our mental health guide. "
                "How are you feeling right now, and what has been on your mind?"
            ),
            "sources": [],
            "search_query": None,
        }
    ]

# Render Message History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🌿"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Grounded in Knowledge Base (Query: *'{msg.get('search_query', '')}'*)"):
                    for i, doc in enumerate(msg["sources"], 1):
                        chapter = doc.metadata.get("chapter", "Support Guide")
                        preview = doc.page_content.strip()
                        st.markdown(f"**{i}. {chapter}**")
                        st.caption(preview)
                        if i < len(msg["sources"]):
                            st.divider()

# Check for queued prompt from sidebar buttons
user_input = st.chat_input("Share what you're experiencing or ask a question...")
if not user_input and st.session_state.get("queued_prompt"):
    user_input = st.session_state.pop("queued_prompt")

# ── Process User Input ───────────────────────────────────────────────────────
if user_input:
    # 1. Render User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Check Safety Guardrail (Crisis Intervention)
    if check_crisis(user_input):
        crisis_reply = (
            "I hear how much deep pain and heaviness you are carrying right now, and I want you to know "
            "that your life and safety matter immensely. You do not have to carry this alone.\n\n"
            "Please pause with me and connect with someone right now. Trained, compassionate crisis counselors "
            "are available 24/7 to support you free of charge. Call or text **988**, or text **HOME to 741741**."
        )
        st.markdown(CRISIS_BANNER_HTML, unsafe_allow_html=True)
        with st.chat_message("assistant", avatar="🌿"):
            st.markdown(crisis_reply)
        st.session_state.messages.append({
            "role": "assistant",
            "content": crisis_reply,
            "sources": [],
            "search_query": None,
            "is_crisis": True,
        })
        st.stop()

    # 3. Conversational RAG Process
    with st.chat_message("assistant", avatar="🌿"):
        with st.spinner("Totoro is reflecting on your thoughts and looking through the support guide…"):
            llm = get_llm(temperature=0.3)

            # Step A: Contextualize Query using Multi-Turn History
            reformulated_query = contextualize_user_query(
                user_input=user_input,
                chat_history=st.session_state.messages[:-1],
                llm=llm,
            )

            # Step B: Similarity Retrieval from Pinecone
            retrieved_docs = vector_store.similarity_search(reformulated_query, k=3)

            # Step C: Format Context for LLM
            context_blocks = []
            for doc in retrieved_docs:
                chapter = doc.metadata.get("chapter", "Support Section")
                context_blocks.append(f"[{chapter}]\n{doc.page_content.strip()}")
            context_str = "\n\n---\n\n".join(context_blocks)

            # Step D: Construct History Messages for Multi-Turn Reasoning
            history_messages: List[BaseMessage] = []
            for m in st.session_state.messages[:-1][-8:]:
                if m["role"] == "user":
                    history_messages.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    history_messages.append(AIMessage(content=m["content"]))

            # Step E: Generate Empathetic, Reasoning Response
            counselor_prompt = ChatPromptTemplate.from_messages([
                ("system", TOTORO_COUNSELOR_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            chain = counselor_prompt | llm | StrOutputParser()
            assistant_response = chain.invoke({
                "context": context_str,
                "chat_history": history_messages,
                "input": user_input,
            })

            # Render generated response
            st.markdown(assistant_response)

            # Show grounding expander
            if retrieved_docs:
                with st.expander(f"📚 Grounded in Knowledge Base (Query: *'{reformulated_query}'*)"):
                    for i, doc in enumerate(retrieved_docs, 1):
                        chapter = doc.metadata.get("chapter", "Support Guide")
                        preview = doc.page_content.strip()
                        st.markdown(f"**{i}. {chapter}**")
                        st.caption(preview)
                        if i < len(retrieved_docs):
                            st.divider()

    # Save to Session State
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_response,
        "sources": retrieved_docs,
        "search_query": reformulated_query,
    })
