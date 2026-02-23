

from langgraph.graph import StateGraph, MessagesState, START, END
import ollama
import sqlite3
from datetime import datetime
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- ChromaDB & embeddings ---
import chromadb
from sentence_transformers import SentenceTransformer
import uuid



MODEL_NAME = "llama3.2:1b"

CHROMA_PATH = "./chroma_memory"
COLLECTION_NAME = "chat_memory"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
chroma_collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- State ---
class AgentState(MessagesState):
    retrieved_memories: list[str]
    turn_id: int

# --- Helper to save messages to vector DB ---
def save_message_to_vector_db(thread_id: str, role: str, content: str, turn_id: int):
    embedding = embedder.encode(content).tolist()
    doc_id = f"{thread_id}_{turn_id}_{role}"

    chroma_collection.add(
        ids=[doc_id],
        documents=[content],
        embeddings=[embedding],
        metadatas=[{
            "thread_id": thread_id,
            "role": role,
            "turn_id": turn_id,
            "timestamp": datetime.utcnow().isoformat()
        }]
    )
    print(f"✅ Saved to vector DB ({role}): {content[:50]}...")

# ---------------------------
# Node 1: Input Node
# ---------------------------
def input_node(state):
    user_input = input("You: ")
    if user_input.lower() == "exit":
        return {"exit": True}

    return {"messages": [HumanMessage(content=user_input)], "exit": False, "turn_id": state.get("turn_id", 0)}

# ---------------------------
# Node 1.5: Save User Message to Memory
# ---------------------------
def save_user_to_memory_node(state, config):
    if state.get("exit", False):
        return state

    messages = state.get("messages", [])
    if not messages:
        return state

    turn_id = state.get("turn_id", 0) + 1
    state["turn_id"] = turn_id

    # Save user message
    last_msg = messages[-1]
    if isinstance(last_msg, HumanMessage):
        save_message_to_vector_db(config["configurable"]["thread_id"], "user", last_msg.content, turn_id)

    return state

# ---------------------------
# Node 2: Retrieve Memory Node
# ---------------------------
def retrieve_memory_node(state, config):
    if state.get("exit", False):
        return state

    thread_id = config["configurable"]["thread_id"]
    messages = state.get("messages", [])
    if not messages:
        state["retrieved_memories"] = []
        return state

    # Embed latest user message
    latest_msg = messages[-1]
    content = latest_msg.content if hasattr(latest_msg, "content") else latest_msg.get("content")
    print(f"\n--- [Retrieval] Searching memories for: '{content}' ---")
    query_embedding = embedder.encode(content).tolist()

    # Query vector DB
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where={"thread_id": thread_id}
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    memories = []
    for d, m in zip(docs, metas):
        memories.append(f"({m.get('role')} | turn {m.get('turn_id')}): {d}")

    print(f"Found {len(memories)} relevant memories.")
    state["retrieved_memories"] = memories
    return state

# ---------------------------
# Node 2.5: Build Prompt Node
# ---------------------------
def build_prompt_node(state):
    if state.get("exit", False):
        return state

    memories = state.get("retrieved_memories", [])
    messages = state.get("messages", [])

    # Inject memories as system message
    if memories:
        memory_context = "\n".join(f"- {m}" for m in memories)
        system_msg = f"You have access to the following relevant memories from past conversations:\n{memory_context}\nUse these to answer the user's question if relevant."
        messages = [{"role": "system", "content": system_msg}] + [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content} for m in messages]

    state["messages"] = messages
    return state

# ---------------------------
# Node 3: Ollama Node
# ---------------------------
def ollama_node(state):
    if state.get("exit", False):
        return state

    messages = state.get("messages", [])
    if not messages:
        return state

    print("\n---------** Ollama Node Received Messages **---------")
    print("Messages (truncated to last 10):")
    for msg in messages[-10:]:
        role = "user" if isinstance(msg, HumanMessage) else "assistant" if isinstance(msg, AIMessage) else msg.get("role", "unknown")
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        print(f"[{role}]: {content[:100]}")  # truncate message content to 100 chars
    print()

    # Convert LangChain messages to Ollama format
    ollama_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            ollama_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            ollama_messages.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            ollama_messages.append({"role": "system", "content": msg.content})
        elif isinstance(msg, dict):
            if "role" in msg:
                ollama_messages.append(msg)
            elif "type" in msg:
                role = "user"
                if msg["type"] == "ai": role = "assistant"
                elif msg["type"] == "system": role = "system"
                ollama_messages.append({"role": role, "content": msg.get("content", "")})

    response = ollama.chat(
        model=MODEL_NAME,
        messages=ollama_messages
    )

    # Append AI message while keeping previous messages
    state["messages"] = state.get("messages", []) + [AIMessage(content=response["message"]["content"])]
    return state

# ---------------------------
# Node 3.5: Save AI Message to Memory
# ---------------------------


def save_ai_to_memory_node(state, config):
    if state.get("exit", False):
        return state

    messages = state.get("messages", [])
    if not messages:
        return state

    last_msg = messages[-1]
    turn_id = state.get("turn_id", 0)
    if isinstance(last_msg, AIMessage):
        save_message_to_vector_db(config["configurable"]["thread_id"], "assistant", last_msg.content, turn_id)
    return state

# ---------------------------
# Node 4: Output Node
# ---------------------------
def output_node(state):
    if state.get("exit", False):
        return state
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage):
            print("Ollama:", last_msg.content)
    return state

# ---------------------------
# Build LangGraph
# ---------------------------
graph = StateGraph(AgentState)

graph.add_node("input_node", input_node)
graph.add_node("save_user_to_memory_node", save_user_to_memory_node)
graph.add_node("retrieve_memory_node", retrieve_memory_node)
graph.add_node("build_prompt_node", build_prompt_node)
graph.add_node("ollama_node", ollama_node)
graph.add_node("save_ai_to_memory_node", save_ai_to_memory_node)
graph.add_node("output_node", output_node)

# Proper node ordering
graph.add_edge(START, "input_node")
graph.add_edge("input_node", "save_user_to_memory_node")
graph.add_edge("save_user_to_memory_node", "retrieve_memory_node")
graph.add_edge("retrieve_memory_node", "build_prompt_node")
graph.add_edge("build_prompt_node", "ollama_node")
graph.add_edge("ollama_node", "save_ai_to_memory_node")
graph.add_edge("save_ai_to_memory_node", "output_node")
graph.add_edge("output_node", END)

# ---------------------------
# Persistence
# ---------------------------
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)
graph = graph.compile(checkpointer=memory)

# ---------------------------
# Interactive chat
# ---------------------------
print("Welcome to LangGraph Ollama chat! Type 'exit' to quit.\n")
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}


while True:
    result = graph.invoke({"messages": [], "turn_id": 0}, config=config)
    print("\n" + "-" * 50 + "\n")
    if result.get("exit", False):
        print("Goodbye!")
        break
