# LangGraph Ollama Memory Chat

A context-aware chatbot implementation using **LangGraph**, **Ollama**, and **ChromaDB**. This project demonstrates how to build a stateful agent that remembers past conversations semantically and persists session state locally.

## Features

*   **Stateful Conversation**: Uses `LangGraph` to manage the chat workflow and state transitions.
*   **Local LLM**: Powered by **Ollama** (defaulting to `llama3.2:1b`) for privacy and local execution without API keys.
*   **Semantic Memory (RAG on Chat History)**: Stores user and assistant messages in a **ChromaDB** vector database using `SentenceTransformer` embeddings.
*   **Context Retrieval**: Automatically retrieves relevant past memories based on the current user query to provide context-aware responses (e.g., asking "What did I say about SQL?" will fetch the specific previous turn).
*   **Session Persistence**: Uses **SQLite** to checkpoint the conversation state, allowing the graph to maintain short-term state across steps.

## Prerequisites

1.  **Python 3.10+**
2.  **Ollama**: Installed and running on your machine.
    *   You must pull the model used in the script:
        ```bash
        ollama pull llama3.2:1b
        ```

## Installation

1.  Clone the repository or download the source code.

2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  Install the required Python packages:
    ```bash
    pip install langgraph langchain-core ollama chromadb sentence-transformers
    ```

## Usage

1.  Ensure the Ollama server is running:
    ```bash
    ollama serve
    ```

2.  Run the chat script:
    ```bash
    python test.py
    ```

3.  **Interact**: Type your messages in the terminal.
    *   Type `exit` to quit the application.

## Architecture

The application follows a directed graph workflow defined in `test.py`:

1.  **Input Node**: Captures user text.
2.  **Save User Memory**: Embeds and stores the user's message in ChromaDB.
3.  **Retrieve Memory**: Searches ChromaDB for past messages semantically similar to the current input.
4.  **Build Prompt**: Injects retrieved memories into the system prompt as context.
5.  **Ollama Node**: Calls the local LLM to generate a response.
6.  **Save AI Memory**: Embeds and stores the AI's response for future retrieval.
7.  **Output Node**: Displays the response to the user.

## Future Improvements & Use Cases

*   **Long-term Personal Assistant**: Can be expanded to remember user preferences, tasks, and details over long periods (days/weeks).
*   **Document Q&A**: The vector store logic can be adapted to ingest external documents (PDFs, text files) to allow the bot to answer questions based on specific knowledge bases.
*   **Multi-Agent Systems**: LangGraph allows adding more specialized nodes (e.g., a web search tool or a Python code executor) to create a more capable agent.
*   **API Deployment**: The graph can be wrapped in a FastAPI service to act as a backend for a web or mobile interface.