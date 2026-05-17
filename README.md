# HealthPilot AI

An AI-powered medical assistant using LangGraph, LangChain, Supabase, and Streamlit.

## Features
- **Medical Safety**: Disclaimers and emergency detection.
- **RAG**: Retrieval-augmented generation using Supabase Vector Store.
- **SQL Agent**: Query database for medicines and doctors.
- **Web Search**: Fallback for general queries using Tavily.
- **Streamlit UI**: User-friendly chat interface.

## Setup

1.  **Install Python**: Ensure Python 3.9+ is installed.

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Variables**:
    The `.env` file has been created with your provided keys.
    
    *Note: For the SQL Agent to fully work (execute arbitrary SQL), you normally need the database password. Currently, it is configured in a limited mode or might require additional setup.*

4.  **Supabase Setup**:
    -   Go to your Supabase Project Dashboard.
    -   Open the **SQL Editor**.
    -   Copy the contents of `supabase_schema.sql` (found in this directory) and run it. This will create the necessary tables and the vector store function.

## Running the App

```bash
streamlit run app/main.py
```

## Architecture
-   `app/graph.py`: Defines the LangGraph workflow.
-   `app/nodes.py`: Contains the logic for each node (Router, Safety, RAG, SQL, Search).
-   `app/tools.py`: Tool definitions.
-   `app/rag.py`: RAG implementation using FastEmbed and Supabase.
