import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import functools
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Configuration
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "faiss_index.bin"
PDF_DATA_DIR = BASE_DIR / "pdf_data"
TABLES_SUMMARY_PATH = PDF_DATA_DIR / "table_info.txt"
ANALYSIS_OUTPUT_DIR = BASE_DIR / "analysis_outputs"

# Ensure output directory exists
ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True)


def load_table_summary():
    """Load table summary with proper error handling."""
    try:
        if TABLES_SUMMARY_PATH.exists():
            return TABLES_SUMMARY_PATH.read_text(encoding="utf-8")
        else:
            print(
                f"Warning: Table summary file not found at {TABLES_SUMMARY_PATH}")
            return ""
    except Exception as e:
        print(f"Error reading table summary file: {e}")
        return ""


# Load table summary
table_summary_content = load_table_summary()


@functools.lru_cache(maxsize=1)
def load_embedding_model():
    """Load and cache the embedding model."""
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        raise


@functools.lru_cache(maxsize=1)
def load_vector_db():
    """Load and cache the FAISS index."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {DB_PATH}. Please create it using create_data.py")

    try:
        index = faiss.read_index(str(DB_PATH))
        return index
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        raise


# Initialize LLM with error handling
try:
    llm = init_chat_model("claude-sonnet-4-20250514")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    raise


def execute_analysis_code(code: str, table_number: int) -> dict:
    """Safely execute generated Python code for data analysis."""

    try:
        # Clean code
        if code.startswith("```python"):
            code = code.strip("```python").rstrip("```").strip()



        local_namespace = {'pd': pd}
        exec(code, {}, local_namespace)
        # Save successful code
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        code_file = ANALYSIS_OUTPUT_DIR / f"executed_code_{timestamp}.py"
        code_file.write_text(code)

        return {'status': 'SUCCESS'}

    except Exception as e:
        print(f"Execution error: {e}")

        # Save failed code for debugging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        code_file = ANALYSIS_OUTPUT_DIR / \
            f"executed_code_FAILED_{timestamp}.py"
        code_file.write_text(f"# Error: {e}\n\n{code}")

        return {'status': 'ERROR', 'error': str(e)}


class MessageClassifier(BaseModel):
    message_type: Literal["general", "data_analysis"] = Field(
        ...,
        description="""You are a query classifier. Classify the user query into one of these categories:
              1. "general" - General questions or asking about the data you have access to not specifically about the data analysis, visualizations.
              2. "data_analysis" - Questions asking to perform data analysis, visualizations , produce plots etc..
            """
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None


def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """You are a query classifier. Classify the user query into one of these categories:
              1. "general" - General questions or asking about the data you have access to not specifically about the data analysis, visualizations.
              2. "data_analysis" - Questions asking to perform data analysis, visualizations , produce plots etc..

            """
        },
        {"role": "user", "content": last_message.content}
    ])
    return {"message_type": result.message_type}


def router(state: State):
    message_type = state.get("message_type", "data_analysis")
    if message_type == "general":
        return {"next": "general"}

    return {"next": "data_analysis"}


def general_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": f"""You are a data analysis copilot. You have acces to the below files.
                        {table_summary_content}
                        Respond to the user query with relevany response. If the user about the files you have acces, give the appropriate reply as the user input.
                        DO NOT generate any code, reply ONLY with normal responses.
                        Be respecful and do not produce anything harmful."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


def data_analysis_agent(state: State):
    """Enhanced data analysis agent with better error handling."""
    last_message = state["messages"][-1]

    try:
        embedding_model = load_embedding_model()
        faiss_index = load_vector_db()

        query_embedding = embedding_model.encode(
            [last_message.content], convert_to_numpy=True).astype('float32')
        distances, indices = faiss_index.search(query_embedding, 1)

        table_number = indices[0][0] + 1
        table_path = PDF_DATA_DIR / f"Table_{table_number}.csv"

        if not table_path.exists():
            return {"messages": [{"role": "assistant", "content": f"Error: Table file {table_path} not found."}]}

        df = pd.read_csv(table_path)

        messages = [
            {"role": "system",
             "content": f"""You are a data analysis assistant. Generate complete, executable Python code for the user query.
                           You are provided with sample table data. Use it to understand the table structure.
                           Load the CSV file from: {table_path}
                           
                           If creating plots/graphs, save them to: {ANALYSIS_OUTPUT_DIR}
                           Include code to create the output directory if it doesn't exist.
                           Do not generate code to show plots - only save them.
                           
                           Handle data type conversions carefully.
                           Generate ONLY Python code - no explanations.
                           
                           Sample data:
                           {df.head().to_string()}
                           
                           User Query: {last_message.content}
                           """
             },
            {"role": "user", "content": last_message.content}
        ]

        reply = llm.invoke(messages)
        result = execute_analysis_code(
            code=reply.content, table_number=table_number)

        if result['status'] == 'ERROR':
            status = f"Error executing the code: {result.get('error', 'Unknown error')}"
        else:
            status = f"Code executed successfully. Check {ANALYSIS_OUTPUT_DIR} for outputs."

        return {"messages": [{"role": "assistant", "content": status}]}

    except Exception as e:
        error_msg = f"Error in data analysis: {e}"
        print(error_msg)
        return {"messages": [{"role": "assistant", "content": error_msg}]}


def build_graph():
    graph_builder = StateGraph(State)

    graph_builder.add_node("classifier", classify_message)
    graph_builder.add_node("router", router)
    graph_builder.add_node("general", general_agent)
    graph_builder.add_node("data_analysis", data_analysis_agent)

    graph_builder.add_edge(START, "classifier")
    graph_builder.add_edge("classifier", "router")

    graph_builder.add_conditional_edges(
        "router",
        lambda state: state.get("next"),
        {"general": "general", "data_analysis": "data_analysis"}
    )

    graph_builder.add_edge("general", END)
    graph_builder.add_edge("data_analysis", END)

    return graph_builder.compile()


def run_chatbot():
    """Run the chatbot with better error handling."""
    try:
        graph = build_graph()
        state = {"messages": [], "message_type": None}

        print("Chatbot initialized. Type 'exit' to quit.")

        while True:
            try:
                user_input = input("Message: ").strip()
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                state["messages"] = state.get("messages", []) + [
                    {"role": "user", "content": user_input}
                ]

                state = graph.invoke(state)

                if state.get("messages") and len(state["messages"]) > 0:
                    last_message = state["messages"][-1]
                    print(f"Assistant: {last_message.content}")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error processing message: {e}")

    except Exception as e:
        print(f"Error initializing chatbot: {e}")


if __name__ == "__main__":
    run_chatbot()
