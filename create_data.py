

import camelot
import pandas as pd
import os
import json
import argparse
from typing import List, Dict, Any

# LangChain components for text extraction and chunking
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_output_directory(output_dir: str) -> None:
    """
    Creates the output directory if it doesn't already exist.
    
    Args:
        output_dir (str): The path to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")




import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer 
import os
import torch

DB_PATH = "faiss_index.bin"


def create_vector_db(concepts: list[str]):
    """
    Creates a FAISS vector database from a list of text concepts.
    The embeddings are generated using 'sentence-transformers/all-MiniLM-L6-v2'.
    The FAISS index and original texts are saved to files.
    """
    print("Loading SentenceTransformer model 'sentence-transformers/all-MiniLM-L6-v2'...")

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model.eval() # Set model to evaluation mode

    print("Generating embeddings for Python concepts...")

    embeddings = model.encode(concepts, convert_to_numpy=True)

    print(f"Embeddings shape: {embeddings.shape}")

    embeddings = embeddings.astype('float32')

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    print("Adding embeddings to FAISS index...")
    index.add(embeddings)

    print(f"Number of vectors in the index: {index.ntotal}")

    faiss.write_index(index, DB_PATH)
    print(f"FAISS index saved to {DB_PATH}")

    # print(f"Original concept texts saved to {TEXTS_PATH}")



def extract_tables_from_pdf(pdf_path: str, output_dir: str) -> None:
    """
    Extracts all tables from a PDF using Camelot and saves them as individual CSV files.
    (This function remains unchanged).

    Args:
        pdf_path (str): The file path of the input PDF.
        output_dir (str): The directory where the output CSV files will be saved.
    """
    print("Starting table extraction...")
    try:
        # Using 'stream' method is often better for PDFs with less-defined table structures.
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
        
        if not tables:
            print("No tables found in the PDF.")
            return

        print(f"Found {tables.n} tables in the document.")

        Table_info = []
        Table_summary = []
        Table_number = 0
        
        for i, table in enumerate(tables):
            # The table parsing report can give insights into accuracy
            print(f"Processing Table {i+1} from Page {table.page}...")
            print(f"Parsing Report: {table.parsing_report}")
            Table_number += 1
            # Use the first row as headers
            df = table.df
            # A simple heuristic to find the header row
            if not df.empty and len(df.columns) > 1:
                df.columns = df.iloc[0]
                df = df[1:].reset_index(drop=True)
            
            # print(f"Tabel Info : {df.head(0).keys()[0]}")
            Table_info.append(df.head(0).keys()[0])
            Table_summary.append(f"Table_{Table_number} : {df.head(0).keys()[0]}")

            # Construct a meaningful filename
            table_filename = os.path.join(output_dir, f"Table_{Table_number}.csv")
            df.to_csv(table_filename, index=False)
            
            print(f"Saved Table {i+1} to {table_filename}")

        summary_file_path = os.path.join(f"{os.getcwd()}/pdf_data", "table_info.txt")
        with open(summary_file_path, "w", encoding="utf-8") as f:
            for line in Table_summary:
                f.write(line + "\n")
        print(f"Table summary saved to {summary_file_path}")


        return Table_info
            
    except Exception as e:
        print(f"An error occurred during table extraction: {e}")
        print("This may be due to the PDF's structure or missing dependencies (like ghostscript).")


def process_pdf(pdf_path: str, output_dir: str) -> None:
    """
    Main function to process a PDF file. It orchestrates the creation of the
    output directory and calls the text and table extraction functions.

    Args:
        pdf_path (str): The path to the PDF file to be processed.
        output_dir (str): The path to the directory where all output files will be stored.
    """
    print(f"--- Starting processing for: {pdf_path} ---")
    
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' was not found.")
        return
        
    create_output_directory(output_dir)

    Table_info = extract_tables_from_pdf(pdf_path, output_dir)

    print(Table_info)

    create_vector_db(Table_info)

    
    print(f"--- Finished processing. All files are saved in: {output_dir} ---")


if __name__ == "__main__":
    # Set up the command-line interface
    parser = argparse.ArgumentParser(
        description="Data Analysis Copilot - Data Creation Script.\n"
                    "Extracts text chunks and tables from a PDF document using LangChain."
    )
    
    parser.add_argument(
        "pdf_file", 
        type=str, 
        help="The full path to the PDF file you want to process."
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="processed_data",
        help="The directory where the output files (JSON, CSVs) will be stored. Defaults to 'processed_data'."
    )
    
    args = parser.parse_args()
    
    process_pdf(args.pdf_file, args.output_dir)

