# Tabular_Data_analysis_copilot

A tabular analysis agent which can extract and analyze tables from PDF files.This system helps  perform intelligent and automated analysis on structured data without writing manual code. It was mainly built using langGraph. 

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SSahas/tabular-data-analysis-copilot.git
cd data-analysis-copilot
```

2. Create a .env file and add anthropic api key
```bash
ANTHROPIC_API_KEY = "your_api_key"
```

3. install dependencies
```bash
pip install -r requirements.txt
```

4. extract tables
```bash
python create_data.py  "sample_pdf_document.pdf" --output_dir "path/to/where_you_want_store_extarcted_tabelfiles"
```

5. Run the main.py file
```bash
python main.py
```
