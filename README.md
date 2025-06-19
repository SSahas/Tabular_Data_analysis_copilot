# Tabular_Data_analysis_copilot

A tabular analysis agent which can extract and analyze tables from PDF files.This system helps  perform intelligent and automated analysis on structured data without writing manual code. It was mainly built using langGraph. 

- The crete_data.py is used to extract the tables from the input pdf
- The main.py contains all the agentic functionality.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SSahas/Tabular_Data_analysis_copilot.git
cd Tabular_Data_analysis_copilot
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

- Please don't change the --output_dir argument , keep it "pdf_data". The agent uses this name to function to load the data tables.

```bash
python create_data.py  "sample_pdf_document.pdf" --output_dir "pdf_data"
```

5. Run the main.py file
```bash
python main.py
```
