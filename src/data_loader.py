from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader

def load_all_documents(data_dir: str) -> List[Any]:
  """
  Load all supported files from data directory and convert to LangChain document structure
  Supported: PDF, TXT, CSV, EXCEL, WORD
  """

  data_path = Path(data_dir).resolve()
  print(f"Data path: {data_path}")
  documents = []

  # PDF
  pdf_files = list(data_path.glob('**/*.pdf'))
  print(f"Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
  for pdf_file in pdf_files:
    print(f"Loading file: {pdf_file}")
    try:
      loader = PyPDFLoader(str(pdf_file))
      loaded = loader.load()
      print(f"Loaded {len(loaded)} PDF docs from {pdf_file}")
      documents.extend(loaded)
    except Exception as e:
      print(f"Failed to load PDF {pdf_file}: {e}")
  
  # TXT
  txt_files = list(data_path.glob('**/*.txt'))
  print(f"Found {len(txt_files)} TXT files: {[str(f) for f in txt_files]}")
  for txt_file in txt_files:
    print(f"Loading file: {txt_file}")
    try:
      loader = TextLoader(str(txt_file))
      loaded = loader.load()
      print(f"Loaded {len(loaded)} TXT docs from {txt_file}")
      documents.extend(loaded)
    except Exception as e:
      print(f"Failed to load TXT {txt_file}: {e}")
  
  # CSV
  csv_files = list(data_path.glob('**/*.csv'))
  print(f"Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")
  for csv_file in csv_files:
    print(f"Loading file: {csv_file}")
    try:
      loader = CSVLoader(str(csv_file))
      loaded = loader.load()
      print(f"Loaded {len(loaded)} CSV docs from {csv_file}")
      documents.extend(loaded)
    except Exception as e:
      print(f"Failed to load CSV {csv_file}: {e}")
  
  # EXCEL
  excel_files = list(data_path.glob('**/*.xlsx'))
  print(f"Found {len(excel_files)} Excel files: {[str(f) for f in excel_files]}")
  for excel_file in excel_files:
    print(f"Loading file: {excel_file}")
    try:
      loader = UnstructuredExcelLoader(str(excel_file))
      loaded = loader.load()
      print(f"Loaded {len(loaded)} Excel docs from {excel_file}")
      documents.extend(loaded)
    except Exception as e:
      print(f"Failed to load Excel {excel_file}: {e}")

  # WORD
  word_files = list(data_path.glob('**/*.docx'))
  print(f"Found {len(word_files)} Word files: {[str(f) for f in word_files]}")
  for word_file in word_files:
    print(f"Loading file: {word_file}")
    try:
      loader = Docx2txtLoader(str(word_file))
      loaded = loader.load()
      print(f"Loaded {len(loaded)} Word docs from {word_file}")
      documents.extend(loaded)
    except Exception as e:
      print(f"Failed to load Word {word_file}: {e}")

  return documents
