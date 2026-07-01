# Free Local RAG-Based Company Chatbot

This is a free local version of the Streamlit RAG chatbot. It does **not** use OpenAI API, so it can run without API billing credits.

## What this version does

- Loads local `.txt` and `.xlsx` files from the `data/` folder
- Converts data into chunks
- Creates local TF-IDF vectors
- Stores vectors in a local vector index; uses FAISS when available
- Retrieves top-3 relevant chunks for each user question
- Gives an extractive answer from the retrieved company data only
- Says `I don't know` when relevant data is not found
- Uses Streamlit chat UI with message history

## Project Structure

```text
rag_streamlit_local_free_project/
├── app.py
├── rag.py
├── requirements.txt
├── README.md
└── data/
    ├── company_info.txt
    └── products.xlsx
```

## Setup on Windows

### 1. Open project folder in VS Code

Open this folder:

```text
rag_streamlit_local_free_project
```

### 2. Create virtual environment

```powershell
python -m venv venv
```

### 3. Activate virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then activate again:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install libraries

```powershell
pip install -r requirements.txt
```

### 5. Run app

```powershell
streamlit run app.py
```

Open the browser link:

```text
http://localhost:8501
```

## Example Questions

```text
What are the company office timings?
How many annual paid leave days do employees receive?
What is the refund policy?
Which product is used for cybersecurity monitoring?
What is the monthly price of CloudMove Pro?
Who is the CEO of the company?
```

The last question should return:

```text
I don't know
```

## Important Note

This version is designed to run without OpenAI billing. It uses local TF-IDF retrieval and extractive answers instead of OpenAI embeddings and OpenAI LLM. It works using local NumPy similarity search. If you also install `faiss-cpu`, the same code will use FAISS automatically.

For strict capstone submission, if your teacher specifically requires OpenAI API, use the OpenAI version after adding API billing credits.


## Optional FAISS Install

The app works without this. If your teacher wants FAISS specifically and your Windows supports it, you can try:

```powershell
pip install -r requirements-faiss-optional.txt
```

Then run the app again.
