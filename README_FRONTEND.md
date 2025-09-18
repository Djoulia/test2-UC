# Frontend Setup Instructions

## Quick Start (Easiest Method)

### Option 1: Start Everything Together
```bash
cd "/Users/milo/Desktop/Python projects/cursor tests/cursor_test_5"
source venv/bin/activate
python start_full_system.py
```

### Option 2: Start Servers Separately

**Terminal 1 - Start Backend:**
```bash
cd "/Users/milo/Desktop/Python projects/cursor tests/cursor_test_5"
source venv/bin/activate
python -m app.main
```

**Terminal 2 - Start Frontend:**
```bash
cd "/Users/milo/Desktop/Python projects/cursor tests/cursor_test_5"
python start_frontend.py
```

## What You'll See

- **Backend API**: http://localhost:8000 (FastAPI)
- **Frontend**: http://localhost:3000 (Simple HTML interface)
- **API Docs**: http://localhost:8000/docs (Swagger UI)

## How to Use

1. **Create Workflow**: 
   - Enter a workflow description like: "Search for documents about the user's question, then analyze those documents"
   - Click "Create Workflow"

2. **Test Workflow**:
   - Enter a test query like: "What is LightOn's main technology?"
   - Click "Test Workflow"
   - Wait for results (may take up to 5 minutes for document analysis)

## Example Workflow Descriptions

- `"Search for documents about the user's question, then analyze those documents to provide a detailed summary"`
- `"For each sentence in user input, search for documents, then format as Question/Answer pairs"`
- `"Search for documents, then use chat completion to summarize findings"`

## Troubleshooting

If you get `ERR_CONNECTION_REFUSED`, the backend isn't running. Make sure both servers are started.

## Tools Available in Workflows

- `paradigm_search(query)` - Document search
- `chat_completion(prompt)` - AI responses  
- `analyze_documents(query, document_ids)` - Document analysis