# Workflow Automation API

An MVP backend system that allows users to describe multi-step processes in natural language, automatically generates executable code using the Anthropic API, and executes workflows that can utilize LightOn Paradigm API endpoints.

## Features

- **Natural Language to Code**: Describe workflows in plain English and get executable Python code
- **LightOn Paradigm Integration**: Built-in support for document search using LightOn Paradigm API
- **Safe Code Execution**: Sandboxed execution environment with timeout protection
- **RESTful API**: Clean FastAPI endpoints for workflow management and execution
- **Async Support**: Full asynchronous operation for better performance

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Keys**
   
   Copy the `.env` file and add your actual API keys:
   ```bash
   # Replace with your actual keys
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   LIGHTON_API_KEY=your_lighton_api_key_here
   ```

3. **Run the Server**
   ```bash
   # From the project root directory
   python -m app.main
   
   # Or using uvicorn directly
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## API Usage

### 1. Create a Workflow

```bash
curl -X POST "http://localhost:8000/workflows" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "For each sentence in user input, search using paradigm_search, then format as Question: [sentence] Answer: [result]",
    "name": "Sentence Processing Workflow"
  }'
```

### 2. Execute a Workflow

```bash
curl -X POST "http://localhost:8000/workflows/{workflow_id}/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "What is machine learning? How does AI work?"
  }'
```

### 3. Get Workflow Details

```bash
curl -X GET "http://localhost:8000/workflows/{workflow_id}"
```

## Example Workflow

The system is designed to handle workflows like the example provided:

**Description**: "User inputs a long prompt with multiple sentences. For each sentence, perform a search using the Paradigm Docsearch tool. Return results formatted as 'Question: [sentence]' followed by 'Answer: [result]'."

**Sample Input**: "What is machine learning? How does artificial intelligence work? What are the benefits of cloud computing?"

**Expected Output**:
```
Question: What is machine learning?
Answer: [Search result about machine learning]

Question: How does artificial intelligence work?
Answer: [Search result about AI]

Question: What are the benefits of cloud computing?
Answer: [Search result about cloud computing benefits]
```

## Available Tools in Workflows

Generated workflows have access to these tools:

- `paradigm_search(query: str) -> str`: Search documents using LightOn Paradigm
- `chat_completion(prompt: str) -> str`: Get AI responses using Anthropic API

## Testing

Run the example test to verify everything works:

```bash
python test_example.py
```

## Project Structure

```
├── .env                    # API keys and configuration
├── requirements.txt        # Python dependencies
├── app/
│   ├── main.py            # FastAPI application
│   ├── config.py          # Configuration management
│   ├── models.py          # API models and schemas
│   ├── utils.py           # Helper functions
│   ├── workflow/
│   │   ├── generator.py   # Code generation using Anthropic
│   │   ├── executor.py    # Safe workflow execution
│   │   └── models.py      # Workflow data models
│   └── integrations/
│       ├── anthropic_client.py  # Anthropic API client
│       └── paradigm_client.py   # LightOn Paradigm client
└── test_example.py        # Test script
```

## API Documentation

Once the server is running, visit `http://localhost:8000/docs` for interactive API documentation.

## Security Features

- **Sandboxed Execution**: Code runs in a restricted environment
- **Timeout Protection**: Executions are limited to prevent infinite loops
- **Input Validation**: All inputs are validated before processing
- **Error Handling**: Comprehensive error handling and logging

## Next Steps

This MVP focuses on document search and chat completion tools. Future enhancements could include:

- Additional LightOn Paradigm tools
- Workflow persistence (database storage)
- User authentication and authorization
- Rate limiting and quotas
- More sophisticated code generation
- Visual workflow builder
- Webhook support for long-running workflows