"""
Direct API Clients for External Services

This module provides direct HTTP-based API clients for external services:
- Anthropic Claude API for AI code generation and chat completion
- LightOn Paradigm API for document search, analysis, and file management

Key Features:
    - Direct HTTP requests using aiohttp for async operations
    - Comprehensive error handling and logging
    - API key injection for secure communication
    - Support for all Paradigm API endpoints (search, analysis, file operations)
    - Polling mechanisms for long-running operations
    - Backward compatibility layer for existing code

Architecture:
    - Pure async/await implementation for high performance
    - Detailed logging for debugging and monitoring
    - Timeout handling and retry logic where appropriate
    - Clean separation between Anthropic and Paradigm functionality

Usage:
    The module exposes both direct functions and mock client classes
    for backward compatibility with existing code patterns.
"""
import aiohttp
import asyncio
import logging
from typing import Optional, List, Dict, Any
from .config import settings

# Set up logging for detailed API call tracking
logger = logging.getLogger(__name__)

# ============================================================================
# ANTHROPIC CLAUDE API CLIENT (Direct HTTP)
# ============================================================================

async def anthropic_generate_code(workflow_description: str, context: Optional[Dict[str, Any]] = None, system_prompt: Optional[str] = None) -> str:
    """
    Generate Python code from a workflow description using Claude via direct HTTP.
    
    Creates complete, self-contained workflow code that includes all necessary
    imports, API clients, and the main execution function. The generated code
    integrates with both Anthropic and Paradigm APIs.
    
    Args:
        workflow_description: Natural language description of the desired workflow
        context: Optional context dictionary with additional parameters
        system_prompt: Optional custom system prompt (uses default if None)
        
    Returns:
        str: Complete Python code ready for execution
        
    Raises:
        Exception: If API call fails or code generation fails
        
    Note:
        Generated code includes placeholder API keys that are replaced during execution
    """
    if not system_prompt:
        system_prompt = """You are a Python code generator for workflow automation. 
        
        Your task is to generate executable Python code that implements the described workflow.
        
        Available tools:
        
        
        Requirements:
        1. Generate clean, executable Python code
        2. Use the available tools for LLM operations
        3. Handle errors gracefully
        4. Return results in the specified format
        5. Split complex tasks into clear steps
        
        The code should define a function called 'execute_workflow(user_input: str) -> str' that takes user input and returns the final result.
        
        Example workflow: "For each sentence in user input, search using paradigm_search, then format as 'Question: [sentence] Answer: [result]'"
        
        Example code structure:
        ```python
        def execute_workflow(user_input: str) -> str:
            # Implementation here
            return final_result
        ```"""
    
    user_prompt = f"""Generate Python code for this workflow:

{workflow_description}

Additional context: {context or 'None'}

Return only the Python code, no explanations or markdown formatting."""

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 18000,
        "temperature": 0,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}]
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": settings.anthropic_api_key,
        "anthropic-version": "2023-06-01"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["content"][0]["text"]
                else:
                    error_text = await response.text()
                    raise Exception(f"Anthropic API error {response.status}: {error_text}")
    except Exception as e:
        raise Exception(f"Failed to generate code: {str(e)}")

async def anthropic_chat_completion(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Get a chat completion response from Claude via direct HTTP.
    
    General-purpose chat interface for AI responses. Used for tasks that
    don't require code generation, such as text analysis or Q&A.
    
    Args:
        prompt: User prompt or question
        system_prompt: Optional system instructions (default: helpful assistant)
        
    Returns:
        str: AI-generated response text
        
    Raises:
        Exception: If API call fails or response processing fails
    """
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "temperature": 0,
        "system": system_prompt or "You are a helpful assistant.",
        "messages": [{"role": "user", "content": prompt}]
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": settings.anthropic_api_key,
        "anthropic-version": "2023-06-01"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["content"][0]["text"]
                else:
                    error_text = await response.text()
                    raise Exception(f"Anthropic API error {response.status}: {error_text}")
    except Exception as e:
        raise Exception(f"Chat completion failed: {str(e)}")

# ============================================================================
# LIGHTON PARADIGM API CLIENT (Direct HTTP)
# ============================================================================

def _get_paradigm_headers() -> Dict[str, str]:
    """
    Get standard headers for Paradigm API requests.
    
    Returns:
        dict: Headers including authorization and content type
    """
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.lighton_api_key}"
    }

async def paradigm_document_search(
    query: str,
    workspace_ids: Optional[List[int]] = None,
    file_ids: Optional[List[int]] = None,
    chat_session_id: Optional[int] = None,
    model: Optional[str] = None,
    company_scope: bool = True,
    private_scope: bool = True,
    tool: str = "DocumentSearch",
    private: bool = False
) -> Dict[str, Any]:
    """
    Perform document search using LightOn Paradigm API via direct HTTP.
    
    Searches through uploaded documents using semantic search capabilities.
    Returns relevant documents with chunks and metadata. Can be scoped to
    specific workspaces, files, or chat sessions.
    
    Args:
        query: Search query in natural language
        workspace_ids: Optional list of workspace IDs to search within
        file_ids: Optional list of specific file IDs to search
        chat_session_id: Optional chat session for context
        model: Optional specific model to use
        company_scope: Whether to search company-wide documents
        private_scope: Whether to search private documents
        tool: Tool type (default: "DocumentSearch")
        private: Whether request is private
        
    Returns:
        dict: Search results with documents, answers, and metadata
        
    Raises:
        Exception: If search API call fails or returns error
        
    Note:
        Comprehensive logging includes query, results count, and document IDs
    """
    endpoint = f"{settings.lighton_base_url}{settings.lighton_docsearch_endpoint}"
    
    payload = {
        "query": query,
        "company_scope": company_scope,
        "private_scope": private_scope,
        "tool": tool,
        "private": private
    }
    
    # Add optional parameters if provided
    if workspace_ids:
        payload["workspace_ids"] = workspace_ids
    if file_ids:
        payload["file_ids"] = file_ids
    if chat_session_id:
        payload["chat_session_id"] = chat_session_id
    if model:
        payload["model"] = model
    
    try:
        logger.info(f"🔍 PARADIGM API CALL: Document Search")
        logger.info(f"📡 ENDPOINT: {endpoint}")
        logger.info(f"🔍 QUERY: {query}")
        logger.info(f"📋 PAYLOAD: {payload}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                json=payload,
                headers=_get_paradigm_headers()
            ) as response:
                response_text = await response.text()
                logger.info(f"📥 RAW RESPONSE: Status {response.status}, Body: {response_text[:500]}...")
                
                if response.status == 200:
                    result = await response.json()
                    # Log response details
                    doc_count = len(result.get("documents", []))
                    logger.info(f"✅ SEARCH SUCCESS: Found {doc_count} documents")
                    logger.info(f"💬 ANSWER: {result.get('answer', 'No answer')[:300]}...")
                    
                    # Log document IDs and chunks
                    if result.get("documents"):
                        doc_ids = [str(doc.get("id", "unknown")) for doc in result.get("documents", [])]
                        logger.info(f"📋 DOCUMENT IDS: {', '.join(doc_ids)}")
                        
                        # Log document chunks in detail
                        for i, doc in enumerate(result.get("documents", [])[:3]):  # First 3 docs
                            logger.info(f"📄 DOCUMENT {i+1}: ID={doc.get('id')}, Title={doc.get('title', 'Untitled')[:50]}...")
                            if 'chunks' in doc:
                                logger.info(f"📄 DOC {i+1} CHUNKS: {len(doc['chunks'])} chunks")
                    
                    return result
                else:
                    logger.error(f"❌ PARADIGM SEARCH API ERROR: {response.status} - {response_text}")
                    raise Exception(f"Paradigm document search API error {response.status}: {response_text}")
    
    except aiohttp.ClientError as e:
        logger.error(f"❌ NETWORK ERROR: {str(e)}")
        raise Exception(f"Network error calling Paradigm document search API: {str(e)}")
    except Exception as e:
        logger.error(f"❌ SEARCH FAILED: {str(e)}")
        raise Exception(f"Paradigm document search failed: {str(e)}")

async def paradigm_document_analysis(
    query: str,
    document_ids: List[str],
    model: Optional[str] = None,
    private: bool = False
) -> Dict[str, Any]:
    """
    Initiate document analysis using LightOn Paradigm API via direct HTTP.
    
    Starts an analysis job for specific documents. Returns a chat_response_id
    that can be used to poll for results. Analysis is performed asynchronously.
    
    Args:
        query: Analysis question or instruction
        document_ids: List of document IDs to analyze
        model: Optional specific model to use for analysis
        private: Whether analysis should be private
        
    Returns:
        dict: Analysis initiation response with chat_response_id
        
    Raises:
        Exception: If analysis API call fails or returns error
        
    Note:
        This starts the analysis - use paradigm_get_analysis_result to retrieve results
    """
    endpoint = f"{settings.lighton_base_url}/api/v2/chat/document-analysis"
    
    payload = {
        "query": query,
        "document_ids": document_ids,
        "private": private
    }
    
    # Add optional parameters if provided
    if model:
        payload["model"] = model
    
    try:
        logger.info(f"📊 PARADIGM API CALL: Document Analysis")
        logger.info(f"📡 ENDPOINT: {endpoint}")
        logger.info(f"🔍 QUERY: {query}")
        logger.info(f"📋 DOCUMENT IDS: {document_ids}")
        logger.info(f"📋 PAYLOAD: {payload}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                json=payload,
                headers=_get_paradigm_headers()
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    chat_response_id = result.get("chat_response_id")
                    logger.info(f"✅ ANALYSIS STARTED: chat_response_id = {chat_response_id}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"❌ PARADIGM ANALYSIS API ERROR: {response.status} - {error_text}")
                    raise Exception(f"Paradigm document analysis API error {response.status}: {error_text}")
    
    except aiohttp.ClientError as e:
        logger.error(f"❌ NETWORK ERROR: {str(e)}")
        raise Exception(f"Network error calling Paradigm document analysis API: {str(e)}")
    except Exception as e:
        logger.error(f"❌ ANALYSIS FAILED: {str(e)}")
        raise Exception(f"Paradigm document analysis failed: {str(e)}")

async def paradigm_get_analysis_result(chat_response_id: int) -> Dict[str, Any]:
    """
    Retrieve the results of a document analysis request via direct HTTP.
    
    Polls the analysis endpoint to get results from a previously initiated
    document analysis. May return "not found" if analysis is still processing.
    
    Args:
        chat_response_id: ID returned from paradigm_document_analysis
        
    Returns:
        dict: Analysis results with status and detailed analysis
        
    Raises:
        Exception: If retrieval fails or analysis result not found
        
    Note:
        Use paradigm_analyze_documents_with_polling for automatic polling
    """
    endpoint = f"{settings.lighton_base_url}/api/v2/chat/document-analysis/{chat_response_id}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                endpoint,
                headers=_get_paradigm_headers()
            ) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    raise Exception(f"Analysis result not found for ID {chat_response_id}")
                else:
                    error_text = await response.text()
                    raise Exception(f"Paradigm get analysis result API error {response.status}: {error_text}")
    
    except aiohttp.ClientError as e:
        raise Exception(f"Network error calling Paradigm get analysis result API: {str(e)}")
    except Exception as e:
        raise Exception(f"Getting Paradigm analysis result failed: {str(e)}")

async def paradigm_analyze_documents_with_polling(
    query: str,
    document_ids: List[str],
    model: Optional[str] = None,
    private: bool = False,
    max_wait_time: int = 300,
    poll_interval: int = 5
) -> str:
    """
    Perform document analysis and poll for results until completion via direct HTTP
    """
    try:
        logger.info(f"📊 STARTING DOCUMENT ANALYSIS WITH POLLING")
        logger.info(f"🔍 QUERY: {query}")
        logger.info(f"📋 DOCUMENT IDS: {document_ids}")
        logger.info(f"🤖 MODEL: {model or 'default'}")
        logger.info(f"🔐 PRIVATE: {private}")
        
        # Start the analysis
        analysis_response = await paradigm_document_analysis(query, document_ids, model, private)
        chat_response_id = analysis_response.get("chat_response_id")
        
        logger.info(f"📊 ANALYSIS STARTED: chat_response_id = {chat_response_id}")
        logger.info(f"📊 FULL RESPONSE: {analysis_response}")
        
        if not chat_response_id:
            raise Exception("No chat_response_id returned from analysis request")
        
        # Poll for results
        elapsed_time = 0
        logger.info(f"🔄 STARTING POLLING: chat_response_id = {chat_response_id}")
        
        while elapsed_time < max_wait_time:
            try:
                result = await paradigm_get_analysis_result(chat_response_id)
                
                status = result.get("status", "")
                progress = result.get("progress", "")
                
                logger.info(f"📊 POLLING STATUS: {status} | Progress: {progress} | Elapsed: {elapsed_time}s")
                
                if status.lower() in ["completed", "complete", "finished", "success"]:
                    # Analysis is complete, return the result
                    logger.info(f"✅ ANALYSIS COMPLETED: chat_response_id = {chat_response_id}")
                    logger.info(f"📊 FULL RESULT DATA: {result}")
                    analysis_result = result.get("result", result.get("detailed_analysis", "Analysis completed but no result found"))
                    logger.info(f"📄 RESULT LENGTH: {len(analysis_result)} characters")
                    logger.info(f"📄 RESULT CONTENT: {analysis_result[:500]}...")
                    return analysis_result
                elif status.lower() in ["failed", "error"]:
                    logger.error(f"❌ ANALYSIS FAILED: {status}")
                    raise Exception(f"Analysis failed with status: {status}")
                
                # Still processing, wait and try again
                await asyncio.sleep(poll_interval)
                elapsed_time += poll_interval
                
            except Exception as e:
                if "not found" in str(e).lower():
                    # Continue polling if result not ready yet
                    logger.info(f"⏳ RESULT NOT READY: Continuing to poll... ({elapsed_time}s)")
                    await asyncio.sleep(poll_interval)
                    elapsed_time += poll_interval
                    continue
                else:
                    logger.error(f"❌ POLLING ERROR: {str(e)}")
                    raise e
        
        raise Exception(f"Analysis timed out after {max_wait_time} seconds")
        
    except Exception as e:
        return f"Document analysis failed: {str(e)}"

async def paradigm_upload_file(
    file_content: bytes,
    filename: str,
    collection_type: str = "private",
    workspace_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Upload a file to Paradigm for analysis via direct HTTP
    """
    endpoint = f"{settings.lighton_base_url}/api/v2/files"
    
    # Prepare multipart form data
    data = aiohttp.FormData()
    data.add_field('file', file_content, filename=filename, content_type='application/octet-stream')
    data.add_field('collection_type', collection_type)
    if workspace_id:
        data.add_field('workspace_id', str(workspace_id))
    
    headers = {
        "Authorization": f"Bearer {settings.lighton_api_key}"
    }
    
    try:
        logger.info(f"📁 PARADIGM API CALL: File Upload")
        logger.info(f"📡 ENDPOINT: {endpoint}")
        logger.info(f"📄 FILENAME: {filename}")
        logger.info(f"📦 FILE SIZE: {len(file_content)} bytes")
        logger.info(f"🗂️ COLLECTION TYPE: {collection_type}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                data=data,
                headers=headers
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    logger.info(f"✅ UPLOAD SUCCESS: File ID = {result.get('id')}, Status = {result.get('status')}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"❌ PARADIGM UPLOAD API ERROR: {response.status} - {error_text}")
                    raise Exception(f"Paradigm file upload API error {response.status}: {error_text}")
    
    except aiohttp.ClientError as e:
        logger.error(f"❌ NETWORK ERROR: {str(e)}")
        raise Exception(f"Network error calling Paradigm file upload API: {str(e)}")
    except Exception as e:
        logger.error(f"❌ UPLOAD FAILED: {str(e)}")
        raise Exception(f"Paradigm file upload failed: {str(e)}")

async def paradigm_get_file_info(file_id: int, include_content: bool = False) -> Dict[str, Any]:
    """
    Get information about an uploaded file via direct HTTP
    """
    endpoint = f"{settings.lighton_base_url}/api/v2/files/{file_id}"
    if include_content:
        endpoint += "?include_content=true"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                endpoint,
                headers=_get_paradigm_headers()
            ) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    raise Exception(f"File not found: {file_id}")
                else:
                    error_text = await response.text()
                    raise Exception(f"Paradigm get file info API error {response.status}: {error_text}")
    
    except aiohttp.ClientError as e:
        raise Exception(f"Network error calling Paradigm get file info API: {str(e)}")
    except Exception as e:
        raise Exception(f"Getting Paradigm file info failed: {str(e)}")

async def paradigm_ask_question_about_file(file_id: int, question: str) -> Dict[str, Any]:
    """
    Ask a question about a specific uploaded file via direct HTTP
    """
    endpoint = f"{settings.lighton_base_url}/api/v2/files/{file_id}/ask"
    
    payload = {
        "question": question
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                json=payload,
                headers=_get_paradigm_headers()
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Paradigm ask question API error {response.status}: {error_text}")
    
    except aiohttp.ClientError as e:
        raise Exception(f"Network error calling Paradigm ask question API: {str(e)}")
    except Exception as e:
        raise Exception(f"Paradigm ask question failed: {str(e)}")

async def paradigm_delete_file(file_id: int) -> bool:
    """
    Delete a file from Paradigm via direct HTTP
    """
    endpoint = f"{settings.lighton_base_url}/api/v2/files/{file_id}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                endpoint,
                headers=_get_paradigm_headers()
            ) as response:
                if response.status == 200:
                    return True
                elif response.status == 404:
                    logger.warning(f"File not found for deletion: {file_id}")
                    return False
                else:
                    error_text = await response.text()
                    raise Exception(f"Paradigm delete file API error {response.status}: {error_text}")
    
    except aiohttp.ClientError as e:
        raise Exception(f"Network error calling Paradigm delete file API: {str(e)}")
    except Exception as e:
        raise Exception(f"Paradigm delete file failed: {str(e)}")

# ============================================================================
# COMPATIBILITY LAYER (for existing code)
# ============================================================================

# Create mock client objects for backward compatibility
class MockAnthropicClient:
    def __init__(self):
        pass
    
    async def generate_code(self, workflow_description: str, context: Optional[Dict[str, Any]] = None, system_prompt: Optional[str] = None) -> str:
        return await anthropic_generate_code(workflow_description, context, system_prompt)
    
    async def chat_completion(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return await anthropic_chat_completion(prompt, system_prompt)

class MockParadigmClient:
    def __init__(self):
        pass
    
    async def document_search(self, query: str, **kwargs) -> Dict[str, Any]:
        return await paradigm_document_search(query, **kwargs)
    
    async def document_analysis(self, query: str, document_ids: List[str], **kwargs) -> Dict[str, Any]:
        return await paradigm_document_analysis(query, document_ids, **kwargs)
    
    async def get_analysis_result(self, chat_response_id: int) -> Dict[str, Any]:
        return await paradigm_get_analysis_result(chat_response_id)
    
    async def analyze_documents_with_polling(self, query: str, document_ids: List[str], **kwargs) -> str:
        return await paradigm_analyze_documents_with_polling(query, document_ids, **kwargs)
    
    async def upload_file(self, file_content: bytes, filename: str, **kwargs) -> Dict[str, Any]:
        return await paradigm_upload_file(file_content, filename, **kwargs)
    
    async def get_file_info(self, file_id: int, **kwargs) -> Dict[str, Any]:
        return await paradigm_get_file_info(file_id, **kwargs)
    
    async def ask_question_about_file(self, file_id: int, question: str) -> Dict[str, Any]:
        return await paradigm_ask_question_about_file(file_id, question)
    
    async def delete_file(self, file_id: int) -> bool:
        return await paradigm_delete_file(file_id)

# Create global instances for backward compatibility
anthropic_client = MockAnthropicClient()
paradigm_client = MockParadigmClient() 