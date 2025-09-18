import asyncio
import logging
import re
from typing import Optional, Dict, Any
from .models import Workflow
from anthropic import Anthropic
from ..config import settings

logger = logging.getLogger(__name__)


class WorkflowGenerator:
    def __init__(self):
        self.anthropic_client = Anthropic(api_key=settings.anthropic_api_key)

    async def generate_workflow(
        self,
        description: str,
        name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Workflow:
        """
        Generate a workflow from a natural language description
        Args:
            description: Natural language description of the workflow
            name: Optional name for the workflow
            context: Additional context for code generation
        Returns:
            Workflow object with generated code
        """
        workflow = Workflow(
            name=name,
            description=description,
            context=context
        )
        
        try:
            workflow.update_status("generating")
            # Generate the code using Anthropic API
            generated_code = await self._generate_code(description, context)
            
            # Validate the generated code
            validation_result = await self._validate_code(generated_code)
            if not validation_result["valid"]:
                raise Exception(f"Generated code validation failed: {validation_result['error']}")
            
            workflow.generated_code = generated_code
            workflow.update_status("ready")
            return workflow
            
        except Exception as e:
            workflow.update_status("failed", str(e))
            raise e

    async def _generate_code(self, description: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate Python code from workflow description
        """
        system_prompt = """You are a Python code generator for workflow automation systems.

CRITICAL INSTRUCTIONS:
1. Generate ONLY executable Python code - no markdown, no explanations, no comments
2. The code must define: async def execute_workflow(user_input: str) -> str
3. Include ALL necessary imports and API client code directly in the workflow
4. Make the workflow completely self-contained and portable
5. *** NEVER USE 'pass' OR PLACEHOLDER COMMENTS - IMPLEMENT ALL FUNCTIONS COMPLETELY ***
6. *** EVERY FUNCTION MUST BE FULLY IMPLEMENTED WITH WORKING CODE ***
7. *** NO STUB FUNCTIONS - ALL CODE MUST BE EXECUTABLE AND FUNCTIONAL ***

REQUIRED STRUCTURE:
```python
import asyncio
import aiohttp
import json
import logging
from typing import Optional, List, Dict, Any

# Configuration - replace with your actual values
LIGHTON_API_KEY = "your_api_key_here"
LIGHTON_BASE_URL = "https://api.lighton.ai"

logger = logging.getLogger(__name__)

class ParadigmClient:
    def __init__(self, api_key: str, base_url: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    async def document_search(self, query: str, **kwargs) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/api/v2/chat/document-search"
        payload = {"query": query, **kwargs}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, headers=self.headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"API error {response.status}: {await response.text()}")
    
    async def analyze_documents_with_polling(self, query: str, document_ids: List[int], **kwargs) -> str:
        # Start analysis
        endpoint = f"{self.base_url}/api/v2/chat/document-analysis"
        payload = {"query": query, "document_ids": document_ids, **kwargs}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, headers=self.headers) as response:
                if response.status == 200:
                    result = await response.json()
                    chat_response_id = result.get("chat_response_id")
                else:
                    raise Exception(f"Analysis API error {response.status}: {await response.text()}")
        
        # Poll for results
        max_wait = 300  # 5 minutes
        poll_interval = 5
        elapsed = 0
        
        while elapsed < max_wait:
            endpoint = f"{self.base_url}/api/v2/chat/document-analysis/{chat_response_id}"
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, headers=self.headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        status = result.get("status", "")
                        if status.lower() in ["completed", "complete", "finished", "success"]:
                            analysis_result = result.get("result") or result.get("detailed_analysis") or "Analysis completed"
                            return analysis_result
                        elif status.lower() in ["failed", "error"]:
                            raise Exception(f"Analysis failed: {status}")
                    elif response.status == 404:
                        # Analysis not ready yet, continue polling
                        pass
                    else:
                        raise Exception(f"Polling API error {response.status}: {await response.text()}")
                    
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval
        
        raise Exception("Analysis timed out")
    
    async def chat_completion(self, prompt: str, model: str = "alfred-4.2") -> str:
        endpoint = f"{self.base_url}/api/v2/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, headers=self.headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"Paradigm chat completion API error {response.status}: {await response.text()}")
    
    async def analyze_image(self, query: str, document_ids: List[str], model: str = None, private: bool = False) -> str:
        endpoint = f"{self.base_url}/api/v2/chat/image-analysis"
        payload = {
            "query": query,
            "document_ids": document_ids
        }
        if model:
            payload["model"] = model
        if private is not None:
            payload["private"] = private
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, headers=self.headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("answer", "No analysis result provided")
                else:
                    raise Exception(f"Image analysis API error {response.status}: {await response.text()}")

# Initialize clients
paradigm_client = ParadigmClient(LIGHTON_API_KEY, LIGHTON_BASE_URL)

async def execute_workflow(user_input: str) -> str:
    # Your workflow implementation here
    pass
```

IMPORTANT LIBRARY RESTRICTIONS:
- Only use built-in Python libraries (asyncio, json, logging, typing, re, etc.)
- Only use aiohttp for HTTP requests (already included in template)
- DO NOT import external libraries like nltk, requests, pandas, numpy, etc.

CRITICAL: AVOID VARIABLE SCOPING ERRORS
Always define variables before using them in conditional blocks:

# WRONG - causes UnboundLocalError
if condition:
    result = await api_call()
print(f"Result: {result}")  # ERROR if condition is False

# CORRECT - define first
result = None  # or appropriate default
if condition:
    result = await api_call()
print(f"Result: {result}")  # Always works

AVAILABLE API METHODS:
1. await paradigm_client.document_search(query: str, workspace_ids=None, file_ids=None, company_scope=True, private_scope=True, tool="DocumentSearch", private=False)
   *** CRITICAL SCOPING: Use file_ids for SPECIFIC documents OR company_scope/private_scope for BROAD search, NOT both together ***
   *** When file_ids is specified, do NOT set company_scope/private_scope (this would expand search beyond target documents) ***
   *** For specific document search: file_ids=[doc_ids] only ***
   *** For broad search: company_scope=True, private_scope=True only ***
   *** VISUAL SEARCH FALLBACK: If standard DocumentSearch doesn't find the required information, use tool="VisionDocumentSearch" as fallback ***
2. await paradigm_client.analyze_documents_with_polling(query: str, document_ids: List[str], model=None, private=False) 
   *** CRITICAL: document_ids can contain MAXIMUM 5 documents. If more than 5, use batching! ***
   *** IMPORTANT: For document type identification, analyze documents ONE BY ONE to get clear ID-to-type mapping ***
3. await paradigm_client.chat_completion(prompt: str, model: str = "Alfred 4.2")
4. await paradigm_client.analyze_image(query: str, document_ids: List[str], model=None, private=False) - Analyze images in documents with AI-powered visual analysis
   *** CRITICAL: document_ids can contain MAXIMUM 5 documents. If more than 5, use batching! ***

CONTEXT PRESERVATION IN API PROMPTS:
When creating prompts for API calls, include relevant context from the original workflow description: examples, formatting requirements, specific field names, and business rules mentioned by the user.

WORKFLOW ACCESS TO ATTACHED FILES:
- Get attached files with: attached_file_ids = globals().get('attached_file_ids', [])
- Pass these IDs to file_ids parameter in document_search
- For document analysis: use attached_file_ids directly as document IDs
- Extract document IDs from search results for analysis ONLY when searching, not when using attached files

CORRECT DOCUMENT SEARCH SCOPING:
# Get attached files (empty list if none)
attached_file_ids = globals().get('attached_file_ids', [])

# For specific document search (when targeting particular documents):
if attached_file_ids:
    search_kwargs = {"query": query, "file_ids": attached_file_ids}
# For broad search (when no specific documents are targeted):
else:
    search_kwargs = {"query": query, "company_scope": True, "private_scope": True}

search_results = await paradigm_client.document_search(**search_kwargs)

# Alternative pattern for workflows that need to specify target documents:
target_document_ids = [123, 456, 789]  # Replace with actual document IDs
search_kwargs = {"query": query, "file_ids": target_document_ids}
# DO NOT add company_scope/private_scope when using file_ids for specific document targeting

CORRECT DOCUMENT_IDS EXTRACTION FOR ANALYSIS OR SEARCH:
document_ids = [str(doc["id"]) for doc in search_results.get("documents", [])]  # Convert to strings
# OR for attached files: document_ids = [str(file_id) for file_id in attached_file_ids]

CRITICAL: DOCUMENT ANALYSIS 5-DOCUMENT LIMIT:
# Document analysis can only handle 5 documents at a time
# If you have more than 5 documents, you MUST split them into batches

# ALWAYS check document count before analysis:
if len(document_ids) > 5:
    # Process in batches of 5
    results = []
    for i in range(0, len(document_ids), 5):
        batch = document_ids[i:i+5]
        result = await paradigm_client.analyze_documents_with_polling(query, batch)
        results.append(result)
    final_analysis = "\\n\\n".join(results)
else:
    # Process all documents at once (5 or fewer)
    final_analysis = await paradigm_client.analyze_documents_with_polling(query, document_ids)

CORRECT SEARCH RESULT USAGE:
search_result = await paradigm_client.document_search(**search_kwargs)
# Use the AI-generated answer from search results
answer = search_result.get("answer", "No answer provided")
# Don't try to extract raw document content - use the answer field

STRUCTURED OUTPUT EXTRACTION PATTERN:
For information extraction workflows, use two-step approach: document_search → chat_completion with JSON schema

# Step 1: Get relevant content
search_result = await paradigm_client.document_search(query, **search_kwargs)
content = search_result.get("answer", "")

# Step 2: Extract structured data with JSON schema
extraction_prompt = f\"\"\"Extract information and return valid JSON only:

JSON SCHEMA:
{{
  "required_field1": "string or null",
  "required_field2": "number or null", 
  "found": "boolean - true if information was found"
}}

CONTENT: {content}

JSON:\"\"\"

json_result = await paradigm_client.chat_completion(extraction_prompt)

# Parse with fallback to visual search
import json
try:
    data = json.loads(json_result)
    if not data.get("found", False):
        # Try visual search as fallback
        visual_kwargs = search_kwargs.copy()
        visual_kwargs["tool"] = "VisionDocumentSearch"
        visual_result = await paradigm_client.document_search(query, **visual_kwargs)
        visual_content = visual_result.get("answer", "")
        visual_prompt = extraction_prompt.replace(content, visual_content)
        visual_json = await paradigm_client.chat_completion(visual_prompt)
        data = json.loads(visual_json)
except json.JSONDecodeError:
    data = {{"error": "Invalid JSON", "found": False}}

INCORRECT (DON'T DO THIS):
file_ids=attached_file_ids if 'attached_file_ids' in globals() else None  # API doesn't accept None
document_ids = [doc["id"] for doc in search_results.get("documents", [])]  # Should convert to strings
import nltk  # External library not available
answer = search_result["documents"][0].get("content", "")  # Raw content extraction

Generate the complete self-contained workflow code that implements the exact logic described.

CRITICAL: NO PLACEHOLDER CODE - NEVER use 'pass' statements, NEVER use placeholder comments, EVERY function must be fully implemented with working code, ALL code must be ready to execute immediately."""
        
        enhanced_description = f"""
Workflow Description: {description}
Additional Context: {context or 'None'}

Generate a complete, self-contained workflow that:
1. Includes all necessary imports and API client classes
2. Implements the execute_workflow function with the exact logic described
3. Can be copy-pasted and run independently on any server
4. Handles the workflow requirements exactly as specified
"""
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=18000,  # Increased for full code generation
                temperature=0,  # For reproducible results
                system=system_prompt,
                messages=[{"role": "user", "content": enhanced_description}]
            )
            
            code = response.content[0].text
            
            # Log the raw generated code for debugging
            logger.info("🔧 RAW GENERATED CODE:")
            logger.info("=" * 50)
            logger.info(code)
            logger.info("=" * 50)
            
            # Clean up the code - remove markdown formatting if present
            code = self._clean_generated_code(code)
            
            # Log the cleaned code for debugging
            logger.info("🔧 CLEANED GENERATED CODE:")
            logger.info("=" * 50)
            logger.info(code)
            logger.info("=" * 50)
            
            return code
            
        except Exception as e:
            raise Exception(f"Code generation failed: {str(e)}")


    def _clean_generated_code(self, code: str) -> str:
        """
        Clean up generated code by removing markdown formatting and ensuring proper structure
        """
        # Remove markdown code blocks
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        # Remove leading/trailing whitespace
        code = code.strip()
        
        # Ensure execute_workflow is async
        if "def execute_workflow(" in code and "async def execute_workflow(" not in code:
            code = code.replace("def execute_workflow(", "async def execute_workflow(")
        
        return code

    async def enhance_workflow_description(self, raw_description: str) -> Dict[str, Any]:
        """
        Enhance a raw workflow description using Claude AI to create a more detailed,
        actionable workflow specification with proper tool usage and clear steps.
        
        Args:
            raw_description: User's initial natural language workflow description
            
        Returns:
            Dict containing enhanced description, questions, and warnings
        """
        enhancement_prompt = """You are an AI assistant that helps users create detailed workflow descriptions for automation systems.

Your task is to analyze the user's raw workflow description and enhance it into a extremely clear, precise and detailed workflow specification that can be effectively implemented using the available Paradigm API tools.

CRITICAL LANGUAGE PRESERVATION RULE:
- ALWAYS respond in the SAME LANGUAGE as the user's input
- NEVER translate specific terms, document names, field names, or technical terminology
- If the user writes in French, respond entirely in French
- If the user writes in English, respond entirely in English
- Preserve ALL original terminology EXACTLY as provided
- Maintain all specific names, acronyms, and regulatory terms without translation

AVAILABLE PARADIGM API TOOLS:
1. Document Search (paradigm_client.document_search) - Search through documents using natural language queries
2. Document Analysis (paradigm_client.analyze_documents_with_polling) - Analyze specific documents with AI (max 5 documents at once)
3. Chat Completion (paradigm_client.chat_completion) - General AI chat for text processing and analysis
4. Image Analysis (paradigm_client.analyze_image) - Analyze images in documents (max 5 documents at once)

CRITICAL INFORMATION : ENHANCEMENT GUIDELINES:
1. Break down the workflow into clear, specific steps
2. For each step, clearly specify:
   - What action will be performed
   - Which Paradigm API tool will be used
   - What the expected input is
   - What the expected output is
   - Any processing logic needed
   - All conditional logic (if/then/else statements)
   - All rules, constraints, and requirements
   - All edge cases and exception handling
   - All key information required to use the output in further steps : intructions to store key information, how to format it, etc. 

3. CRITICAL: Preserve EVERY detail from the original description with ZERO information loss
4. Capture ALL conditional statements ("if this, then that", "when X occurs, do Y", etc.)
5. Include ALL specific rules, constraints, validation requirements, and business logic
6. Preserve ALL quantities, percentages, dates, formats, and technical specifications
7. Keep ALL specific terms, names, and terminology EXACTLY as provided
8. Document ALL decision points, branching logic, and alternative paths
9. Include ALL error conditions, fallback mechanisms, and exception scenarios
10. Maintain ALL dependencies between steps and prerequisite conditions
11. Capture ALL data validation rules, format requirements, and compliance checks

INFORMATION PRESERVATION REQUIREMENTS:
- Document names (e.g., DC4, JOUE, BOAMP) must remain unchanged
- Field names and section references must be preserved exactly
- Legal and regulatory terms must not be translated
- Company names, addresses, and identifiers must remain intact
- Technical specifications and requirements must be kept verbatim
- ALL conditional logic and if/then statements must be captured
- ALL numerical values, percentages, thresholds must be preserved
- ALL validation rules, format specifications must be included
- ALL error conditions and fallback scenarios must be documented
- ALL business rules and compliance requirements must be maintained
- ALL decision trees and branching logic must be explicit

LIMITATIONS TO CHECK FOR:
- Web searching is NOT available - only document searching within Paradigm
- External API calls (except Paradigm) are NOT available, unless full documentation for these is provided by the user in their initial description
- Complex data processing libraries (pandas, numpy, etc.) are NOT available - try to avoid them if possible, if you do need these, clearly specify what imports are needed in the step description
- Only built-in Python libraries and aiohttp are available

OUTPUT FORMAT:
CRITICAL: Provide your response as PLAIN TEXT ONLY, in a format that will be easy to understand for an LLM.
DO NOT wrap your response in ```json or ``` blocks.
DO NOT use curly braces { } or quotes around your response.
Return the enhanced workflow steps directly in plain text using the step format structure below.

STEP FORMAT STRUCTURE:
For each workflow step, use this exact format:

STEP X: [Highly detailed description of the workflow step with ALL information needed for an LLM to convert the step with all specific requirements (if/then statements, subtle rules, validation logic, API parameters, error conditions, etc.) into very clear code. There should be ABSOLUTELY NO information loss in this step description. Each step should respect the ENHANCEMENT GUIDELINES described above.]

QUESTIONS AND LIMITATIONS: 
- Write "None" if the step is crystal clear and entirely feasible with Paradigm tools alone. Think carefully about potential edge cases and missing information such as "if, then" statements that would clarify these. 
- Otherwise, clearly list:
  * Questions to clarify any ambiguities in the user's description
  * Questions to get extra information needed (external API documentation, business rules, data formats, etc.)
  * Indications that the step requires tools not available (web search, external APIs beyond Paradigm, etc.)

The goal is that STEP X contains everything needed for code generation, and QUESTIONS AND LIMITATIONS only points out what's missing or impossible.

EXAMPLES:

Simple Input: "Search for documents about my question and analyze them"
Plain Text Response:
STEP 1: Search for relevant documents using paradigm_client.document_search with the user's query as the search parameter, setting company_scope=True and private_scope=True to search across all available document collections, and store the returned search results which contain document metadata including IDs, titles, and relevance scores.

QUESTIONS AND LIMITATIONS: None

---

STEP 2: Extract document IDs from the search results by accessing the 'documents' array in the search response, converting each document's 'id' field to string format, and handling the API limitation that maximum 5 documents can be analyzed at once by implementing batching logic if more than 5 documents are found.

QUESTIONS AND LIMITATIONS: None

---

STEP 3: Analyze the found documents using paradigm_client.analyze_documents_with_polling with the user's original question as the analysis query, implementing the polling mechanism with up to 5-minute timeout, processing documents in batches of maximum 5 documents per API call, and collecting all analysis results which contain AI-generated insights based on document content.

QUESTIONS AND LIMITATIONS: None

---

STEP 4: Compile all analysis results from processed documents into a comprehensive summary by combining insights from all batches, formatting the response in clear, readable structure with proper line breaks and organization, including source document references for transparency, and returning the final formatted summary to the user.

QUESTIONS AND LIMITATIONS: None

Now enhance this workflow description and return ONLY the plain text response:"""

        user_message = f"Raw workflow description: {raw_description}"
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-opus-4-1-20250805",
                max_tokens=8000,  # Increased for complex workflows
                temperature=0,  # For reproducible results
                system=enhancement_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            
            result_text = response.content[0].text.strip()
            
            # Parse plain text response
            return {
                "enhanced_description": result_text or raw_description,
                "questions": [],  # Questions are now embedded in each step
                "warnings": []    # Warnings are now embedded in each step
            }
                
        except Exception as e:
            logger.error(f"Failed to enhance workflow description: {str(e)}")
            raise Exception(f"Workflow description enhancement failed: {str(e)}")

    async def _validate_code(self, code: str) -> Dict[str, Any]:
        """
        Enhanced validation for generated code reliability
        """
        try:
            # Basic syntax check
            compile(code, '<string>', 'exec')
            
            # Required function check
            if 'def execute_workflow(' not in code:
                return {"valid": False, "error": "Missing execute_workflow function"}
            
            if 'async def execute_workflow(' not in code:
                return {"valid": False, "error": "execute_workflow must be async"}
            
            # Required imports check
            required_imports = ['import asyncio', 'import aiohttp', 'import json']
            missing_imports = []
            for imp in required_imports:
                if imp not in code:
                    missing_imports.append(imp)
            
            if missing_imports:
                return {"valid": False, "error": f"Missing required imports: {', '.join(missing_imports)}"}
            
            # Common error pattern detection
            error_patterns = [
                ('if.*:\\s*\\n.*=.*await.*\\n.*f.*{.*}', 'Potential UnboundLocalError: variable defined only in conditional block'),
                ('json\\.loads\\([^)]+\\)(?!.*except)', 'Missing error handling for JSON parsing'),
                ('await.*\\.get\\(.*\\).*f.*{.*}.*(?!except)', 'API result used in f-string without null check')
            ]
            
            for pattern, warning in error_patterns:
                if re.search(pattern, code, re.DOTALL):
                    # Don't fail validation, but log warning
                    logger.warning(f"Code pattern warning: {warning}")
            
            return {"valid": True, "error": None}
            
        except SyntaxError as e:
            return {"valid": False, "error": f"Syntax error at line {e.lineno}: {e.msg}"}
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}


# Global generator instance
workflow_generator = WorkflowGenerator()