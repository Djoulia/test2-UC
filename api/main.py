"""
Main FastAPI Application for Workflow Automation System

This is the core FastAPI application that provides REST API endpoints for:
- Creating workflows from natural language descriptions
- Executing workflows with user input and file attachments
- Managing file uploads and processing
- Handling workflow feedback and regeneration

Key Features:
    - AI-powered workflow generation using Anthropic Claude
    - Document processing via LightOn Paradigm API
    - File upload and management
    - Real-time workflow execution with timeout handling
    - Comprehensive CORS support for web frontends
    - Error handling and logging

API Endpoints:
    - POST /workflows - Create new workflow
    - GET /workflows/{id} - Get workflow details
    - POST /workflows/{id}/execute - Execute workflow
    - POST /files/upload - Upload files for processing
    - File management endpoints for questioning and deletion
    
The application supports cross-domain deployment with multiple frontend origins
and provides comprehensive API documentation via FastAPI's automatic OpenAPI integration.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from .config import settings
from .models import (
    WorkflowCreateRequest,
    WorkflowExecuteRequest,
    WorkflowCodeExecuteRequest,
    WorkflowResponse,
    WorkflowExecutionResponse,
    ErrorResponse,
    FileUploadResponse,
    FileInfoResponse,
    FileQuestionRequest,
    FileQuestionResponse,
    WorkflowWithFilesRequest,
    WorkflowDescriptionEnhanceRequest,
    WorkflowDescriptionEnhanceResponse,
    TestExample,
    AutomatedTestRequest,
    AutomatedTestResponse,
    TestResult,
)
from .workflow.generator import workflow_generator
from .workflow.executor import workflow_executor
from .workflow.models import Workflow, WorkflowExecution, ExecutionStatus
from .api_clients import paradigm_client  # Updated import

# Configure logging based on debug settings
logging.basicConfig(level=logging.INFO if settings.debug else logging.WARNING)
logger = logging.getLogger(__name__)

# In-memory storage for test examples (session-only)
session_test_examples: Dict[str, List[TestExample]] = {}

# API key validation helpers
def validate_anthropic_api_key():
    """
    Validate that Anthropic API key is available.
    
    Returns:
        bool: True if API key is available
        
    Raises:
        HTTPException: 503 if API key is missing
    """
    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=503,
            detail="Anthropic API key not configured. Please set ANTHROPIC_API_KEY environment variable."
        )
    return True

def validate_lighton_api_key():
    """
    Validate that LightOn API key is available.
    
    Returns:
        bool: True if API key is available
        
    Raises:
        HTTPException: 503 if API key is missing
    """
    if not settings.lighton_api_key:
        raise HTTPException(
            status_code=503,
            detail="LightOn API key not configured. Please set LIGHTON_API_KEY environment variable."
        )
    return True

# Create FastAPI app with comprehensive metadata
app = FastAPI(
    title="Workflow Automation API",
    description="API for creating and executing automated workflows using AI",
    version="1.0.0",
    debug=settings.debug
)

# Create API router with /api prefix
from fastapi import APIRouter
api_router = APIRouter()

# Add CORS middleware for cross-domain frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://127.0.0.1:3000",
        "https://scaffold-ai-test2.vercel.app",  # Production frontend
        "https://scaffold-ai-test2-milo-rignells-projects.vercel.app",  # Your current deployment
        "https://scaffold-ai-test2-fi4dvy1xl-milo-rignells-projects.vercel.app",
        "https://scaffold-ai-test2-tawny.vercel.app",  # Your other deployment
        "https://scaffold-ai-test2-git-main-milo-rignells-projects.vercel.app/",
        "https://*.vercel.app",  # All Vercel deployments
        "https://*.netlify.app",  # Netlify deployments
        "https://*.github.io",   # GitHub Pages
        "https://*.surge.sh",    # Surge deployments
        "https://*.firebaseapp.com"  # Firebase hosting
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def serve_frontend():
    """
    Serve the frontend HTML page.
    
    Returns the main application interface when accessing the root URL.
    """
    try:
        # Try to read the index.html file from the project root
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        # Fallback to API info if index.html not found
        return {
            "message": "Workflow Automation API",
            "version": "1.0.0", 
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "note": "Frontend file not found - API only mode"
        }

@app.get("/file-mode", response_class=HTMLResponse, tags=["Frontend"])
async def serve_file_mode_frontend():
    """
    Serve the file-mode frontend HTML page.
    
    Returns the file-mode application interface that always uses workflow_code.py.
    """
    try:
        # Try to read the file-workflow.html file from the project root
        with open("file-workflow.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        # Fallback if file-workflow.html not found
        return HTMLResponse(
            content="<h1>File-mode frontend not found</h1><p>Please ensure file-workflow.html exists.</p>",
            status_code=404
        )

@app.get("/lighton-logo.png", tags=["Static"])
async def serve_logo():
    """
    Serve the LightOn logo image.
    """
    try:
        with open("lighton-logo.png", "rb") as f:
            image_data = f.read()
        return Response(content=image_data, media_type="image/png")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Logo not found")

@app.get("/health", tags=["Health"]) 
async def health_check():
    """
    Health check endpoint for monitoring.
    
    Provides service status information for deployment platforms.
    """
    return {
        "message": "Workflow Automation API",
        "version": "1.0.0",
        "status": "healthy", 
        "timestamp": datetime.utcnow().isoformat()
    }

@api_router.post("/workflows/enhance-description", response_model=WorkflowDescriptionEnhanceResponse, tags=["Workflows"])
async def enhance_workflow_description(request: WorkflowDescriptionEnhanceRequest):
    """
    Enhance a raw workflow description using Claude AI.
    
    Takes a user's initial natural language workflow description and transforms it
    into a detailed, actionable workflow specification with clear steps, proper
    tool usage, and identification of any missing information or limitations.
    
    Args:
        request: Enhancement request containing the raw workflow description
        
    Returns:
        WorkflowDescriptionEnhanceResponse: Enhanced description with questions and warnings
        
    Raises:
        HTTPException: 503 if API keys are missing, 500 if enhancement fails
        
    Example:
        POST /workflows/enhance-description
        {
            "description": "Search for documents and analyze them"
        }
        
        Returns enhanced description with specific steps and tool usage details.
    """
    # Validate required API keys
    validate_anthropic_api_key()
    
    try:
        logger.info(f"Enhancing workflow description: {request.description[:100]}...")
        
        # Enhance the description
        result = await workflow_generator.enhance_workflow_description(request.description)
        
        logger.info("Workflow description enhanced successfully")
        
        return WorkflowDescriptionEnhanceResponse(
            enhanced_description=result["enhanced_description"],
            questions=result["questions"],
            warnings=result["warnings"]
        )
        
    except Exception as e:
        logger.error(f"Failed to enhance workflow description: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to enhance workflow description: {str(e)}"
        )

@api_router.post("/workflows", response_model=WorkflowResponse, tags=["Workflows"])
async def create_workflow(request: WorkflowCreateRequest):
    """
    Create a new workflow from a natural language description.
    
    This endpoint uses AI to generate executable Python code from a natural
    language workflow description. The generated code integrates with both
    Anthropic and LightOn Paradigm APIs for document processing and analysis.
    
    Args:
        request: Workflow creation request containing description, optional name, and context
        
    Returns:
        WorkflowResponse: Complete workflow details including generated code
        
    Raises:
        HTTPException: 503 if API keys are missing, 500 if workflow generation fails
        
    Example:
        POST /workflows
        {
            \"description\": \"Search documents about AI, then analyze findings\",
            \"name\": \"AI Research Workflow\"
        }
    """
    # Validate required API keys
    validate_anthropic_api_key()
    
    try:
        logger.info(f"Creating workflow: {request.description[:100]}...")
        
        # Generate the workflow
        workflow = await workflow_generator.generate_workflow(
            description=request.description,
            name=request.name,
            context=request.context
        )
        
        # Store the workflow in the executor
        workflow_executor.store_workflow(workflow)
        
        logger.info(f"Workflow created successfully: {workflow.id}")
        
        return WorkflowResponse(
            id=workflow.id,
            name=workflow.name,
            description=workflow.description,
            status=workflow.status,
            generated_code=workflow.generated_code,
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
            error=workflow.error
        )
        
    except Exception as e:
        logger.error(f"Failed to create workflow: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create workflow: {str(e)}"
        )

@api_router.get("/workflows/{workflow_id}", response_model=WorkflowResponse, tags=["Workflows"])
async def get_workflow(workflow_id: str):
    """
    Retrieve details of a specific workflow by ID.
    
    Returns complete workflow information including generated code,
    current status, and metadata. Used to check workflow status
    and retrieve generated code for inspection.
    
    Args:
        workflow_id: Unique identifier of the workflow to retrieve
        
    Returns:
        WorkflowResponse: Complete workflow details
        
    Raises:
        HTTPException: 404 if workflow not found, 500 for other errors
    """
    try:
        workflow = workflow_executor.get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow {workflow_id} not found"
            )
        
        return WorkflowResponse(
            id=workflow.id,
            name=workflow.name,
            description=workflow.description,
            status=workflow.status,
            generated_code=workflow.generated_code,
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
            error=workflow.error
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow {workflow_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get workflow: {str(e)}"
        )

@api_router.post("/workflows/{workflow_id}/execute", response_model=WorkflowExecutionResponse, tags=["Execution"])
async def execute_workflow(workflow_id: str, request: WorkflowExecuteRequest):
    """
    Execute a workflow with user input and optional file attachments.
    
    Runs the generated workflow code with the provided user input.
    Supports file attachments that can be processed within the workflow.
    Execution is performed in a secure, isolated environment with timeout protection.
    
    Args:
        workflow_id: ID of the workflow to execute
        request: Execution request with user input and optional file IDs
        
    Returns:
        WorkflowExecutionResponse: Execution results with status and timing
        
    Raises:
        HTTPException: 400 for validation errors, 500 for execution failures
        
    Note:
        Execution timeout is configured via settings.max_execution_time (default: 5 minutes)
    """
    try:
        logger.info(f"Executing workflow {workflow_id} with input: {request.user_input[:100]}...")
        
        # Execute the workflow
        execution = await workflow_executor.execute_workflow(workflow_id, request.user_input, request.attached_file_ids)
        
        logger.info(f"Workflow execution completed: {execution.id} (status: {execution.status})")
        
        return WorkflowExecutionResponse(
            workflow_id=execution.workflow_id,
            execution_id=execution.id,
            result=execution.result,
            status=execution.status.value,
            execution_time=execution.execution_time,
            error=execution.error,
            created_at=execution.created_at
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to execute workflow {workflow_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute workflow: {str(e)}"
        )

@api_router.post("/workflows/{workflow_id}/apply-feedback", response_model=WorkflowResponse, tags=["Workflows"])
async def apply_feedback_to_workflow(workflow_id: str, feedback: str = Body(..., embed=True)):
    """
    Apply user feedback to modify an existing workflow's generated code.
    
    Takes user feedback and regenerates the workflow code while maintaining
    all original requirements and following code generation best practices.
    
    Args:
        workflow_id: ID of the workflow to improve
        feedback: User feedback describing desired changes
        
    Returns:
        WorkflowResponse: Updated workflow with improved code
        
    Raises:
        HTTPException: 404 if workflow not found, 500 if improvement fails
    """
    # Validate required API keys
    validate_anthropic_api_key()
    
    try:
        # Get the original workflow
        original_workflow = workflow_executor.get_workflow(workflow_id)
        if not original_workflow:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow {workflow_id} not found"
            )
        
        logger.info(f"Applying feedback to workflow {workflow_id}: {feedback[:100]}...")
        
        # Create enhanced description with feedback for regeneration
        enhanced_description = f"""
ORIGINAL WORKFLOW: {original_workflow.description}

CURRENT GENERATED CODE:
{original_workflow.generated_code}

USER FEEDBACK FOR IMPROVEMENT:
{feedback}

INSTRUCTIONS: Modify the workflow code based on the user feedback while:
1. Maintaining all original workflow requirements and functionality
2. Following all code generation standards and best practices
3. Preserving the self-contained nature of the code
4. Keeping all imports and API client implementations
5. Maintaining the async execute_workflow function signature
6. Following structured output patterns when extracting information
7. Including visual search fallback mechanisms where applicable

Generate the improved workflow code that incorporates the user feedback."""
        
        # Generate improved workflow
        improved_workflow = await workflow_generator.generate_workflow(
            description=enhanced_description,
            name=f"Improved: {original_workflow.name or 'Workflow'}",
            context={
                "feedback_iteration": True,
                "original_workflow_id": workflow_id,
                "user_feedback": feedback,
                **(original_workflow.context or {})
            }
        )
        
        # Store the improved workflow
        workflow_executor.store_workflow(improved_workflow)
        
        logger.info(f"Workflow improved successfully: {improved_workflow.id}")
        
        return WorkflowResponse(
            id=improved_workflow.id,
            name=improved_workflow.name,
            description=improved_workflow.description,
            status=improved_workflow.status,
            generated_code=improved_workflow.generated_code,
            created_at=improved_workflow.created_at,
            updated_at=improved_workflow.updated_at,
            error=improved_workflow.error
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to apply feedback to workflow {workflow_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to apply feedback: {str(e)}"
        )

async def evaluate_test_result_with_ai(
    test_example: TestExample,
    actual_output: str,
    execution_error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate test result using Claude AI as a judge.
    
    Args:
        test_example: The test example with validation criteria
        actual_output: The actual workflow output
        execution_error: Any error that occurred during execution
        
    Returns:
        Dict containing evaluation result (passed: bool, feedback: str)
    """
    try:
        system_prompt = """You are an AI judge evaluating workflow test results. 

Your task is to determine if a workflow execution passes the user's validation criteria and provide feedback for improvement.

EVALUATION GUIDELINES:
1. Compare the actual output against the user's validation criteria
2. Consider both content and format requirements
3. Be precise but fair in evaluation
4. For partial matches, consider the intent and practical usefulness
5. If execution failed with an error, automatically fail the test

OUTPUT FORMAT:
Return ONLY a JSON object with this exact structure:
{
  "passed": boolean,
  "feedback": "detailed feedback explaining the evaluation decision and suggestions for improvement"
}

EVALUATION PRINCIPLES:
- If validation criteria mentions "all X should be Y", check that ALL instances meet the requirement
- If criteria specifies a format (JSON, table, etc.), verify the output matches that format
- If criteria includes specific content requirements, verify those are present
- Consider semantic meaning, not just exact text matches
- For error cases, explain what went wrong and suggest fixes"""

        if execution_error:
            evaluation_content = f"""
TEST VALIDATION CRITERIA:
{test_example.validation_criteria}

EXPECTED OUTPUT (if provided):
{test_example.expected_output or "None provided"}

ACTUAL EXECUTION RESULT:
EXECUTION FAILED WITH ERROR: {execution_error}

ACTUAL OUTPUT: {actual_output or "No output due to error"}

Evaluate if this test passes the validation criteria."""
        else:
            evaluation_content = f"""
TEST VALIDATION CRITERIA:
{test_example.validation_criteria}

EXPECTED OUTPUT (if provided):
{test_example.expected_output or "None provided"}

ACTUAL OUTPUT:
{actual_output}

Evaluate if this test passes the validation criteria."""

        response = workflow_generator.anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": evaluation_content}]
        )
        
        result_text = response.content[0].text.strip()
        
        # Parse JSON response
        import json
        try:
            result = json.loads(result_text)
            return {
                "passed": result.get("passed", False),
                "feedback": result.get("feedback", "Evaluation completed")
            }
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "passed": False,
                "feedback": f"AI evaluation failed to parse: {result_text[:200]}..."
            }
            
    except Exception as e:
        logger.error(f"AI evaluation failed: {str(e)}")
        return {
            "passed": False,
            "feedback": f"Evaluation error: {str(e)}"
        }

@api_router.post("/workflows/automated-test", response_model=AutomatedTestResponse, tags=["Workflows"])
async def run_automated_workflow_testing(request: AutomatedTestRequest):
    """
    Run automated workflow testing and improvement cycle.
    
    Executes test examples against workflow code, evaluates results using AI,
    and iteratively improves the code based on test failures until all tests
    pass or maximum iterations are reached.
    
    Args:
        request: Automated testing request with workflow code and test examples
        
    Returns:
        AutomatedTestResponse: Results including improved code and test outcomes
        
    Raises:
        HTTPException: 503 if API keys are missing, 400 for validation errors
    """
    # Validate required API keys
    validate_anthropic_api_key()
    
    if not request.test_examples:
        raise HTTPException(status_code=400, detail="At least one test example is required")
    
    try:
        logger.info(f"Starting automated testing with {len(request.test_examples)} test examples")
        
        current_code = request.workflow_code
        iteration_history = []
        total_iterations = 0
        all_tests_passed = False
        problematic_tests = []
        stopped_reason = "unknown"
        
        # Determine iteration limit based on mode
        if request.iteration_mode == "fixed_iterations":
            max_iterations = request.fixed_iterations or 1
        else:
            max_iterations = request.max_iterations or 10
            
        logger.info(f"Testing mode: {request.iteration_mode}, max iterations: {max_iterations}")
        
        while total_iterations < max_iterations and not all_tests_passed:
            total_iterations += 1
            iteration_start_time = datetime.utcnow()
            
            logger.info(f"Starting iteration {total_iterations}/{max_iterations}")
            
            # Run all test examples
            test_results = []
            failed_tests = []
            
            for test_example in request.test_examples:
                try:
                    logger.info(f"Running test: {test_example.id}")
                    
                    # Execute workflow with test input
                    execution_start_time = datetime.utcnow()
                    execution = await workflow_executor.execute_code_directly(
                        code=current_code,
                        user_input=test_example.query,
                        attached_file_ids=test_example.attached_file_ids or []
                    )
                    execution_time = (datetime.utcnow() - execution_start_time).total_seconds()
                    
                    # Evaluate result with AI
                    evaluation = await evaluate_test_result_with_ai(
                        test_example=test_example,
                        actual_output=execution.result or "",
                        execution_error=execution.error
                    )
                    
                    test_result = TestResult(
                        test_id=test_example.id,
                        passed=evaluation["passed"],
                        output=execution.result,
                        evaluation_feedback=evaluation["feedback"],
                        execution_time=execution_time,
                        error=execution.error
                    )
                    
                    test_results.append(test_result)
                    
                    if not evaluation["passed"]:
                        failed_tests.append({
                            "test_id": test_example.id,
                            "feedback": evaluation["feedback"],
                            "error": execution.error
                        })
                        
                    logger.info(f"Test {test_example.id}: {'PASSED' if evaluation['passed'] else 'FAILED'}")
                    
                except Exception as e:
                    logger.error(f"Test execution failed for {test_example.id}: {str(e)}")
                    test_result = TestResult(
                        test_id=test_example.id,
                        passed=False,
                        output=None,
                        evaluation_feedback=f"Test execution failed: {str(e)}",
                        execution_time=0,
                        error=str(e)
                    )
                    test_results.append(test_result)
                    failed_tests.append({
                        "test_id": test_example.id,
                        "feedback": f"Execution error: {str(e)}",
                        "error": str(e)
                    })
            
            # Check if all tests passed
            all_tests_passed = all(result.passed for result in test_results)
            
            # Store iteration history
            iteration_history.append({
                "iteration": total_iterations,
                "timestamp": iteration_start_time.isoformat(),
                "all_tests_passed": all_tests_passed,
                "passed_tests": len([r for r in test_results if r.passed]),
                "total_tests": len(test_results),
                "test_results": [result.dict() for result in test_results]
            })
            
            if all_tests_passed:
                stopped_reason = "all_passed"
                logger.info(f"All tests passed after {total_iterations} iterations!")
                break
                
            # If not all tests passed and we have more iterations, improve the code
            if total_iterations < max_iterations:
                logger.info(f"Improving code based on {len(failed_tests)} failed tests")
                
                # Generate improvement feedback
                improvement_feedback = "Based on test failures, please improve the workflow code:\n\n"
                for failed_test in failed_tests:
                    improvement_feedback += f"Test '{failed_test['test_id']}' failed: {failed_test['feedback']}\n"
                    if failed_test['error']:
                        improvement_feedback += f"Error: {failed_test['error']}\n"
                    improvement_feedback += "\n"
                
                improvement_feedback += "\nPlease fix these issues while maintaining all original functionality."
                
                # Use existing feedback application mechanism to improve code
                enhanced_description = f"""
CURRENT WORKFLOW CODE TO IMPROVE:
{current_code}

TEST FAILURE FEEDBACK:
{improvement_feedback}

INSTRUCTIONS: Modify the workflow code to address all test failures while:
1. Maintaining all original functionality  
2. Following all code generation standards and best practices
3. Preserving the self-contained nature of the code
4. Keeping all imports and API client implementations
5. Maintaining the async execute_workflow function signature
6. Addressing each specific test failure mentioned above

Generate the improved workflow code."""

                try:
                    improved_workflow = await workflow_generator.generate_workflow(
                        description=enhanced_description,
                        name=f"Auto-improved Iteration {total_iterations}",
                        context={
                            "automated_testing": True,
                            "iteration": total_iterations,
                            "failed_tests": [f["test_id"] for f in failed_tests]
                        }
                    )
                    
                    current_code = improved_workflow.generated_code
                    logger.info(f"Code improved for iteration {total_iterations + 1}")
                    
                except Exception as e:
                    logger.error(f"Code improvement failed: {str(e)}")
                    stopped_reason = "improvement_failed"
                    break
        
        # Determine final stop reason
        if stopped_reason == "unknown":
            if total_iterations >= max_iterations:
                stopped_reason = "max_iterations"
            
        # Identify consistently problematic tests (failed in multiple iterations)
        if len(iteration_history) > 1:
            test_fail_counts = {}
            for iteration in iteration_history:
                for test_result in iteration["test_results"]:
                    if not test_result["passed"]:
                        test_id = test_result["test_id"]
                        test_fail_counts[test_id] = test_fail_counts.get(test_id, 0) + 1
            
            # Tests that failed in multiple iterations are problematic
            problematic_tests = [
                test_id for test_id, fail_count in test_fail_counts.items() 
                if fail_count > 1 and fail_count >= len(iteration_history) * 0.5
            ]
        
        final_response = AutomatedTestResponse(
            improved_workflow_code=current_code,
            total_iterations=total_iterations,
            all_tests_passed=all_tests_passed,
            test_results=test_results,
            iteration_history=iteration_history,
            stopped_reason=stopped_reason,
            problematic_tests=problematic_tests if problematic_tests else None
        )
        
        logger.info(f"Automated testing completed: {stopped_reason}, {total_iterations} iterations")
        return final_response
        
    except Exception as e:
        logger.error(f"Automated testing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Automated testing failed: {str(e)}"
        )

@api_router.post("/workflows/execute-code", response_model=WorkflowExecutionResponse, tags=["Execution"])
async def execute_workflow_code(request: WorkflowCodeExecuteRequest):
    """
    Execute workflow code either from UI or from project's workflow_code.py file.
    
    If request.code is provided and non-empty, executes code from UI.
    If request.code is empty/None, reads and executes code from workflow_code.py file.
    This allows both UI-based testing and file-based workflow execution.
    
    Args:
        request: Code execution request containing optional code, user input, and optional file IDs
        
    Returns:
        WorkflowExecutionResponse: Execution results including status, timing, and output
        
    Raises:
        HTTPException: If no code source is available or execution fails
        
    Note:
        The code will have access to global 'attached_file_ids' variable
        containing the list of file IDs from request.attached_file_ids
    """
    try:
        logger.info(f"Executing workflow code")
        logger.info(f"User input: {request.user_input[:100]}...")
        if request.attached_file_ids:
            logger.info(f"Attached files: {request.attached_file_ids}")
        
        # Determine code source: UI code or file-based code
        if request.code and request.code.strip():
            # Use code from UI request
            workflow_code = request.code
            code_source = "UI"
            workflow_id = "ui-code-execution"
            logger.info(f"Using code from UI: {len(workflow_code)} characters")
        else:
            # Read workflow code from the project file
            try:
                with open("workflow_code.py", "r", encoding="utf-8") as f:
                    workflow_code = f.read()
                code_source = "File"
                workflow_id = "file-based-execution"
                logger.info(f"Using code from workflow_code.py: {len(workflow_code)} characters")
            except FileNotFoundError:
                raise HTTPException(
                    status_code=400, 
                    detail="No workflow code provided in request and workflow_code.py file not found. Either provide code in the request or create workflow_code.py file with your workflow code."
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to read workflow_code.py: {str(e)}"
                )
        
        # Execute the workflow code using workflow executor's safe execution method
        execution = await workflow_executor.execute_code_directly(workflow_code, request.user_input, request.attached_file_ids)
        
        logger.info(f"Code execution completed from {code_source}: {execution.id} (status: {execution.status})")
        
        return WorkflowExecutionResponse(
            workflow_id=workflow_id,
            execution_id=execution.id,
            result=execution.result,
            status=execution.status.value,
            execution_time=execution.execution_time,
            error=execution.error,
            created_at=execution.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Code execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Code execution failed: {str(e)}")

@api_router.get("/workflows/{workflow_id}/executions/{execution_id}", response_model=WorkflowExecutionResponse, tags=["Execution"])
async def get_execution(workflow_id: str, execution_id: str):
    """
    Retrieve details of a specific workflow execution.
    
    Returns execution results, status, timing information, and any errors
    that occurred during execution. Used for monitoring and debugging
    workflow executions.
    
    Args:
        workflow_id: ID of the parent workflow
        execution_id: Unique identifier of the execution to retrieve
        
    Returns:
        WorkflowExecutionResponse: Complete execution details
        
    Raises:
        HTTPException: 404 if execution not found, 400 if execution doesn't belong to workflow
    """
    try:
        execution = workflow_executor.get_execution(execution_id)
        if not execution:
            raise HTTPException(
                status_code=404,
                detail=f"Execution {execution_id} not found"
            )
        
        if execution.workflow_id != workflow_id:
            raise HTTPException(
                status_code=400,
                detail=f"Execution {execution_id} does not belong to workflow {workflow_id}"
            )
        
        return WorkflowExecutionResponse(
            workflow_id=execution.workflow_id,
            execution_id=execution.id,
            result=execution.result,
            status=execution.status.value,
            execution_time=execution.execution_time,
            error=execution.error,
            created_at=execution.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get execution {execution_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get execution: {str(e)}"
        )

# File upload and management endpoints

@api_router.post("/files/upload", response_model=FileUploadResponse, tags=["Files"])
async def upload_file(
    file: UploadFile = File(...),
    collection_type: str = Form("private"),
    workspace_id: Optional[int] = Form(None)
):
    """
    Upload a file to Paradigm for document processing and analysis.
    
    Files are automatically processed, indexed, and made available for use
    in workflows. Supports various document formats and collection types
    for organizing files within different scopes.
    
    Args:
        file: The file to upload (multipart/form-data)
        collection_type: Collection scope - 'private', 'company', or 'workspace'
        workspace_id: Required if collection_type is 'workspace'
        
    Returns:
        FileUploadResponse: File metadata including ID, size, and processing status
        
    Raises:
        HTTPException: 503 if API keys are missing, 500 if upload fails
        
    Note:
        Files are processed asynchronously and may take time to become fully searchable
    """
    # Validate required API keys
    validate_lighton_api_key()
    
    try:
        logger.info(f"Uploading file: {file.filename}")
        
        # Read file content
        file_content = await file.read()
        
        # Upload to Paradigm
        result = await paradigm_client.upload_file(
            file_content=file_content,
            filename=file.filename,
            collection_type=collection_type,
            workspace_id=workspace_id
        )
        
        logger.info(f"File uploaded successfully: {result.get('id')}")
        
        return FileUploadResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to upload file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file: {str(e)}"
        )

@api_router.get("/files/{file_id}", response_model=FileInfoResponse, tags=["Files"])
async def get_file_info(file_id: int, include_content: bool = False):
    """
    Retrieve metadata and optionally content of an uploaded file.
    
    Provides file information including processing status, size, and creation time.
    Can optionally include the full file content for inspection.
    
    Args:
        file_id: Unique identifier of the file
        include_content: Whether to include file content in response
        
    Returns:
        FileInfoResponse: File metadata and optionally content
        
    Raises:
        HTTPException: 503 if API keys are missing, 500 if retrieval fails
    """
    # Validate required API keys
    validate_lighton_api_key()
    
    try:
        result = await paradigm_client.get_file_info(file_id, include_content)
        return FileInfoResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to get file info for {file_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get file info: {str(e)}"
        )

@api_router.post("/files/{file_id}/ask", response_model=FileQuestionResponse, tags=["Files"])
async def ask_question_about_file(file_id: int, request: FileQuestionRequest):
    """
    Ask a natural language question about a specific uploaded file.
    
    Uses AI to analyze the file content and provide answers to user questions.
    Returns both the answer and relevant document chunks for transparency.
    
    Args:
        file_id: ID of the file to question
        request: Question request containing the natural language query
        
    Returns:
        FileQuestionResponse: AI-generated answer with supporting document chunks
        
    Raises:
        HTTPException: 503 if API keys are missing, 500 if question processing fails
    """
    # Validate required API keys
    validate_lighton_api_key()
    
    try:
        result = await paradigm_client.ask_question_about_file(file_id, request.question)
        return FileQuestionResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to ask question about file {file_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ask question: {str(e)}"
        )

@api_router.delete("/files/{file_id}", tags=["Files"])
async def delete_file(file_id: int):
    """
    Delete an uploaded file from the system.
    
    Permanently removes the file and all associated metadata from Paradigm.
    The file will no longer be available for workflows or questioning.
    
    Args:
        file_id: ID of the file to delete
        
    Returns:
        dict: Success status and confirmation message
        
    Raises:
        HTTPException: 503 if API keys are missing, 500 if deletion fails
        
    Warning:
        This operation is irreversible
    """
    # Validate required API keys
    validate_lighton_api_key()
    
    try:
        success = await paradigm_client.delete_file(file_id)
        return {"success": success, "message": f"File {file_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to delete file {file_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete file: {str(e)}"
        )

@api_router.post("/workflows-with-files", response_model=WorkflowResponse, tags=["Workflows"])
async def create_workflow_with_files(request: WorkflowWithFilesRequest):
    """
    Create a workflow that has access to specific uploaded files.
    
    Generates workflow code that can process and analyze the specified
    uploaded files. The file IDs are embedded in the workflow context
    so the generated code can reference them directly.
    
    Args:
        request: Workflow creation request with file IDs to attach
        
    Returns:
        WorkflowResponse: Complete workflow details with file access capabilities
        
    Raises:
        HTTPException: 503 if API keys are missing, 500 if workflow generation fails
        
    Note:
        Generated workflow will have access to global 'attached_file_ids' variable
    """
    # Validate required API keys
    validate_anthropic_api_key()
    
    try:
        logger.info(f"Creating workflow with files: {request.uploaded_file_ids}")
        
        # Add file IDs to context
        context = request.context or {}
        if request.uploaded_file_ids:
            context["uploaded_file_ids"] = request.uploaded_file_ids
            context["use_uploaded_files"] = True
        
        # Generate the workflow
        workflow = await workflow_generator.generate_workflow(
            description=request.description,
            name=request.name,
            context=context
        )
        
        # Store the workflow in the executor
        workflow_executor.store_workflow(workflow)
        
        logger.info(f"Workflow with files created successfully: {workflow.id}")
        
        return WorkflowResponse(
            id=workflow.id,
            name=workflow.name,
            description=workflow.description,
            status=workflow.status,
            generated_code=workflow.generated_code,
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
            error=workflow.error
        )
        
    except Exception as e:
        logger.error(f"Failed to create workflow with files: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create workflow with files: {str(e)}"
        )


# Include the API router in the main app
app.include_router(api_router, prefix="/api")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors.
    
    Catches all unhandled exceptions and returns a consistent error response.
    In debug mode, includes detailed error information for troubleshooting.
    In production mode, returns generic error messages to avoid information leakage.
    
    Args:
        request: The HTTP request that caused the exception
        exc: The unhandled exception
        
    Returns:
        ErrorResponse: Standardized error response with timestamp
        
    Note:
        All exceptions are logged for monitoring and debugging purposes
    """
    logger.error(f"Unhandled exception: {str(exc)}")
    return ErrorResponse(
        error="Internal server error",
        details=str(exc) if settings.debug else None,
        timestamp=datetime.utcnow()
    )

# Development server entry point
if __name__ == "__main__":
    import uvicorn
    # Run the development server with auto-reload in debug mode
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )