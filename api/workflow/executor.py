import asyncio
import time
import sys
import io
import logging
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, Dict, Any, List
from .models import Workflow, WorkflowExecution, ExecutionStatus
from ..config import settings

logger = logging.getLogger(__name__)

class WorkflowExecutor:
    def __init__(self):
        self.max_execution_time = settings.max_execution_time
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
    
    def store_workflow(self, workflow: Workflow) -> None:
        """Store a workflow for later execution"""
        self.workflows[workflow.id] = workflow
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Retrieve a stored workflow"""
        return self.workflows.get(workflow_id)
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Retrieve an execution record"""
        return self.executions.get(execution_id)
    
    async def execute_workflow(self, workflow_id: str, user_input: str, attached_file_ids: Optional[List[int]] = None) -> WorkflowExecution:
        """
        Execute a workflow with given user input
        
        Args:
            workflow_id: ID of the workflow to execute
            user_input: Input data for the workflow
            attached_file_ids: Optional list of file IDs attached to this execution
        
        Returns:
            WorkflowExecution object with results
        """
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        if workflow.status != "ready":
            raise ValueError(f"Workflow {workflow_id} is not ready for execution (status: {workflow.status})")
        
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            user_input=user_input,
            status=ExecutionStatus.RUNNING
        )
        
        self.executions[execution.id] = execution
        
        start_time = time.time()
        
        try:
            # Execute the workflow code
            result = await self._execute_code_safely(workflow.generated_code, user_input, attached_file_ids)
            execution_time = time.time() - start_time
            
            execution.mark_completed(result, execution_time)
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            execution.status = ExecutionStatus.TIMEOUT
            execution.mark_failed("Execution timeout", execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            execution.mark_failed(str(e), execution_time)
        
        return execution
    
    async def execute_code_directly(self, code: str, user_input: str, attached_file_ids: Optional[List[int]] = None) -> WorkflowExecution:
        """
        Execute workflow code directly without requiring a stored workflow.
        
        Args:
            code: Complete workflow code to execute
            user_input: Input data for the workflow
            attached_file_ids: Optional list of file IDs attached to this execution
        
        Returns:
            WorkflowExecution object with results
        """
        execution = WorkflowExecution(
            workflow_id="direct-code-execution",
            user_input=user_input,
            status=ExecutionStatus.RUNNING
        )
        
        self.executions[execution.id] = execution
        
        start_time = time.time()
        
        try:
            # Execute the workflow code
            result = await self._execute_code_safely(code, user_input, attached_file_ids)
            execution_time = time.time() - start_time
            
            execution.mark_completed(result, execution_time)
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            execution.status = ExecutionStatus.TIMEOUT
            execution.mark_failed("Execution timeout", execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            execution.mark_failed(str(e), execution_time)
        
        return execution
    
    async def _execute_code_safely(self, code: str, user_input: str, attached_file_ids: Optional[List[int]] = None) -> str:
        """
        Safely execute the generated workflow code with timeout
        
        Args:
            code: The complete self-contained Python code to execute
            user_input: Input for the workflow
            attached_file_ids: Optional list of attached file IDs
        
        Returns:
            The result from the workflow execution
        """
        # Create execution environment with API keys injected
        execution_globals = self._create_execution_environment(attached_file_ids)
        
        try:
            # Inject actual API keys into the code
            code = self._inject_api_keys(code)
            
            # Compile the code
            compiled_code = compile(code, '<workflow>', 'exec')
            
            # Execute with timeout
            result = await asyncio.wait_for(
                self._run_code_with_capture(compiled_code, execution_globals, user_input),
                timeout=self.max_execution_time
            )
            
            return result
            
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Workflow execution exceeded {self.max_execution_time} seconds")
        except Exception as e:
            raise Exception(f"Workflow execution failed: {str(e)}")
    
    def _inject_api_keys(self, code: str) -> str:
        """
        Inject actual API keys into the generated code
        """
        # Replace placeholder API keys with actual values
        code = code.replace(
            'LIGHTON_API_KEY = "your_api_key_here"',
            f'LIGHTON_API_KEY = "{settings.lighton_api_key}"'
        )
        code = code.replace(
            'ANTHROPIC_API_KEY = "your_anthropic_api_key_here"',
            f'ANTHROPIC_API_KEY = "{settings.anthropic_api_key}"'
        )
        code = code.replace(
            'LIGHTON_BASE_URL = "https://api.lighton.ai"',
            f'LIGHTON_BASE_URL = "{settings.lighton_base_url}"'
        )
        
        return code
    
    async def _run_code_with_capture(self, compiled_code, execution_globals: Dict[str, Any], user_input: str) -> str:
        """
        Run compiled code and capture output
        """
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            logger.info(f"🔧 STARTING CODE EXECUTION")
            logger.info(f"🔧 USER INPUT: {user_input}")
            logger.info(f"🔧 ATTACHED FILE IDS: {execution_globals.get('attached_file_ids', 'None')}")
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the compiled code (this includes all imports, classes, and function definitions)
                logger.info(f"🔧 EXECUTING COMPILED CODE")
                exec(compiled_code, execution_globals)
                
                # Get the execute_workflow function
                if 'execute_workflow' not in execution_globals:
                    raise Exception("execute_workflow function not found in generated code")
                
                workflow_func = execution_globals['execute_workflow']
                logger.info(f"🔧 FOUND WORKFLOW FUNCTION: {workflow_func}")
                
                # Execute the workflow function
                if asyncio.iscoroutinefunction(workflow_func):
                    logger.info(f"🔧 EXECUTING ASYNC WORKFLOW FUNCTION")
                    result = await workflow_func(user_input)
                else:
                    logger.info(f"🔧 EXECUTING SYNC WORKFLOW FUNCTION")
                    result = workflow_func(user_input)
                
                logger.info(f"🔧 WORKFLOW FUNCTION COMPLETED")
                logger.info(f"🔧 RESULT TYPE: {type(result)}")
                logger.info(f"🔧 RESULT: {str(result)[:300]}...")
                
                return str(result) if result is not None else ""
                
        except Exception as e:
            logger.error(f"❌ WORKFLOW EXECUTION ERROR: {str(e)}")
            logger.error(f"❌ ERROR TYPE: {type(e)}")
            
            # Include captured stderr in error message
            stderr_content = stderr_capture.getvalue()
            stdout_content = stdout_capture.getvalue()
            
            if stderr_content:
                logger.error(f"❌ STDERR CONTENT: {stderr_content}")
            if stdout_content:
                logger.info(f"📄 STDOUT CONTENT: {stdout_content}")
            
            if stderr_content:
                raise Exception(f"{str(e)}. Stderr: {stderr_content}")
            raise e
    
    def _create_execution_environment(self, attached_file_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Create a minimal execution environment for self-contained code
        
        The generated code will include all necessary imports and API client classes,
        so we only need to provide basic built-ins and attached file context.
        """
        restricted_globals = {
            '__name__': '__main__',  # Add __name__ for logger setup
            '__builtins__': {
                # Allow safe built-ins only
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'sorted': sorted,
                'reversed': reversed,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'print': print,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'type': type,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'Exception': Exception,
                'RuntimeError': RuntimeError,
                'NameError': NameError,
                '__import__': __import__,
                'any': any,
                'all': all,
                'globals': globals,
                # Essential built-ins for class definitions and code execution
                '__build_class__': __build_class__,
                '__name__': '__main__',
                'object': object,
                'super': super,
                'property': property,
                'staticmethod': staticmethod,
                'classmethod': classmethod,
                'bytes': bytes,
                'bytearray': bytearray,
                'memoryview': memoryview,
                'iter': iter,
                'next': next,
                'slice': slice,
                'map': map,
                'filter': filter,
                'vars': vars,
                'dir': dir,
                'id': id,
                'hash': hash,
                'ord': ord,
                'chr': chr,
                'bin': bin,
                'oct': oct,
                'hex': hex,
                'divmod': divmod,
                'pow': pow,
                'callable': callable,
            },
        }
        
        # Add attached file IDs to the global context
        if attached_file_ids:
            restricted_globals['attached_file_ids'] = attached_file_ids
            restricted_globals['ATTACHED_FILES'] = attached_file_ids  # Also provide as constant
        
        return restricted_globals


# Global executor instance
workflow_executor = WorkflowExecutor()