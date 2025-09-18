"""
Workflow Domain Models

This module defines the core domain models for the workflow system:
- Workflow: Represents a workflow with generated code and metadata
- WorkflowExecution: Tracks individual workflow execution instances
- ExecutionStatus: Enumeration of possible execution states
- CodeGenerationContext: Context for code generation operations

Key Features:
    - Dataclass-based models for immutability and type safety
    - UUID-based unique identifiers
    - Comprehensive status tracking with timestamps
    - Execution timing and error handling
    - Context management for code generation

Architecture:
    - Domain-driven design with clear separation of concerns
    - Status management with proper state transitions
    - Comprehensive metadata tracking for debugging and monitoring
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import uuid

class ExecutionStatus(str, Enum):
    """
    Enumeration of workflow execution status values.
    
    States:
        PENDING: Execution has been queued but not started
        RUNNING: Execution is currently in progress
        COMPLETED: Execution finished successfully
        FAILED: Execution failed due to an error
        TIMEOUT: Execution was terminated due to timeout
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class Workflow:
    """
    Represents a workflow with AI-generated code and metadata.
    
    A workflow contains the user's description, AI-generated Python code,
    status information, and timing data. Workflows can be executed multiple
    times with different inputs.
    
    Attributes:
        id: Unique identifier (UUID)
        name: Optional human-readable name
        description: Natural language description provided by user
        generated_code: AI-generated Python code for execution
        status: Current workflow status
        created_at: Timestamp of creation
        updated_at: Timestamp of last modification
        error: Error message if workflow creation failed
        context: Additional context used during code generation
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    description: str = ""
    generated_code: Optional[str] = None
    status: str = "created"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
    def update_status(self, status: str, error: Optional[str] = None):
        """
        Update workflow status and timestamp.
        
        Updates the workflow status, sets the updated timestamp to current time,
        and optionally records an error message.
        
        Args:
            status: New status value
            error: Optional error message to record
        """
        self.status = status
        self.updated_at = datetime.utcnow()
        if error:
            self.error = error

@dataclass
class WorkflowExecution:
    """
    Represents a single execution instance of a workflow.
    
    Tracks the execution of a workflow with specific user input,
    including results, timing, and any errors that occurred.
    
    Attributes:
        id: Unique execution identifier (UUID)
        workflow_id: ID of the parent workflow
        user_input: Input provided by user for this execution
        result: Execution result or None if not completed
        status: Current execution status
        execution_time: Time taken to execute (seconds)
        error: Error message if execution failed
        created_at: Timestamp when execution started
        completed_at: Timestamp when execution finished (if applicable)
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    user_input: str = ""
    result: Optional[str] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    execution_time: Optional[float] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def mark_completed(self, result: str, execution_time: float):
        """
        Mark execution as successfully completed.
        
        Sets the execution status to COMPLETED, records the result,
        execution time, and completion timestamp.
        
        Args:
            result: The execution result string
            execution_time: Time taken to execute in seconds
        """
        self.result = result
        self.status = ExecutionStatus.COMPLETED
        self.execution_time = execution_time
        self.completed_at = datetime.utcnow()
    
    def mark_failed(self, error: str, execution_time: Optional[float] = None):
        """
        Mark execution as failed with error details.
        
        Sets the execution status to FAILED, records the error message,
        optional execution time, and completion timestamp.
        
        Args:
            error: Error message describing the failure
            execution_time: Optional time taken before failure (seconds)
        """
        self.error = error
        self.status = ExecutionStatus.FAILED
        self.execution_time = execution_time
        self.completed_at = datetime.utcnow()

@dataclass
class CodeGenerationContext:
    """
    Context information for AI code generation.
    
    Provides configuration and context data used during the
    code generation process to influence the AI's output.
    
    Attributes:
        available_tools: List of tool names available to generated code
        max_steps: Maximum number of workflow steps allowed
        timeout_seconds: Maximum execution time allowed
        additional_context: Extra context data for specialized workflows
    """
    """Context for code generation"""
    available_tools: List[str] = field(default_factory=lambda: ["paradigm_search", "chat_completion"])
    max_steps: int = 50
    timeout_seconds: int = 300
    additional_context: Optional[Dict[str, Any]] = None