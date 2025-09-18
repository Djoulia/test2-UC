"""
Application Configuration Settings

This module defines the configuration settings for the workflow automation system.
It loads environment variables and provides default values for all configurable parameters.

Environment Variables:
    ANTHROPIC_API_KEY: API key for Anthropic Claude AI service
    LIGHTON_API_KEY: API key for LightOn Paradigm service
    DEBUG: Enable debug mode (true/false)
    HOST: Server host address (default: 0.0.0.0)
    PORT: Server port number (default: 8000)

Features:
    - Environment variable loading via python-dotenv
    - Configuration validation
    - Default values for all settings
    - LightOn Paradigm API endpoint configuration
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """
    Application settings configuration class.
    
    Loads configuration from environment variables and provides validation.
    All settings have sensible defaults and can be overridden via environment variables.
    """
    
    def __init__(self):
        # Core API keys - required for operation
        self.anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
        self.lighton_api_key: str = os.getenv("LIGHTON_API_KEY", "")
        
        # Server configuration
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8000"))
        
        # LightOn Paradigm API settings
        self.lighton_base_url: str = "https://paradigm.lighton.ai"
        self.lighton_docsearch_endpoint: str = "/api/v2/chat/document-search"
        
        # Workflow execution settings
        self.max_execution_time: int = 1800  # 20 minutes maximum execution time
        self.max_workflow_steps: int = 50   # Maximum number of workflow steps
        
    def validate(self) -> None:
        """
        Validate that required settings are present.
        
        Raises:
            ValueError: If any required API key is missing
        """
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")
        if not self.lighton_api_key:
            raise ValueError("LIGHTON_API_KEY is required")

# Global settings instance - used throughout the application
settings = Settings()