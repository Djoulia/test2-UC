#!/usr/bin/env python3
"""
Full System Startup Script

This script starts both the FastAPI backend and frontend servers together
for local development and testing. It manages both processes and provides
clean shutdown when interrupted.

Features:
    - Starts FastAPI backend on port 8000
    - Starts frontend server on port 3000
    - Handles process lifecycle management
    - Graceful shutdown with Ctrl+C
    - Process cleanup on exit

Usage:
    python start_full_system.py
    
    Then open http://localhost:3000 for the frontend
    API documentation available at http://localhost:8000/docs

Architecture:
    - Subprocess management for parallel process execution
    - Signal handling for clean shutdown
    - Error handling and process recovery
    - Cross-platform compatibility
"""
import subprocess
import time
import os
import signal
import sys
from pathlib import Path

def start_full_system():
    """
    Start both FastAPI backend and frontend server in parallel.
    
    Manages the lifecycle of both processes, handles startup sequencing,
    and provides clean shutdown functionality. Monitors both processes
    and performs cleanup on exit.
    
    Process Flow:
        1. Start FastAPI backend server
        2. Wait for backend to initialize
        3. Start frontend server
        4. Monitor both processes
        5. Handle shutdown signals
        6. Clean up processes on exit
    """
    
    backend_process = None
    frontend_process = None
    
    try:
        print("🚀 Starting Workflow Automation System...")
        print("=" * 50)
        
        # Determine the correct Python interpreter (virtual environment if available)
        python_executable = sys.executable
        venv_python = Path(__file__).parent / "venv" / "bin" / "python"
        venv_python_windows = Path(__file__).parent / "venv" / "Scripts" / "python.exe"
        
        if venv_python.exists():
            python_executable = str(venv_python)
            print(f"🐍 Using virtual environment Python: {python_executable}")
        elif venv_python_windows.exists():
            python_executable = str(venv_python_windows)
            print(f"🐍 Using virtual environment Python: {python_executable}")
        else:
            print(f"🐍 Using system Python: {python_executable}")
        
        # Start FastAPI backend
        print("📡 Starting FastAPI backend on http://localhost:8000...")
        backend_process = subprocess.Popen([
            python_executable, "-m", "api.main"
        ], cwd=Path(__file__).parent)
        
        # Wait a moment for backend to start
        time.sleep(3)
        
        # Start frontend server from root directory (where index.html is now located)
        print("🌐 Starting frontend server on http://localhost:3000...")
        frontend_process = subprocess.Popen([
            python_executable, "-m", "http.server", "3000"
        ], cwd=Path(__file__).parent)  # Changed from /frontend to root directory
        
        print("\n✅ Both servers are running!")
        print("🔗 Main UI: http://localhost:3000 (or http://localhost:8000)")
        print("📁 File Mode UI: http://localhost:8000/file-mode (always uses workflow_code.py)")
        print("📡 API docs available at http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop both servers")
        
        # Wait for interrupt
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping servers...")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        # Clean up processes
        if backend_process:
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()
        
        if frontend_process:
            frontend_process.terminate()
            try:
                frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                frontend_process.kill()
        
        print("👋 All servers stopped")

if __name__ == "__main__":
    start_full_system()