#!/bin/bash

echo "🚀 Starting Workflow Automation System"
echo "======================================"

# Navigate to project directory
cd "/Users/milo/Desktop/Python projects/cursor tests/cursor_test_5"

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Start backend in background
echo "📡 Starting backend server (http://localhost:8000)..."
python -m api.main &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 5

# Check if backend is running
if curl -s http://localhost:8000/ > /dev/null; then
    echo "✅ Backend is running!"
else
    echo "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start frontend
echo "🌐 Starting frontend server (http://localhost:3000)..."
python start_frontend.py &
FRONTEND_PID=$!

echo ""
echo "✅ System is ready!"
echo "🔗 Open http://localhost:3000 in your browser"
echo "📡 API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "👋 All servers stopped"
    exit 0
}

# Set trap to cleanup on Ctrl+C
trap cleanup INT

# Wait for user interrupt
while true; do
    sleep 1
done