"""
Debug endpoint to check environment variables on Vercel
"""
import os
from fastapi import FastAPI

app = FastAPI()

@app.get("/debug")
async def debug_env():
    return {
        "anthropic_key_present": bool(os.getenv("ANTHROPIC_API_KEY")),
        "lighton_key_present": bool(os.getenv("LIGHTON_API_KEY")),
        "anthropic_key_length": len(os.getenv("ANTHROPIC_API_KEY", "")),
        "lighton_key_length": len(os.getenv("LIGHTON_API_KEY", "")),
        "debug_mode": os.getenv("DEBUG", "false"),
        "all_env_vars": list(os.environ.keys())
    }

# For Vercel
handler = app