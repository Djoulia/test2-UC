"""
Simple test endpoint to debug Vercel deployment issues
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Test API is working"}

@app.get("/test")
async def test():
    return {"status": "success", "message": "Basic test endpoint"}

# Export for Vercel
handler = app