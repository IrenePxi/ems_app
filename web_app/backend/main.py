"""
FastAPI backend for EMS Web Application
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from datetime import date, datetime
from typing import Optional, Dict, List
import uvicorn

from api import data, devices, ems, optimization

app = FastAPI(title="EMS Web Application", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(devices.router, prefix="/api/devices", tags=["devices"])
app.include_router(ems.router, prefix="/api/ems", tags=["ems"])
app.include_router(optimization.router, prefix="/api/optimization", tags=["optimization"])

@app.get("/")
async def root():
    return {"message": "EMS Web Application API", "version": "1.0.0"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
