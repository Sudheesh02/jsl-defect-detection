"""
Launcher script to run the Jindal Stainless AI Surface Defect Detection Platform.
Usage:
    python run_server.py [--port 8000] [--host 0.0.0.0]
"""
import sys
import argparse
from pathlib import Path
import uvicorn

# Ensure root directory is on python path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

def main():
    parser = argparse.ArgumentParser(description="Jindal Stainless AI Defect Inspection Platform")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print("=" * 70)
    print("  JINDAL STAINLESS - AI SURFACE DEFECT DETECTION PLATFORM")
    print(f"  Interactive Dashboard: http://localhost:{args.port}/")
    print(f"  Interactive API Docs:  http://localhost:{args.port}/docs")
    print("=" * 70)

    uvicorn.run(
        "backend.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
