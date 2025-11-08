#!/usr/bin/env python3
"""
Wrapper script to run Streamlit app from any directory.
"""
import sys
from pathlib import Path

# Add SurgiControl to path
sys.path.insert(0, str(Path(__file__).parent / "SurgiControl"))

# Change to SurgiControl directory for proper path resolution
import os
os.chdir(Path(__file__).parent / "SurgiControl")

# Now run streamlit
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    
    # Get the path to streamlit_app.py
    app_path = Path(__file__).parent / "SurgiControl" / "streamlit_app.py"
    
    # Run streamlit
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())

