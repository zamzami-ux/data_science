"""
Entry point for Streamlit Cloud deployment.
Delegates execution directly to app.py.
"""
import runpy

runpy.run_path("app.py", run_name="__main__")
