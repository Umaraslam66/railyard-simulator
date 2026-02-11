"""
Vercel serverless entry point.
Exposes the Dash app's Flask server as a WSGI application.
"""
import sys
import os

# Add project root to Python path so imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from drl_visualizer import app  # noqa: E402

# Vercel expects a variable named `app` that is a WSGI-compatible application.
# Dash wraps Flask; expose the underlying Flask server.
app = app.server
