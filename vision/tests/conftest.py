"""Tests never look for a real video bucket (an empty VIDEO_BUCKET turns that off); tests that need one set it."""
import os

os.environ.setdefault("VIDEO_BUCKET", "")
