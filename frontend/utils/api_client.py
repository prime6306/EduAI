"""
frontend/utils/api_client.py
Thin HTTP client wrapper for calling FastAPI backend from Streamlit.
"""
import streamlit as st
import httpx
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TIMEOUT = 120.0


def _headers() -> dict:
    token = st.session_state.get("token", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def post(endpoint: str, json: dict | None = None, files=None, data=None) -> dict:
    url = f"{BACKEND_URL}{endpoint}"
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            if files:
                resp = client.post(url, headers=_headers(), files=files, data=data)
            else:
                resp = client.post(url, headers=_headers(), json=json)
        if resp.status_code == 200:
            return {"ok": True, "data": resp.json()}
        return {"ok": False, "error": resp.json().get("detail", resp.text)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get(endpoint: str, params: dict | None = None) -> dict:
    url = f"{BACKEND_URL}{endpoint}"
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.get(url, headers=_headers(), params=params or {})
        if resp.status_code == 200:
            return {"ok": True, "data": resp.json()}
        return {"ok": False, "error": resp.json().get("detail", resp.text)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def delete(endpoint: str) -> dict:
    url = f"{BACKEND_URL}{endpoint}"
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.delete(url, headers=_headers())
        if resp.status_code == 200:
            return {"ok": True, "data": resp.json()}
        return {"ok": False, "error": resp.json().get("detail", resp.text)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def require_auth():
    """Call at top of each page to enforce login."""
    if not st.session_state.get("token"):
        st.warning("⚠️ Please log in to continue.")
        st.page_link("Home.py", label="Go to Login →")
        st.stop()
    return st.session_state.get("user", {})
