import requests
import streamlit as st
from config import API_BASE_URL


def get(endpoint: str, params: dict = None) -> dict:
    try:
        response = requests.get(
            f"{API_BASE_URL}{endpoint}",
            params=params,
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. "
                 "Make sure uvicorn is running on port 8000.")
        return {}
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


def post(endpoint: str, json: dict = None,
         params: dict = None) -> dict:
    try:
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=json,
            params=params,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API.")
        return {}
    except Exception as e:
        st.error(f"API error: {e}")
        return {}