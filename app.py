# Streamlit Frontend for LLM-Powered Fact Checker
# Provides a user-friendly interface for fact-checking claims

import streamlit as st
import httpx
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Fact Checker",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .verdict-true {
        background-color: #28a745;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
    }
    .verdict-false {
        background-color: #dc3545;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
    }
    .verdict-unverifiable {
        background-color: #ffc107;
        color: black;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
    }
    .evidence-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .reasoning-box {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Backend URL
BACKEND_URL = "http://localhost:8000"


def check_backend_health():
    """Check if backend is healthy"""
    try:
        response = httpx.get(f"{BACKEND_URL}/health", timeout=5.0)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def fact_check(claim: str):
    """Send claim to backend for fact-checking"""
    try:
        response = httpx.post(
            f"{BACKEND_URL}/check",
            json={"claim": claim},
            timeout=300.0  # 5 minutes - LLM can be slow on CPU
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            return None
    except httpx.TimeoutException:
        st.error("Request timed out. The backend might be processing a large request.")
        return None
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return None


def get_verdict_emoji(verdict: str) -> str:
    """Get emoji for verdict"""
    verdict_lower = verdict.lower()
    if "true" in verdict_lower:
        return "✅"
    elif "false" in verdict_lower:
        return "❌"
    else:
        return "🤷"


def get_verdict_class(verdict: str) -> str:
    """Get CSS class for verdict"""
    verdict_lower = verdict.lower()
    if "true" in verdict_lower:
        return "verdict-true"
    elif "false" in verdict_lower:
        return "verdict-false"
    else:
        return "verdict-unverifiable"


# Main App
def main():
    # Header
    st.markdown('<p class="main-header">🔍 LLM-Powered Fact Checker</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Backend status check
    with st.sidebar:
        st.header("System Status")
        health = check_backend_health()
        if health:
            st.success(f"Backend: {health['status']}")
            st.info(f"Documents: {health['document_count']}")
            st.caption(f"Last check: {health['timestamp']}")
        else:
            st.error("Backend: Offline")
            st.warning("Make sure the backend server is running:\n`uv run main.py`")
    
    # Input Section
    st.subheader("📝 Enter a Claim to Verify")
    
    claim = st.text_area(
        "Claim",
        placeholder="Enter a news statement or claim to fact-check...\n\nExample: The Indian government has announced free electricity to all farmers starting July 2025.",
        height=100,
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        check_button = st.button("🔎 Check Fact", use_container_width=True, type="primary")
    
    # Process claim
    if check_button and claim.strip():
        with st.spinner("Analyzing claim... This may take a moment."):
            result = fact_check(claim)
        
        if result:
            st.markdown("---")
            
            # Verdict
            verdict = result.get("verdict", "Unverifiable")
            emoji = get_verdict_emoji(verdict)
            verdict_class = get_verdict_class(verdict)
            
            st.markdown(
                f'<div class="{verdict_class}">{emoji} {verdict}</div>',
                unsafe_allow_html=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Extracted Claim
            st.subheader("🎯 Extracted Claim")
            st.info(result.get("claim", claim))
            
            # Evidence
            st.subheader("📚 Evidence")
            evidence = result.get("evidence", [])
            if evidence:
                for i, ev in enumerate(evidence, 1):
                    st.markdown(
                        f'<div class="evidence-box"><strong>Evidence {i}:</strong><br>{ev}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No relevant evidence found in the database.")
            
            # Reasoning
            st.subheader("💭 Reasoning")
            reasoning = result.get("reasoning", "No reasoning provided.")
            st.markdown(
                f'<div class="reasoning-box">{reasoning}</div>',
                unsafe_allow_html=True
            )
            
            # Confidence Score
            confidence = result.get("confidence_score")
            if confidence is not None:
                st.caption(f"Confidence Score: {confidence:.2%}")
    
    elif check_button:
        st.warning("Please enter a claim to check.")
    
    # Footer
    st.markdown("---")
    st.caption("Powered by RAG + LLM | Data sources: Press Information Bureau (PIB)")


if __name__ == "__main__":
    main()
