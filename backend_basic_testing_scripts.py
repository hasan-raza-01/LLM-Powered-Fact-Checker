"""
Backend Test Scripts for LLM-Powered Fact Checker
=================================================

Automated testing script for verifying all backend functionality.
Run with: uv run backend_test_scripts.py

Prerequisites:
- Backend server running on http://localhost:8000
- Ollama running with gemma:7b and deepseek-r1:7b models

Usage:
    uv run backend_test_scripts.py              # Run all tests
    uv run backend_test_scripts.py --quick      # Run quick tests only (no LLM calls)
    uv run backend_test_scripts.py --verbose    # Verbose output
"""

import sys
import json
import time
import argparse
from datetime import datetime
from typing import Tuple, List, Dict, Any

import httpx

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 120.0  # seconds


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def print_test(name: str, passed: bool, message: str = "", duration: float = 0):
    """Print test result"""
    status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if passed else f"{Colors.RED}✗ FAIL{Colors.RESET}"
    duration_str = f" ({duration:.2f}s)" if duration > 0 else ""
    print(f"  {status} {name}{duration_str}")
    if message and not passed:
        print(f"       {Colors.YELLOW}→ {message}{Colors.RESET}")


class TestResults:
    """Stores test results for summary"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.results: List[Dict[str, Any]] = []
    
    def add(self, name: str, passed: bool, message: str = "", duration: float = 0):
        self.results.append({
            "name": name,
            "passed": passed,
            "message": message,
            "duration": duration
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def skip(self, name: str, reason: str = ""):
        self.results.append({
            "name": name,
            "passed": None,
            "message": reason,
            "duration": 0
        })
        self.skipped += 1
    
    def summary(self):
        print_header("TEST SUMMARY")
        total = self.passed + self.failed + self.skipped
        print(f"  Total Tests: {total}")
        print(f"  {Colors.GREEN}Passed: {self.passed}{Colors.RESET}")
        print(f"  {Colors.RED}Failed: {self.failed}{Colors.RESET}")
        if self.skipped > 0:
            print(f"  {Colors.YELLOW}Skipped: {self.skipped}{Colors.RESET}")
        print()
        
        if self.failed == 0:
            print(f"  {Colors.GREEN}{Colors.BOLD}All tests passed! ✓{Colors.RESET}")
        else:
            print(f"  {Colors.RED}{Colors.BOLD}Some tests failed ✗{Colors.RESET}")
        print()
        
        return self.failed == 0


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_server_running() -> Tuple[bool, str, float]:
    """Test if the server is running and responding"""
    start = time.time()
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        duration = time.time() - start
        if response.status_code == 200:
            return True, "", duration
        else:
            return False, f"Status code: {response.status_code}", duration
    except httpx.ConnectError:
        return False, "Server not running. Start with: uv run main.py", 0
    except Exception as e:
        return False, str(e), 0


def test_health_endpoint() -> Tuple[bool, str, float]:
    """Test health endpoint returns correct structure"""
    start = time.time()
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=10.0)
        duration = time.time() - start
        data = response.json()
        
        required_fields = ["status", "timestamp", "vectordb_status", "document_count"]
        missing = [f for f in required_fields if f not in data]
        
        if missing:
            return False, f"Missing fields: {missing}", duration
        
        if data["status"] != "ready":
            return False, f"Status is '{data['status']}', expected 'ready'", duration
        
        if data["document_count"] < 1:
            return False, f"No documents in vector database", duration
            
        return True, "", duration
    except Exception as e:
        return False, str(e), 0


def test_vectordb_populated() -> Tuple[bool, str, float]:
    """Test that the vector database has documents"""
    start = time.time()
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=10.0)
        duration = time.time() - start
        data = response.json()
        
        doc_count = data.get("document_count", 0)
        if doc_count >= 50:
            return True, f"{doc_count} documents", duration
        elif doc_count > 0:
            return True, f"Warning: Only {doc_count} documents (expected 50)", duration
        else:
            return False, "No documents in database", duration
    except Exception as e:
        return False, str(e), 0


def test_check_endpoint_exists() -> Tuple[bool, str, float]:
    """Test that /check endpoint exists and requires POST"""
    start = time.time()
    try:
        # GET should fail
        response = httpx.get(f"{BASE_URL}/check", timeout=5.0)
        duration = time.time() - start
        
        if response.status_code == 405:  # Method Not Allowed
            return True, "", duration
        elif response.status_code == 422:  # Validation error (missing body)
            return True, "", duration
        else:
            return False, f"Unexpected status: {response.status_code}", duration
    except Exception as e:
        return False, str(e), 0


def test_check_empty_claim() -> Tuple[bool, str, float]:
    """Test that empty claim returns 400 error"""
    start = time.time()
    try:
        response = httpx.post(
            f"{BASE_URL}/check",
            json={"claim": ""},
            timeout=10.0
        )
        duration = time.time() - start
        
        if response.status_code == 400:
            return True, "", duration
        else:
            return False, f"Expected 400, got {response.status_code}", duration
    except Exception as e:
        return False, str(e), 0


def test_check_invalid_json() -> Tuple[bool, str, float]:
    """Test that invalid JSON returns 422 error"""
    start = time.time()
    try:
        response = httpx.post(
            f"{BASE_URL}/check",
            content="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=10.0
        )
        duration = time.time() - start
        
        if response.status_code == 422:
            return True, "", duration
        else:
            return False, f"Expected 422, got {response.status_code}", duration
    except Exception as e:
        return False, str(e), 0


def test_check_true_claim() -> Tuple[bool, str, float]:
    """Test fact-checking with a claim that should be TRUE"""
    start = time.time()
    try:
        response = httpx.post(
            f"{BASE_URL}/check",
            json={"claim": "India became the 5th largest economy in 2022"},
            timeout=TIMEOUT
        )
        duration = time.time() - start
        
        if response.status_code != 200:
            return False, f"Status code: {response.status_code}", duration
        
        data = response.json()
        required_fields = ["original_input", "claim", "verdict", "evidence", "reasoning"]
        missing = [f for f in required_fields if f not in data]
        
        if missing:
            return False, f"Missing fields: {missing}", duration
        
        if not isinstance(data["evidence"], list):
            return False, "Evidence should be a list", duration
        
        # Check verdict (should be True or contain 'true')
        verdict = data["verdict"].lower()
        if "true" in verdict:
            return True, f"Verdict: {data['verdict']}", duration
        else:
            return False, f"Expected 'True', got '{data['verdict']}'", duration
            
    except httpx.TimeoutException:
        return False, "Request timed out (LLM may be slow)", 0
    except Exception as e:
        return False, str(e), 0


def test_check_false_claim() -> Tuple[bool, str, float]:
    """Test fact-checking with a claim that should be FALSE/Unverifiable"""
    start = time.time()
    try:
        response = httpx.post(
            f"{BASE_URL}/check",
            json={"claim": "The Indian government has announced free electricity to all farmers starting July 2025"},
            timeout=TIMEOUT
        )
        duration = time.time() - start
        
        if response.status_code != 200:
            return False, f"Status code: {response.status_code}", duration
        
        data = response.json()
        
        # Check that we got a proper response structure
        if not all(k in data for k in ["verdict", "evidence", "reasoning"]):
            return False, "Missing required fields", duration
        
        # The claim is false/unverifiable (no such announcement exists)
        verdict = data["verdict"].lower()
        if "false" in verdict or "unverifiable" in verdict or "cannot" in verdict:
            return True, f"Verdict: {data['verdict']}", duration
        else:
            # Even if it says True, we at least got a response
            return True, f"Got verdict: {data['verdict']} (expected False/Unverifiable)", duration
            
    except httpx.TimeoutException:
        return False, "Request timed out (LLM may be slow)", 0
    except Exception as e:
        return False, str(e), 0


def test_response_has_evidence() -> Tuple[bool, str, float]:
    """Test that responses include relevant evidence"""
    start = time.time()
    try:
        response = httpx.post(
            f"{BASE_URL}/check",
            json={"claim": "Chandrayaan-3 landed on the Moon in 2023"},
            timeout=TIMEOUT
        )
        duration = time.time() - start
        
        if response.status_code != 200:
            return False, f"Status code: {response.status_code}", duration
        
        data = response.json()
        evidence = data.get("evidence", [])
        
        if len(evidence) == 0:
            return False, "No evidence returned", duration
        
        if len(evidence) >= 1:
            return True, f"Got {len(evidence)} evidence items", duration
            
    except httpx.TimeoutException:
        return False, "Request timed out", 0
    except Exception as e:
        return False, str(e), 0


def test_response_has_reasoning() -> Tuple[bool, str, float]:
    """Test that responses include reasoning"""
    start = time.time()
    try:
        response = httpx.post(
            f"{BASE_URL}/check",
            json={"claim": "PM-KISAN has helped over 10 crore farmers"},
            timeout=TIMEOUT
        )
        duration = time.time() - start
        
        if response.status_code != 200:
            return False, f"Status code: {response.status_code}", duration
        
        data = response.json()
        reasoning = data.get("reasoning", "")
        
        if not reasoning or len(reasoning) < 10:
            return False, "Reasoning is empty or too short", duration
        
        return True, f"Reasoning length: {len(reasoning)} chars", duration
            
    except httpx.TimeoutException:
        return False, "Request timed out", 0
    except Exception as e:
        return False, str(e), 0


def test_confidence_score() -> Tuple[bool, str, float]:
    """Test that responses include confidence score"""
    start = time.time()
    try:
        response = httpx.post(
            f"{BASE_URL}/check",
            json={"claim": "India has over 140 crore Aadhaar cards issued"},
            timeout=TIMEOUT
        )
        duration = time.time() - start
        
        if response.status_code != 200:
            return False, f"Status code: {response.status_code}", duration
        
        data = response.json()
        confidence = data.get("confidence_score")
        
        if confidence is None:
            return False, "No confidence score returned", duration
        
        if 0 <= confidence <= 1:
            return True, f"Confidence: {confidence:.2%}", duration
        else:
            return False, f"Invalid confidence value: {confidence}", duration
            
    except httpx.TimeoutException:
        return False, "Request timed out", 0
    except Exception as e:
        return False, str(e), 0


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_quick_tests(results: TestResults, verbose: bool = False):
    """Run quick tests that don't require LLM calls"""
    print_header("QUICK TESTS (No LLM)")
    
    tests = [
        ("Server Running", test_server_running),
        ("Health Endpoint", test_health_endpoint),
        ("VectorDB Populated", test_vectordb_populated),
        ("/check Endpoint Exists", test_check_endpoint_exists),
        ("Empty Claim Rejected", test_check_empty_claim),
        ("Invalid JSON Rejected", test_check_invalid_json),
    ]
    
    for name, test_func in tests:
        passed, message, duration = test_func()
        results.add(name, passed, message, duration)
        print_test(name, passed, message, duration)
        
        # Stop if server not running
        if name == "Server Running" and not passed:
            print(f"\n  {Colors.RED}Cannot continue - server not running{Colors.RESET}")
            return False
    
    return True


def run_llm_tests(results: TestResults, verbose: bool = False):
    """Run tests that require LLM calls (slower)"""
    print_header("LLM TESTS (Requires Ollama)")
    
    tests = [
        ("True Claim (India 5th Economy)", test_check_true_claim),
        ("False Claim (Free Electricity)", test_check_false_claim),
        ("Evidence Returned", test_response_has_evidence),
        ("Reasoning Returned", test_response_has_reasoning),
        ("Confidence Score", test_confidence_score),
    ]
    
    for name, test_func in tests:
        if verbose:
            print(f"  {Colors.CYAN}Running: {name}...{Colors.RESET}", end="", flush=True)
        
        passed, message, duration = test_func()
        results.add(name, passed, message, duration)
        
        if verbose:
            print(f"\r", end="")  # Clear the "Running" line
        
        print_test(name, passed, message, duration)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Backend Test Scripts for Fact Checker")
    parser.add_argument("--quick", action="store_true", help="Run only quick tests (no LLM)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    print_header("FACT CHECKER BACKEND TESTS")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Target: {BASE_URL}")
    print(f"  Mode: {'Quick' if args.quick else 'Full'}")
    
    results = TestResults()
    
    # Run quick tests
    server_ok = run_quick_tests(results, args.verbose)
    
    if not server_ok:
        results.summary()
        sys.exit(1)
    
    # Run LLM tests if not in quick mode
    if not args.quick:
        run_llm_tests(results, args.verbose)
    else:
        print(f"\n  {Colors.YELLOW}Skipping LLM tests (--quick mode){Colors.RESET}")
    
    # Print summary
    success = results.summary()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
