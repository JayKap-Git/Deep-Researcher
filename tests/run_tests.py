"""Test runner script for the research application."""

import pytest
import sys
import os
from pathlib import Path

def main():
    """Run all tests in the tests directory."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Add the project root to the Python path
    sys.path.insert(0, str(project_root))
    
    # Run pytest with custom arguments
    pytest.main([
        "tests/backend",  # Test directory
        "-v",  # Verbose output
        "--cov=backend",  # Coverage reporting
        "--cov-report=term-missing",  # Show missing lines in coverage
        "--cov-report=html",  # Generate HTML coverage report
        "--cov-report=xml",  # Generate XML coverage report
        "-p", "no:warnings",  # Suppress warnings
        "--timeout=30",  # Set test timeout
        "--durations=10",  # Show 10 slowest tests
    ])

if __name__ == "__main__":
    main() 