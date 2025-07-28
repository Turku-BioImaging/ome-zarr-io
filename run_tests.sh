#!/bin/bash
# Test runner script for ome-zarr-writer

set -e

echo "OME-Zarr Writer Test Suite"
echo "========================="

# Function to run a command and show result
run_command() {
    echo "Running: $1"
    echo "----------------------------------------"
    if eval "$1"; then
        echo "✓ PASSED"
    else
        echo "✗ FAILED"
        exit 1
    fi
    echo ""
}

# Parse command line arguments
case "${1:-test}" in
    "test"|"")
        echo "Running basic tests (excluding GPU tests)..."
        run_command "pytest tests/ -v -m 'not gpu'"
        ;;
    "cov"|"coverage")
        echo "Running tests with coverage (excluding GPU tests)..."
        run_command "pytest tests/ --cov=src/ome_zarr_writer --cov-report=term-missing -m 'not gpu'"
        ;;
    "gpu")
        echo "Running GPU tests only..."
        run_command "pytest tests/ -v -m 'gpu'"
        ;;
    "all-gpu")
        echo "Running ALL tests including GPU tests..."
        run_command "pytest tests/ -v"
        ;;
    "fast")
        echo "Running fast tests only (excluding GPU and slow tests)..."
        run_command "pytest tests/ -v -m 'not slow and not gpu'"
        ;;
    "examples")
        echo "Running examples..."
        run_command "python examples/schema_example.py"
        run_command "python examples/integration_example.py"
        ;;
    "lint")
        echo "Running linting checks..."
        if command -v flake8 &> /dev/null; then
            run_command "flake8 src/ tests/"
        else
            echo "flake8 not found, skipping..."
        fi
        if command -v mypy &> /dev/null; then
            run_command "mypy src/"
        else
            echo "mypy not found, skipping..."
        fi
        ;;
    "format")
        echo "Formatting code..."
        if command -v black &> /dev/null; then
            run_command "black src/ tests/ examples/"
        else
            echo "black not found, skipping..."
        fi
        ;;
    "all")
        echo "Running full test suite (excluding GPU tests)..."
        run_command "pytest tests/ --cov=src/ome_zarr_writer --cov-report=term-missing -m 'not gpu'"
        echo "Running examples..."
        run_command "python examples/schema_example.py"
        run_command "python examples/integration_example.py"
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  test, (default)  - Run basic tests (excluding GPU tests)"
        echo "  cov, coverage    - Run tests with coverage (excluding GPU tests)"
        echo "  gpu             - Run GPU tests only (requires CUDA/CuPy)"
        echo "  all-gpu         - Run ALL tests including GPU tests"
        echo "  fast            - Run fast tests only (excluding GPU and slow tests)"
        echo "  examples        - Run example scripts"
        echo "  lint            - Run linting checks"
        echo "  format          - Format code with black"
        echo "  all             - Run tests, coverage, and examples (excluding GPU)"
        echo "  help            - Show this help"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac

echo "All tests completed successfully! ✓"
