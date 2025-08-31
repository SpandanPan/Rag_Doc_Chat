#!/bin/bash

# ===============================
# Run all pytest tests with coverage
# ===============================

# Folder where tests are
TEST_DIR="tests"

# Optional: HTML coverage report output folder
COVERAGE_HTML_DIR="htmlcov"

echo "Running pytest for tests in $TEST_DIR ..."

# Run pytest with coverage, verbose, and ignoring deprecation warnings
pytest $TEST_DIR \
    --cov=src/document_ingestion \
    --cov-report=term-missing \
    --cov-report=html:$COVERAGE_HTML_DIR \
    -v \
    -W ignore::DeprecationWarning

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Tests completed successfully."
else
    echo "❌ Some tests failed."
fi

echo "Coverage HTML report is available at $COVERAGE_HTML_DIR/index.html"
