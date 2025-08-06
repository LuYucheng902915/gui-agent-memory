#!/bin/bash

# Local code quality check script
echo "🔍 Starting local code quality checks..."
echo "======================================"

# 1. Code formatting check
echo "📝 Checking code format (Ruff)..."
uv run ruff format --check --diff gui_agent_memory tests
if [ $? -eq 0 ]; then
    echo "✅ Ruff format check passed"
else
    echo "❌ Ruff format check failed, run 'uv run ruff format gui_agent_memory tests' to fix"
fi
echo ""

# 2. Code quality check
echo "🔧 Checking code quality (Ruff Linting)..."
uv run ruff check gui_agent_memory tests
if [ $? -eq 0 ]; then
    echo "✅ Ruff linting check passed"
else
    echo "❌ Ruff linting check failed, run 'uv run ruff check --fix gui_agent_memory tests' to fix"
fi
echo ""

# 3. Type checking
echo "🏷️ Checking type annotations (MyPy)..."
uv run mypy gui_agent_memory tests
if [ $? -eq 0 ]; then
    echo "✅ MyPy check passed"
else
    echo "⚠️ MyPy found type issues, please review above output"
fi
echo ""

# 4. Run tests
echo "🧪 Running tests..."
uv run pytest tests/ -v --tb=short
if [ $? -eq 0 ]; then
    echo "✅ All tests passed"
else
    echo "❌ Some tests failed"
fi
echo ""

echo "======================================"
echo "✨ Local checks completed!"
