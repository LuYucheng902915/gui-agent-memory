#!/bin/bash

# 本地代码质量检查脚本 - 绕过网络问题
echo "🔍 开始本地代码质量检查..."
echo "======================================"

# 1. Black 格式检查
echo "📝 检查代码格式 (Black)..."
uv run black --check --diff gui_agent_memory tests
if [ $? -eq 0 ]; then
    echo "✅ Black 检查通过"
else
    echo "❌ Black 检查失败，运行 'uv run black gui_agent_memory tests' 修复"
fi
echo ""

# 2. isort 导入排序检查
echo "📦 检查导入排序 (isort)..."
uv run isort --check-only --diff gui_agent_memory tests
if [ $? -eq 0 ]; then
    echo "✅ isort 检查通过"
else
    echo "❌ isort 检查失败，运行 'uv run isort gui_agent_memory tests' 修复"
fi
echo ""

# 3. Flake8 代码质量检查（忽略行长度问题）
echo "🔧 检查代码质量 (Flake8)..."
uv run flake8 gui_agent_memory tests --max-line-length=88 --extend-ignore=E203,W503,E501
if [ $? -eq 0 ]; then
    echo "✅ Flake8 检查通过"
else
    echo "⚠️ Flake8 发现问题，请查看上方输出"
fi
echo ""

# 4. MyPy 类型检查（宽松模式）
echo "🏷️ 检查类型注解 (MyPy)..."
uv run mypy gui_agent_memory --ignore-missing-imports --no-strict-optional
if [ $? -eq 0 ]; then
    echo "✅ MyPy 检查通过"
else
    echo "⚠️ MyPy 发现类型问题，但不影响运行"
fi
echo ""

# 5. 运行测试
echo "🧪 运行测试..."
uv run pytest tests/ -v --tb=short
if [ $? -eq 0 ]; then
    echo "✅ 所有测试通过"
else
    echo "❌ 有测试失败"
fi
echo ""

echo "======================================"
echo "✨ 本地检查完成！"
