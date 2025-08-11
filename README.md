# GUI Agent Memory System

🧠 Enhanced Memory Module (RAG-based) for GUI Agent

## 📝 概述

这是一个为GUI智能体设计的增强记忆模块，基于检索增强生成(RAG)技术，提供长期存储、管理和检索功能：

- **情景记忆**：可重用的操作流程和经验
- **语义记忆**：关于系统和应用程序的客观事实和规则

## 🚀 特性

- ✅ 基于ChromaDB的向量存储
- ✅ 支持中文文本处理(jieba分词)
- ✅ 完整的测试覆盖
- ✅ 类型安全(mypy)
- ✅ 现代化代码质量管理(ruff)

## 📦 安装

### 前置条件
- Python >= 3.11
- uv 包管理器

### 安装步骤

```bash
# 克隆项目
git clone <repository-url>
cd gui-agent-memory

# 创建虚拟环境并安装依赖
uv venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装项目依赖
uv sync --all-extras
```

## 🔧 配置

1. 复制环境变量模板：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，配置必要的API密钥：

## 🏃‍♂️ 快速开始

```python
from gui_agent_memory import MemorySystem

def main():
    # 初始化记忆系统
    memory = MemorySystem()

    # 学习一个成功的任务经验
    # 注意：实际使用时，你需要提供真实的API密钥在.env文件中
    try:
        task_history = {
            "goal": "打开文件管理器",
            "steps": [
                {"action": "key_press", "target": "cmd+space"},
                {"action": "type", "content": "finder"},
                {"action": "key_press", "target": "enter"}
            ],
            "outcome": "成功打开Finder应用"
        }

        # 使用 learn_from_task 方法存储经验
        record_id = memory.learn_from_task(
            raw_history=[task_history],
            is_successful=True,
            source_task_id="task-12345"
        )
        print(f"成功学习并存储经验，记录ID: {record_id}")

        # 检索相关经验
        results = memory.retrieve_memories("如何打开文件管理器")
        print("检索结果:")
        print(results)

    except Exception as e:
        print(f"操作失败: {e}")
        print("请确保您的 .env 文件中已配置了有效的API密钥。")

if __name__ == "__main__":
    main()

```

## 🔗 在其他项目中使用

如果要在另一个项目中使用这个记忆包，推荐以下方式：

### 方法1：本地可编辑安装

在主项目目录结构中：
```
your-main-project/
├── agent.py
├── ui_controller.py
├── requirements.txt
└── gui-agent-memory/  # <-- 将本项目作为子模块或直接拷贝
```

在主项目中，通过本地路径进行可编辑安装：
```bash
# 确保在主项目的虚拟环境中
uv pip install -e ./gui-agent-memory
```

### 方法2：通过Git仓库安装 ✅

直接从URL安装：

#### 使用 uv 安装
```bash
# 创建虚拟环境
uv venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 使用 uv 安装
uv pip install git+https://github.com/LuYucheng902915/gui-agent-memory.git
```

安装后可以直接使用：
```python
from gui_agent_memory import MemorySystem

# 创建记忆系统实例
memory = MemorySystem()
```

## 📁 项目结构

```
gui-agent-memory/
├── gui_agent_memory/          # 主要源代码
│   ├── __init__.py           # 包初始化
│   ├── main.py              # 主API接口
│   ├── models.py            # 数据模型
│   ├── config.py            # 配置管理
│   ├── storage.py           # 存储层
│   ├── retriever.py         # 检索层
│   ├── ingestion.py         # 数据摄取
│   └── prompts/             # 提示词模板
├── tests/                    # 测试文件
├── memory_system/           # 数据存储目录
├── pyproject.toml          # 项目配置
├── uv.lock                 # 依赖锁定文件
└── README.md               # 项目文档
```

## 🛠️ 开发

### 代码质量管理

#### 🔧 开发时快速格式化（主动修复）
```bash
# 格式化代码（直接修改文件）
uv run ruff format gui_agent_memory tests

# 修复代码问题（导入排序、代码风格等）
uv run ruff check --fix gui_agent_memory tests

# 类型检查（仅检查，不修改）
uv run mypy gui_agent_memory tests

# 执行所有的测试代码
uv run pytest

# 一次性安装提交与推送钩子（需要网络）
uv run pre-commit install
uv run pre-commit install --hook-type pre-push

# 手动模拟提交阶段（pre-commit hooks 在 commit 阶段执行）
uv run pre-commit run --all-files

# 手动模拟推送阶段（pre-push hooks 在 push 前执行）
uv run pre-commit run --all-files --hook-stage pre-push
```

**使用建议：**
1. **日常开发**：使用 Ruff 快速格式化和修复代码
2. **提交代码**：依赖 pre-commit 自动检查（与本地命令完全一致）

## 📋 API 文档

### MemorySystem 类

#### 核心检索功能
- `retrieve_memories(query: str, top_n: int = 3) -> RetrievalResult` - 检索相关的经验和事实（主要检索接口）
  - **参数说明**: `top_n` 表示**每种类型**最多返回的结果数量，即最多返回 `top_n` 个经验记录 + `top_n` 个事实记录

#### 学习和添加功能
- `learn_from_task(raw_history: list[dict], is_successful: bool, source_task_id: str, app_name: str = "", task_description: str = "") -> str` - 从原始任务历史中学习并存储为经验
- `add_experience(experience: ExperienceRecord) -> str` - 添加预结构化的经验记录
- `add_fact(content: str, keywords: list[str], source: str = "manual") -> str` - 添加语义事实
- `batch_add_facts(facts_data: list[dict]) -> list[str]` - 批量添加多个事实

#### 专门检索功能
- `get_similar_experiences(task_description: str, top_n: int = 5) -> list[ExperienceRecord]` - 获取相似的经验记录
- `get_related_facts(topic: str, top_n: int = 5) -> list[FactRecord]` - 获取相关的事实记录

#### 系统管理功能
- `get_system_stats() -> dict[str, Any]` - 获取系统统计信息
- `validate_system() -> bool` - 验证系统配置和运行状态
- `clear_all_memories() -> str` - 清除所有存储的记忆（测试用途）

#### 便捷函数
- `create_memory_system() -> MemorySystem` - 创建和初始化记忆系统实例

详细API文档请参考代码中的docstring。

## 🤝 贡献

欢迎提交问题和拉取请求！

## 📄 许可证

本项目使用 [MIT License](LICENSE) 授权。

## 🔗 相关链接

- [ChromaDB 文档](https://docs.trychroma.com/)
- [Pydantic 文档](https://docs.pydantic.dev/)
- [uv 包管理器](https://github.com/astral-sh/uv)

## 📞 支持

如果您有任何问题或建议，请：

1. 查看 [Issues](../../issues) 页面
2. 创建新的 Issue
3. 联系维护团队

---

**GUI Agent Memory System** - 让AI记忆更持久，让交互更智能 🚀
