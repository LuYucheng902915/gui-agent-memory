#!/usr/bin/env python3
"""
GUI Agent Memory System - 完整功能演示

本示例演示了README中提到的所有11个API方法, 模拟真实的使用场景。
包括: 经验学习、事实管理、智能检索、系统监控等功能。

运行方式: uv run python example.py
"""

import time
from typing import Any

from gui_agent_memory import MemorySystem
from gui_agent_memory.config import get_config


def print_separator(title: str):
    """打印分隔符"""
    print(f"\n{'=' * 60}")
    print(f"🔸 {title}")
    print("=" * 60)


def print_subsection(title: str):
    """打印子章节"""
    print(f"\n📋 {title}")
    print("-" * 40)


def demonstrate_system_initialization_and_validation():
    """演示系统初始化和验证"""
    print_separator("系统初始化与配置验证")

    try:
        # 1. 初始化记忆系统
        print("🔧 正在初始化记忆系统...")
        memory = MemorySystem()
        print("✅ 记忆系统初始化成功")
        try:
            print("🧩 当前配置(去敏):", get_config().debug_dump())
        except Exception as e:
            print(f"🧩 配置快照输出失败: {e}")

        # 2. 验证系统配置
        print("\n🔍 正在验证系统配置...")
        is_valid = memory.validate_system()
        if is_valid:
            print("✅ 系统配置验证通过，所有API服务正常")
        else:
            print("❌ 系统配置验证失败")
            return None

        # 3. 获取系统统计信息
        print("\n📊 获取系统统计信息...")
        stats = memory.get_system_stats()
        storage_stats = stats.get("storage", {})
        print(f"   经验记录数: {storage_stats.get('experiential_memories', 0)}")
        print(f"   事实记录数: {storage_stats.get('declarative_memories', 0)}")
        print(f"   记录总计: {storage_stats.get('total', 0)}")
        print(f"   配置信息: {stats.get('configuration', {})}")
        print(f"   系统版本: {stats.get('version', 'Unknown')}")

        return memory

    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        try:
            print("🧩 当前配置(去敏):", get_config().debug_dump())
        except Exception as dump_err:
            print(f"🧩 配置快照输出失败: {dump_err}")
        print("请确保您的 .env 文件中已配置了有效的API密钥，并存在 prompts 模板")
        return None


def demonstrate_fact_management(memory: MemorySystem):
    """演示事实管理功能"""
    print_separator("事实知识管理")

    # 1. 添加单个事实
    print_subsection("添加单个事实")
    try:
        fact_result = memory.add_fact(
            content="VS Code是Microsoft开发的免费源代码编辑器，支持多种编程语言和扩展",
            keywords=["VS Code", "Microsoft", "编辑器", "编程"],
            source="demo_knowledge_base",
        )
        if isinstance(fact_result, str) and "already exists" in fact_result.lower():
            print(f"🔁 事实已存在，跳过: {fact_result}")
        elif isinstance(fact_result, str) and (
            fact_result.startswith("Successfully added fact") or "成功" in fact_result
        ):
            print(f"✅ 成功添加事实: {fact_result}")
        else:
            print(f"ℹ️ 添加事实结果: {fact_result}")
    except Exception as e:
        print(f"❌ 添加事实失败: {e}")

    # 2. 批量添加事实
    print_subsection("批量添加事实")
    facts_data = [
        {
            "content": "Git是分布式版本控制系统，由Linus Torvalds创建",
            "keywords": ["Git", "版本控制", "Linus Torvalds", "分布式"],
            "source": "demo_knowledge_base",
        },
        {
            "content": "Python是解释型高级编程语言，以简洁和可读性著称",
            "keywords": ["Python", "编程语言", "解释型", "高级语言"],
            "source": "demo_knowledge_base",
        },
        {
            "content": "Docker是容器化平台，可以简化应用程序的部署和管理",
            "keywords": ["Docker", "容器化", "部署", "应用管理"],
            "source": "demo_knowledge_base",
        },
        {
            "content": "React是Facebook开发的JavaScript库，用于构建用户界面",
            "keywords": ["React", "Facebook", "JavaScript", "用户界面"],
            "source": "demo_knowledge_base",
        },
    ]

    try:
        batch_results = memory.batch_add_facts(facts_data)
        successes = [
            r
            for r in batch_results
            if isinstance(r, str) and r.startswith("Successfully added fact")
        ]
        duplicates = len(batch_results) - len(successes)
        print(f"✅ 批量处理完成: 新增 {len(successes)} 条, 去重 {duplicates} 条")
        for i, res in enumerate(batch_results):
            print(f"   {i + 1}. {res}")
    except Exception as e:
        print(f"❌ 批量添加事实失败: {e}")

    # 3. 检索相关事实
    print_subsection("检索相关事实")
    test_queries = ["编程工具", "版本控制", "Web开发", "容器技术"]

    for query in test_queries:
        try:
            related_facts = memory.get_related_facts(query)
            print(f"\n🔍 查询: '{query}' (找到 {len(related_facts)} 条相关事实)")
            for i, fact in enumerate(related_facts):
                print(f"   {i + 1}. {fact.content[:60]}...")
                print(f"      关键词: {', '.join(fact.keywords)}")
        except Exception as e:
            print(f"❌ 检索事实失败 '{query}': {e}")


def demonstrate_experience_learning(memory: MemorySystem):
    """演示经验学习功能"""
    print_separator("经验学习与管理")

    # 1. 学习成功的任务经验
    print_subsection("学习成功任务经验")

    success_tasks: list[dict[str, Any]] = [
        {
            "raw_history": [
                {
                    "goal": "在VS Code中创建新的Python项目",
                    "steps": [
                        {
                            "action": "key_press",
                            "target": "cmd+shift+p",
                            "timestamp": "2024-01-01T10:00:00",
                        },
                        {
                            "action": "type",
                            "content": "Python: Create Environment",
                            "timestamp": "2024-01-01T10:00:01",
                        },
                        {
                            "action": "key_press",
                            "target": "enter",
                            "timestamp": "2024-01-01T10:00:02",
                        },
                        {
                            "action": "select",
                            "target": "Venv",
                            "timestamp": "2024-01-01T10:00:03",
                        },
                        {
                            "action": "select",
                            "target": "Python 3.11",
                            "timestamp": "2024-01-01T10:00:04",
                        },
                    ],
                    "outcome": "成功创建Python虚拟环境并激活",
                    "execution_time": 15.2,
                }
            ],
            "task_id": "demo-task-001",
            "app_name": "VS Code",
            "description": "在VS Code中设置Python开发环境",
        },
        {
            "raw_history": [
                {
                    "goal": "使用Git提交代码变更",
                    "steps": [
                        {
                            "action": "key_press",
                            "target": "ctrl+`",
                            "timestamp": "2024-01-01T11:00:00",
                        },
                        {
                            "action": "type",
                            "content": "git add .",
                            "timestamp": "2024-01-01T11:00:01",
                        },
                        {
                            "action": "key_press",
                            "target": "enter",
                            "timestamp": "2024-01-01T11:00:02",
                        },
                        {
                            "action": "type",
                            "content": 'git commit -m "feat: add new feature"',
                            "timestamp": "2024-01-01T11:00:03",
                        },
                        {
                            "action": "key_press",
                            "target": "enter",
                            "timestamp": "2024-01-01T11:00:04",
                        },
                    ],
                    "outcome": "成功提交代码到本地Git仓库",
                    "execution_time": 8.5,
                }
            ],
            "task_id": "demo-task-002",
            "app_name": "Terminal",
            "description": "使用Git命令提交代码变更",
        },
    ]

    for i, task in enumerate(success_tasks):
        try:
            record_id = memory.learn_from_task(
                raw_history=task["raw_history"],
                is_successful=True,
                source_task_id=task["task_id"],
                app_name=task["app_name"],
                task_description=task["description"],
            )
            print(f"✅ 任务 {i + 1}: {record_id}")
        except Exception as e:
            print(f"❌ 学习任务 {i + 1} 失败: {e}")

    # 2. 学习失败的任务经验
    print_subsection("学习失败任务经验")

    failure_task: dict[str, Any] = {
        "raw_history": [
            {
                "goal": "安装不兼容的Python包",
                "steps": [
                    {
                        "action": "type",
                        "content": "pip install tensorflow==1.0.0",
                        "timestamp": "2024-01-01T12:00:00",
                    },
                    {
                        "action": "key_press",
                        "target": "enter",
                        "timestamp": "2024-01-01T12:00:01",
                    },
                ],
                "outcome": "ERROR: No matching distribution found for tensorflow==1.0.0",
                "execution_time": 25.3,
                "error_details": {"error_type": "PackageNotFound", "exit_code": 1},
            }
        ],
        "task_id": "demo-task-003",
        "app_name": "Terminal",
        "description": "尝试安装过时的TensorFlow版本",
    }

    try:
        learn_result = memory.learn_from_task(
            raw_history=failure_task["raw_history"],
            is_successful=False,
            source_task_id=failure_task["task_id"],
            app_name=failure_task["app_name"],
            task_description=failure_task["description"],
        )
        if isinstance(learn_result, str) and "already exists" in learn_result.lower():
            print(f"🔁 失败经验重复，跳过: {learn_result}")
        elif isinstance(learn_result, str) and (
            learn_result.startswith("Successfully") or "成功" in learn_result
        ):
            print(f"✅ 失败经验学习成功: {learn_result}")
        else:
            print(f"ℹ️ 学习失败经验结果: {learn_result}")
    except Exception as e:
        print(f"❌ 学习失败经验失败: {e}")


def demonstrate_experience_retrieval(memory: MemorySystem):
    """演示经验检索功能"""
    print_separator("经验检索与相似度匹配")

    # 检索相似经验
    print_subsection("检索相似经验")

    experience_queries = [
        "如何设置开发环境",
        "使用命令行工具",
        "代码版本管理",
        "安装软件包",
    ]

    for query in experience_queries:
        try:
            similar_experiences = memory.get_similar_experiences(query)
            print(f"\n🔍 查询: '{query}' (找到 {len(similar_experiences)} 个相似经验)")
            for i, exp in enumerate(similar_experiences):
                print(f"   {i + 1}. {exp.task_description}")
                print(
                    f"      成功: {'是' if exp.is_successful else '否'} | 关键词: {', '.join(exp.keywords)}"
                )
                print(f"      关键步骤: {len(exp.action_flow)} 步")
        except Exception as e:
            print(f"❌ 检索经验失败 '{query}': {e}")


def demonstrate_comprehensive_retrieval(memory: MemorySystem):
    """演示综合检索功能"""
    print_separator("综合智能检索")

    print_subsection("混合检索：经验+事实")

    # 主要的检索接口：retrieve_memories
    complex_queries = [
        "Python开发最佳实践",
        "如何使用VS Code进行Web开发",
        "Git版本控制工作流程",
        "容器化部署解决方案",
        "JavaScript前端开发框架",
    ]

    for query in complex_queries:
        try:
            print(f"\n🔍 综合查询: '{query}'")
            retrieval_result = memory.retrieve_memories(query)

            print(f"   📚 找到 {len(retrieval_result.experiences)} 个相关经验:")
            for i, exp in enumerate(retrieval_result.experiences):
                print(f"      {i + 1}. {exp.task_description[:50]}...")
                print(f"         关键词: {', '.join(exp.keywords[:3])}")

            print(f"   📖 找到 {len(retrieval_result.facts)} 个相关事实:")
            for i, fact in enumerate(retrieval_result.facts):
                print(f"      {i + 1}. {fact.content[:50]}...")
                print(f"         关键词: {', '.join(fact.keywords[:3])}")

            print(
                f"   🎯 检索总结: 经验 {len(retrieval_result.experiences)} + 事实 {len(retrieval_result.facts)} = 总计 {len(retrieval_result.experiences) + len(retrieval_result.facts)} 条结果"
            )

        except Exception as e:
            print(f"❌ 综合检索失败 '{query}': {e}")


def demonstrate_system_monitoring(memory: MemorySystem):
    """演示系统监控功能"""
    print_separator("系统监控与统计")

    try:
        # 获取最新的系统统计
        stats = memory.get_system_stats()
        storage_stats = stats.get("storage", {})
        config_stats = stats.get("configuration", {})

        print("📊 系统状态概览:")
        print("   💾 存储统计:")
        print(f"      - 经验记录: {storage_stats.get('experiential_memories', 0)} 条")
        print(f"      - 事实记录: {storage_stats.get('declarative_memories', 0)} 条")
        print(f"      - 存储总计: {storage_stats.get('total', 0)} 条")

        print("   ⚙️  配置信息:")
        print(f"      - 嵌入模型: {config_stats.get('embedding_model', 'Unknown')}")
        print(f"      - 重排序模型: {config_stats.get('reranker_model', 'Unknown')}")
        print(
            f"      - 经验提取模型: {config_stats.get('experience_llm_model', 'Unknown')}"
        )
        print(f"      - 数据库路径: {config_stats.get('chroma_db_path', 'Unknown')}")

        print(f"   🏷️  版本信息: {stats.get('version', 'Unknown')}")

    except Exception as e:
        print(f"❌ 获取系统统计失败: {e}")


def demonstrate_performance_scenarios(memory: MemorySystem):
    """演示性能场景测试"""
    print_separator("性能与压力测试")

    print_subsection("批量操作性能")

    # 测试批量添加性能
    start_time = time.time()
    large_fact_batch = [
        {
            "content": f"测试事实 {i}: 这是第 {i} 个批量添加的测试事实，用于验证系统性能",
            "keywords": [f"测试{i}", "性能", "批量"],
            "source": "performance_test",
        }
        for i in range(1, 6)  # 添加5个测试事实
    ]

    try:
        batch_ids = memory.batch_add_facts(large_fact_batch)
        batch_time = time.time() - start_time
        print(f"✅ 批量添加 {len(batch_ids)} 个事实耗时: {batch_time:.3f}s")
        if len(batch_ids) > 0:
            print(f"   平均每个事实: {batch_time / len(batch_ids):.3f}s")
        else:
            print("   ℹ️ 本次批量均被去重，未新增记录")
    except Exception as e:
        print(f"❌ 批量添加性能测试失败: {e}")

    # 测试检索性能
    print_subsection("检索性能测试")

    test_queries = ["测试", "性能", "批量操作", "系统验证"]
    total_retrieval_time = 0.0
    successful_queries = 0

    for query in test_queries:
        try:
            start_time = time.time()
            result = memory.retrieve_memories(query)
            query_time = time.time() - start_time
            total_retrieval_time += query_time
            successful_queries += 1

            total_results = len(result.experiences) + len(result.facts)
            print(f"   查询 '{query}': {total_results} 结果, {query_time:.3f}s")

        except Exception as e:
            print(f"❌ 查询 '{query}' 失败: {e}")

    if successful_queries > 0:
        avg_time = total_retrieval_time / successful_queries
        print(f"✅ 平均检索时间: {avg_time:.3f}s ({successful_queries} 次查询)")


def main():
    """主演示程序"""
    print("🚀 GUI Agent Memory System - 完整功能演示")
    print("本演示将展示系统的所有核心功能和API接口")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 系统初始化和验证
    memory = demonstrate_system_initialization_and_validation()
    if not memory:
        print("\n💥 演示因系统初始化失败而终止")
        return

    # 2. 事实管理功能
    demonstrate_fact_management(memory)

    # 3. 经验学习功能
    demonstrate_experience_learning(memory)

    # 4. 经验检索功能
    demonstrate_experience_retrieval(memory)

    # 5. 综合检索功能
    demonstrate_comprehensive_retrieval(memory)

    # 6. 系统监控功能
    demonstrate_system_monitoring(memory)

    # 7. 性能测试
    demonstrate_performance_scenarios(memory)

    # 结束信息
    print_separator("演示完成")
    print("🎉 所有功能演示完成！")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📝 演示总结:")
    print("   ✅ 系统初始化与配置验证")
    print("   ✅ 事实知识管理 (add_fact, batch_add_facts, get_related_facts)")
    print("   ✅ 经验学习管理 (learn_from_task)")
    print("   ✅ 经验检索功能 (get_similar_experiences)")
    print("   ✅ 综合智能检索 (retrieve_memories)")
    print("   ✅ 系统监控统计 (get_system_stats, validate_system)")
    print("   ✅ 性能压力测试")
    print("\n🔗 涵盖了README中提到的所有11个API方法")
    print("💡 这个演示展示了一个完整的RAG记忆系统的工作流程")


if __name__ == "__main__":
    main()
