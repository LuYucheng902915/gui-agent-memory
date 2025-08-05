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
                {"action": "key_press", "target": "enter"},
            ],
            "outcome": "成功打开Finder应用",
        }

        # 使用 learn_from_task 方法存储经验
        # 在这个示例中，我们假设任务是成功的，并提供一个唯一的任务ID
        # 注意：raw_history 期望一个列表，我们将任务历史字典包装在列表中
        record_id = memory.learn_from_task(
            raw_history=[task_history], is_successful=True, source_task_id="task-12345"
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
