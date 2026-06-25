from rag_agent.memory import ConversationMemory


def test_memory_keeps_recent_turns_only():
    memory = ConversationMemory(max_turns=2)

    memory.add_turn("什么是列表？", "列表是可变序列。")
    memory.add_turn("什么是元组？", "元组是不可变序列。")
    memory.add_turn("什么是字典？", "字典保存键值对。")

    assert memory.as_messages() == [
        {"role": "user", "content": "什么是元组？"},
        {"role": "assistant", "content": "元组是不可变序列。"},
        {"role": "user", "content": "什么是字典？"},
        {"role": "assistant", "content": "字典保存键值对。"},
    ]


def test_memory_exposes_recent_questions_for_contextual_retrieval():
    memory = ConversationMemory(max_turns=3)
    memory.add_turn("对象是什么", "对象是类创建出来的实例。")
    memory.add_turn("类是什么", "类是对象的模板。")

    assert memory.recent_questions(limit=2) == ["对象是什么", "类是什么"]


def test_memory_ignores_blank_turns():
    memory = ConversationMemory(max_turns=3)

    memory.add_turn("   ", "空问题")
    memory.add_turn("有效问题", "")

    assert memory.as_messages() == []
