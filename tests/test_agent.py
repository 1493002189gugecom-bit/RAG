from langchain_core.documents import Document

from rag_agent.agent import KnowledgeQAAgent
from rag_agent.memory import ConversationMemory
from rag_agent.retriever import InMemoryKnowledgeBase


class KeywordEmbedding:
    keywords = [
        "list",
        "tuple",
        "dict",
        "function",
        "python",
        "class",
        "object",
        "difference",
        "compare",
        "relationship",
    ]

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        lowered = text.lower()
        return [float(lowered.count(keyword)) for keyword in self.keywords]


class RecordingLLM:
    def __init__(self):
        self.last_prompt = ""

    def generate(self, prompt):
        self.last_prompt = prompt
        return "根据知识库：Python list 是可变序列。"


def test_agent_answers_only_when_knowledge_context_exists():
    kb = InMemoryKnowledgeBase(embedding_model=KeywordEmbedding())
    kb.add_documents(
        [
            Document(
                page_content="Python list 是可变序列。",
                metadata={"source": "list.md"},
            )
        ]
    )
    llm = RecordingLLM()
    agent = KnowledgeQAAgent(kb, llm, ConversationMemory(max_turns=2))

    response = agent.answer("list 是什么？")

    assert "list" in response.answer
    assert response.sources == ["list.md"]
    assert "Python list 是可变序列" in llm.last_prompt


def test_agent_refuses_blank_question():
    kb = InMemoryKnowledgeBase(embedding_model=KeywordEmbedding())
    agent = KnowledgeQAAgent(kb, RecordingLLM(), ConversationMemory(max_turns=2))

    response = agent.answer("  ")

    assert response.answer == "请输入有效问题。"
    assert response.sources == []


def test_agent_refuses_when_no_retrieved_context():
    kb = InMemoryKnowledgeBase(embedding_model=KeywordEmbedding())
    kb.add_documents(
        [
            Document(
                page_content="Python dict 用于保存键值对。",
                metadata={"source": "dict.md"},
            )
        ]
    )
    agent = KnowledgeQAAgent(kb, RecordingLLM(), ConversationMemory(max_turns=2))

    response = agent.answer("路由器硬件如何维修？")

    assert (
        response.answer
        == "知识库中未找到与问题相关的内容，请补充资料后再提问。"
    )
    assert response.sources == []


def test_agent_uses_two_recent_subjects_for_follow_up_comparison():
    kb = InMemoryKnowledgeBase(embedding_model=KeywordEmbedding())
    kb.add_documents(
        [
            Document(
                page_content=(
                    "类 class 是对象 object 的抽象模板，"
                    "对象 object 是类 class 创建出来的实例。"
                ),
                metadata={"source": "class_object.md"},
            ),
            Document(
                page_content=(
                    "list tuple difference compare: list 是可变序列，"
                    "tuple 是不可变序列。"
                ),
                metadata={"source": "list_tuple.md"},
            ),
        ]
    )
    memory = ConversationMemory(max_turns=3)
    memory.add_turn("对象是什么", "对象是 class 的实例。")
    memory.add_turn("类是什么", "类是 object 的模板。")
    agent = KnowledgeQAAgent(kb, RecordingLLM(), memory)

    response = agent.answer("两者有什么关系和区别")

    assert response.sources[0] == "class_object.md"
    assert "list_tuple.md" not in response.sources
    assert "对象、类有什么关系和区别" in agent.llm.last_prompt


def test_agent_resolves_follow_up_relation_without_difference_keyword():
    kb = InMemoryKnowledgeBase(embedding_model=KeywordEmbedding())
    kb.add_documents(
        [
            Document(
                page_content=(
                    "类 class 用于定义对象 object 的属性和方法，"
                    "对象 object 是类 class 的实例。"
                ),
                metadata={"source": "class_object.md"},
            ),
            Document(
                page_content="list tuple difference compare: list 和 tuple 都是有序序列。",
                metadata={"source": "list_tuple.md"},
            ),
        ]
    )
    memory = ConversationMemory(max_turns=3)
    memory.add_turn("类是什么", "类用于定义对象。")
    memory.add_turn("什么是对象", "对象是类的实例。")
    agent = KnowledgeQAAgent(kb, RecordingLLM(), memory)

    response = agent.answer("两者有什么关系")

    assert response.sources == ["class_object.md"]
    assert "类、对象有什么关系" in agent.llm.last_prompt


def test_agent_uses_three_recent_subjects_for_general_follow_up():
    kb = InMemoryKnowledgeBase(embedding_model=KeywordEmbedding())
    kb.add_documents(
        [
            Document(
                page_content=(
                    "列表 list、元组 tuple 和字典 dict "
                    "都是 Python 常用容器类型，"
                    "它们在可变性、存储结构和访问方式上不同。"
                ),
                metadata={"source": "list_tuple_dict.md"},
            ),
            Document(
                page_content=(
                    "function def relationship: 函数用 def 定义，"
                    "可以封装可复用逻辑。"
                ),
                metadata={"source": "function.md"},
            ),
        ]
    )
    memory = ConversationMemory(max_turns=4)
    memory.add_turn("列表是什么", "列表是可变序列。")
    memory.add_turn("元组是什么", "元组是不可变序列。")
    memory.add_turn("字典是什么", "字典保存键值对。")
    agent = KnowledgeQAAgent(kb, RecordingLLM(), memory)

    response = agent.answer("三者有什么关系和区别")

    assert response.sources[0] == "list_tuple_dict.md"
    assert "function.md" not in response.sources
    assert (
        "列表、元组、字典有什么关系和区别"
        in agent.llm.last_prompt
    )


def test_agent_understands_front_three_subjects_wording():
    kb = InMemoryKnowledgeBase(embedding_model=KeywordEmbedding())
    kb.add_documents(
        [
            Document(
                page_content=(
                    "列表 list、元组 tuple 和字典 dict "
                    "都是 Python 容器。"
                ),
                metadata={"source": "containers.md"},
            )
        ]
    )
    memory = ConversationMemory(max_turns=3)
    memory.add_turn("列表是什么", "列表是容器。")
    memory.add_turn("元组是什么", "元组是容器。")
    memory.add_turn("字典是什么", "字典是容器。")
    agent = KnowledgeQAAgent(kb, RecordingLLM(), memory)

    response = agent.answer("前面三个有什么关系")

    assert response.sources == ["containers.md"]
    assert "列表、元组、字典有什么关系" in agent.llm.last_prompt


def test_agent_prioritizes_chunks_whose_heading_matches_context_subjects():
    kb = InMemoryKnowledgeBase(embedding_model=KeywordEmbedding())
    kb.add_documents(
        [
            Document(
                page_content=(
                    "文件对象 object。\n\n"
                    "## 模块导入\n"
                    "from module import name 可以导入指定对象 object。"
                    "class object relationship difference compare."
                ),
                metadata={"source": "module.md"},
            ),
            Document(
                page_content=(
                    "## 类与对象\n"
                    "类 class 用于定义对象 object 的属性和方法，"
                    "对象 object 是类 class 的实例。"
                ),
                metadata={"source": "class_object.md"},
            ),
        ]
    )
    memory = ConversationMemory(max_turns=3)
    memory.add_turn("类是什么", "类用于定义对象。")
    memory.add_turn("对象是什么", "对象是类的实例。")
    agent = KnowledgeQAAgent(kb, RecordingLLM(), memory)

    response = agent.answer("两者有什么关系和区别")

    assert response.sources[0] == "class_object.md"
