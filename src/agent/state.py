"""面试状态管理"""
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class InterviewPhase(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    ENDED = "ended"


class CompanyLevel(str, Enum):
    BIG = "大厂"
    MID = "中厂"
    SMALL = "小厂"


class DifficultyLevel(str, Enum):
    CAMPUS = "校招"
    SOCIAL = "社招"
    BOTH = "两者兼顾"


# 各难度的考察侧重配置
DIFFICULTY_CONFIG = {
    DifficultyLevel.CAMPUS: {
        "description": "校招候选人，应届或实习经验为主",
        "focus_emphasis": "侧重考察基础知识的理解深度、校园项目/课程项目经历、学习能力与潜力。问题从课本概念、经典原理出发，难易适中。",
        "question_style": "问题清晰直接，以原理理解和知识掌握为主；若涉及项目，多为对校园项目、比赛项目或课程设计的考察。",
        "depth": "基础到进阶",
        "evaluation_hint": "评分时重点看：基础概念是否准确、原理是否理解到位、表达是否清晰有条理。经验不足可以理解，但要看到潜力和学习能力。",
    },
    DifficultyLevel.SOCIAL: {
        "description": "社招候选人，有真实项目工作经验",
        "focus_emphasis": "侧重考察真实项目落地能力、业务场景问题处理、架构设计与工程实践、工作经验中的深度思考。问题多为实际场景、工程实现和遇到过的坑。",
        "question_style": "问题偏向真实业务场景和工程实践，会深入追问项目细节、权衡取舍和踩坑经验，考察候选人的实战深度。",
        "depth": "进阶到高级",
        "evaluation_hint": "评分时重点看：是否有真实落地经验、能否讲清项目中的技术决策与权衡、对线上问题是否有处理经验、是否有自己的思考沉淀。",
    },
    DifficultyLevel.BOTH: {
        "description": "同时考察基础深度与实战经验（校招+社招综合，难度最高）",
        "focus_emphasis": "既要基础扎实，又要能解决复杂工程问题。问题会从原理一路深挖到实际落地，既考概念理解，又考场景设计和复杂问题拆解。",
        "question_style": "问题层层递进，从基础概念逐步加深到复杂场景、系统设计与棘手难题，综合考察广度与深度。",
        "depth": "全栈覆盖 + 高级难",
        "evaluation_hint": "评分时重点看：基础是否真正吃透、能否将理论应用到复杂工程、面对难题时的分析能力和解题思路是否清晰。",
    },
}


# 各公司级别的面试配置
INTERVIEW_CONFIG = {
    CompanyLevel.BIG: {
        "total_rounds": 3,
        "questions_per_round": 6,
        "rounds": [
            {
                "round_name": "第一轮：基础知识 + 项目理解",
                "focus": "基础理论知识和项目经历的初步了解",
                "categories": ["基础理论", "工程实践"],
                "depth": "基础",
            },
            {
                "round_name": "第二轮：技术深度 + 场景设计",
                "focus": "技术深度考察和实际场景设计能力",
                "categories": ["RAG", "微调", "推理优化"],
                "depth": "进阶",
            },
            {
                "round_name": "第三轮：系统设计 + 难题 + 进阶项目深挖",
                "focus": "系统设计能力、难题解决和项目深度挖掘",
                "categories": ["分布式训练", "评测", "工程实践"],
                "depth": "高级",
            },
        ]
    },
    CompanyLevel.MID: {
        "total_rounds": 2,
        "questions_per_round": 5,
        "rounds": [
            {
                "round_name": "第一轮：基础技术问答",
                "focus": "大模型相关基础技术知识",
                "categories": ["基础理论", "RAG", "工程实践"],
                "depth": "基础到进阶",
            },
            {
                "round_name": "第二轮：项目深度挖掘",
                "focus": "项目经历的技术细节和深度理解",
                "categories": ["RAG", "微调", "推理优化"],
                "depth": "进阶",
            },
        ]
    },
    CompanyLevel.SMALL: {
        "total_rounds": 1,
        "questions_per_round": 10,
        "rounds": [
            {
                "round_name": "第一轮：项目理解 + 深度挖掘",
                "focus": "项目经历的全面了解和技术深度",
                "categories": ["基础理论", "RAG", "微调", "工程实践"],
                "depth": "基础到进阶",
            },
        ]
    },
}


@dataclass
class QAItem:
    """单条问答记录"""
    round_num: int
    question_num: int
    question: str
    answer: str
    reference_answer: str = ""  # RAG检索的参考答案
    score: float = 0.0
    evaluation: str = ""  # 点评


@dataclass
class InterviewState:
    """面试状态数据类"""
    phase: InterviewPhase = InterviewPhase.NOT_STARTED
    company_level: CompanyLevel = CompanyLevel.MID

    # 进度
    current_round: int = 1  # 当前第几轮
    current_question: int = 0  # 当前轮第几题
    total_rounds: int = 2
    questions_per_round: int = 5

    # 记录
    qa_history: List[QAItem] = field(default_factory=list)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    current_qa: Optional[QAItem] = None  # 当前正在进行的问答

    # 配置
    difficulty: str = "社招"
    focus_category: Optional[str] = None
    resume_info: Dict = field(default_factory=dict)

    # 最终结论
    final_verdict: str = ""  # 是否录用等

    def add_message(self, role: str, content: str):
        """添加一条对话记录"""
        self.conversation_history.append({"role": role, "content": content})

    def start_new_question(self, question: str):
        """开始一个新问题"""
        self.current_question += 1
        self.current_qa = QAItem(
            round_num=self.current_round,
            question_num=self.current_question,
            question=question,
            answer="",
        )
        self.add_message("assistant", question)

    def record_answer(self, answer: str, reference_answer: str = "", evaluation: str = "", score: float = 0.0):
        """记录候选人的回答"""
        if self.current_qa:
            self.current_qa.answer = answer
            self.current_qa.reference_answer = reference_answer
            self.current_qa.evaluation = evaluation
            self.current_qa.score = score
            self.qa_history.append(self.current_qa)
            self.add_message("user", answer)

    def next_round(self) -> bool:
        """进入下一轮，返回是否还有下一轮"""
        if self.current_round < self.total_rounds:
            self.current_round += 1
            self.current_question = 0
            return True
        return False

    def is_round_complete(self) -> bool:
        """当前轮是否结束"""
        return self.current_question >= self.questions_per_round

    def is_interview_complete(self) -> bool:
        """整个面试是否结束"""
        return self.current_round >= self.total_rounds and self.is_round_complete()

    def get_current_round_config(self) -> Dict:
        """获取当前轮的配置"""
        config = INTERVIEW_CONFIG.get(self.company_level, {})
        rounds = config.get("rounds", [])
        if 1 <= self.current_round <= len(rounds):
            return rounds[self.current_round - 1]
        return {}

    def get_progress_text(self) -> str:
        """获取进度文字"""
        return f"第 {self.current_round}/{self.total_rounds} 轮 · 第 {self.current_question}/{self.questions_per_round} 题"

    def get_difficulty_config(self) -> Dict:
        """获取当前难度档位对应的配置"""
        try:
            level = DifficultyLevel(self.difficulty)
            return DIFFICULTY_CONFIG.get(level, {})
        except ValueError:
            return {}

    def reset(self):
        """重置面试状态"""
        self.phase = InterviewPhase.NOT_STARTED
        self.current_round = 1
        self.current_question = 0
        self.qa_history = []
        self.conversation_history = []
        self.current_qa = None
        self.final_verdict = ""
