"""面试官Agent核心逻辑"""
from typing import Optional, Dict, List
from openai import OpenAI

from config import LLM_CONFIG
from src.agent.prompts import (
    SYSTEM_PROMPT,
    RESUME_CONTEXT_PROMPT,
    OPENING_PROMPT,
    ROUND_START_PROMPT,
    FOLLOWUP_PROMPT,
    ROUND_END_PROMPT,
    INTERVIEW_END_PROMPT,
    REPORT_PROMPT,
)
from src.agent.state import (
    InterviewState,
    InterviewPhase,
    CompanyLevel,
    INTERVIEW_CONFIG,
)
from src.rag.retriever import get_retriever


class InterviewerAgent:
    """大模型应用开发面试官Agent - 分轮次结构化面试"""

    def __init__(self):
        self.client = OpenAI(
            api_key=LLM_CONFIG["api_key"],
            base_url=LLM_CONFIG["base_url"],
        )
        self.model = LLM_CONFIG["model"]
        self.temperature = LLM_CONFIG["temperature"]
        self.max_tokens = LLM_CONFIG["max_tokens"]
        self.retriever = get_retriever()
        self.state = InterviewState()

    def start_interview(
        self,
        company_level: str = "中厂",
        difficulty: str = "社招",
        focus_category: Optional[str] = None,
        resume_info: Optional[Dict] = None,
    ) -> str:
        """
        开始一场新的面试
        company_level: 大厂 / 中厂 / 小厂
        difficulty: 校招 / 社招 / 两者兼顾
        返回面试官开场白 + 第一个问题
        """
        # 初始化状态
        level = CompanyLevel(company_level)
        config = INTERVIEW_CONFIG[level]

        self.state = InterviewState(
            phase=InterviewPhase.IN_PROGRESS,
            company_level=level,
            total_rounds=config["total_rounds"],
            questions_per_round=config["questions_per_round"],
            difficulty=difficulty,
            focus_category=focus_category,
            resume_info=resume_info or {},
        )

        # 构建简历上下文
        resume_context = self._build_resume_context()

        # 第一轮配置
        round_config = self.state.get_current_round_config()

        # RAG检索
        context_docs = self._retrieve_for_round(round_config)
        context_str = self.retriever.format_context(context_docs)

        # 构建Prompt
        user_msg = OPENING_PROMPT.format(
            company_level=company_level,
            total_rounds=config["total_rounds"],
            questions_per_round=config["questions_per_round"],
            round_name=round_config.get("round_name", ""),
            difficulty=difficulty,
            difficulty_focus=self._get_difficulty_focus(),
            resume_context=resume_context,
            context=context_str,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        response = self._call_llm(messages)

        # 若返回无效，给一个兜底问题
        if not self._is_valid_response(response):
            response = "你好，欢迎参加本次面试。我们先从基础开始：请你简单介绍一下大语言模型（LLM）的基本工作原理，以及你对此的理解。"

        # 记录第一个问题
        self.state.start_new_question(response)
        self.state.add_message("assistant", response)

        return response

    def chat(self, user_input: str) -> str:
        """
        接收候选人回答，返回点评 + 下一个问题
        如果当前轮结束，会宣布进入下一轮
        如果全部结束，会宣布面试结束
        """
        if self.state.phase == InterviewPhase.NOT_STARTED:
            return "面试还未开始，请先点击开始面试。"

        if self.state.phase == InterviewPhase.ENDED:
            return "本轮面试已结束。可以开始新的面试。"

        if not user_input.strip():
            return ""

        # RAG检索当前问题相关知识（用于评估回答）
        context_docs = self.retriever.search(user_input, k=4, category=self.state.focus_category)
        context_str = self.retriever.format_context(context_docs)
        reference_answer = "\n".join([d["content"] for d in context_docs[:3]])

        # 记录回答
        self.state.record_answer(
            answer=user_input,
            reference_answer=reference_answer,
        )

        # 判断当前轮是否结束
        if self.state.is_round_complete():
            # 本轮结束
            if self.state.current_round < self.state.total_rounds:
                # 还有下一轮
                return self._transition_to_next_round(context_str)
            else:
                # 全部结束
                self.state.phase = InterviewPhase.ENDED
                return self._generate_closing()
        else:
            # 继续当前轮，生成下一个问题
            return self._generate_next_question(user_input, context_str)

    def _generate_next_question(self, user_answer: str, context_str: str) -> str:
        """生成下一个问题（带点评）"""
        round_config = self.state.get_current_round_config()
        resume_context = self._build_resume_context()
        asked_questions = self._get_asked_questions_str()

        user_msg = FOLLOWUP_PROMPT.format(
            difficulty=self.state.difficulty,
            difficulty_focus=self._get_difficulty_focus(),
            company_level=self.state.company_level.value,
            current_round=self.state.current_round,
            total_rounds=self.state.total_rounds,
            round_name=round_config.get("round_name", ""),
            round_focus=round_config.get("focus", ""),
            current_question=self.state.current_question,
            questions_per_round=self.state.questions_per_round,
            user_answer=user_answer,
            last_question=self.state.current_qa.question if self.state.current_qa else "",
            context=context_str,
            asked_questions=asked_questions,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + resume_context},
            {"role": "user", "content": user_msg},
        ]

        response = self._call_llm(messages)

        # 若返回无效（空/错误），给一个兜底问题，避免出现空题目卡住流程
        if not self._is_valid_response(response):
            response = "抱歉，网络似乎有点波动。我们继续：请你结合自己的项目经历，谈谈在实际开发中遇到过的最有挑战性的技术问题，以及你是怎么解决的。"

        # 记录新问题
        self.state.start_new_question(response)

        return response

    def _transition_to_next_round(self, context_str: str) -> str:
        """从当前轮过渡到下一轮"""
        next_round_num = self.state.current_round + 1
        next_config = INTERVIEW_CONFIG[self.state.company_level]["rounds"][next_round_num - 1]
        asked_questions = self._get_asked_questions_str()

        # 为下一轮检索相关资料
        next_context_docs = self._retrieve_for_round(next_config)
        next_context_str = self.retriever.format_context(next_context_docs)

        user_msg = ROUND_END_PROMPT.format(
            current_round=self.state.current_round,
            total_rounds=self.state.total_rounds,
            questions_per_round=self.state.questions_per_round,
            difficulty_focus=self._get_difficulty_focus(),
            next_round_info=f"第{next_round_num}轮：{next_config.get('round_name', '')}，考察重点：{next_config.get('focus', '')}",
            context=next_context_str,
            asked_questions=asked_questions,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        response = self._call_llm(messages)

        # 若返回无效，给一个兜底问题
        if not self._is_valid_response(response):
            response = "我们进入下一轮。请你谈谈在项目中使用过哪些技术栈，以及为什么选择它们。"

        # 进入下一轮
        self.state.next_round()
        self.state.start_new_question(response)

        return response

    def _generate_closing(self) -> str:
        """生成面试结束语"""
        user_msg = INTERVIEW_END_PROMPT.format(
            total_rounds=self.state.total_rounds,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        return self._call_llm(messages)

    def generate_report(self) -> str:
        """生成详细的面试评估报告"""
        if not self.state.qa_history:
            return "暂无面试记录，无法生成报告。"

        self.state.phase = InterviewPhase.ENDED

        # 构建每题的记录
        qa_records = ""
        for i, qa in enumerate(self.state.qa_history, 1):
            qa_records += f"""
### 第{i}题（第{qa.round_num}轮）
**问题：** {qa.question}
**候选人回答：** {qa.answer}
**参考答案要点：** {qa.reference_answer[:500] if qa.reference_answer else "（无）"}
"""

        # 各轮名称（动态生成，适配 1/2/3 轮）
        rounds_config = INTERVIEW_CONFIG[self.state.company_level]["rounds"]
        rounds_summary_parts = []
        for i, r in enumerate(rounds_config, start=1):
            rounds_summary_parts.append(
                f"### 第{i}轮：{r.get('round_name', f'第{i}轮')}\n"
                f"- 表现评级：优秀 / 良好 / 一般 / 较差\n"
                f"- 简要评价：..."
            )
        rounds_summary = "\n\n".join(rounds_summary_parts)

        report_prompt = REPORT_PROMPT.format(
            difficulty=self.state.difficulty,
            difficulty_focus=self._get_difficulty_focus(),
            company_level=self.state.company_level.value,
            qa_records=qa_records,
            rounds_summary=rounds_summary,
            total_rounds=self.state.total_rounds,
            total_questions=len(self.state.qa_history),
        )

        messages = [
            {"role": "system", "content": "你是一位专业的技术面试官，擅长生成详细的结构化面试评估报告。"},
            {"role": "user", "content": report_prompt},
        ]

        report = self._call_llm(messages, max_tokens=6000)
        return report

    def end_interview(self):
        """结束面试，返回 (结束语, 评估报告)"""
        self.state.phase = InterviewPhase.ENDED
        closing = "好的，今天的面试就到这里。感谢你的参与，我会为你生成一份详细的面试评估报告..."

        report = self.generate_report()
        return closing, report

    def get_progress(self) -> str:
        """获取当前面试进度"""
        if self.state.phase == InterviewPhase.NOT_STARTED:
            return "面试未开始"
        if self.state.phase == InterviewPhase.ENDED:
            return "面试已结束"
        return self.state.get_progress_text()

    # ===== 内部辅助方法 =====

    def _build_resume_context(self) -> str:
        """构建简历上下文字符串"""
        if not self.state.resume_info:
            return "（无简历信息）"

        return RESUME_CONTEXT_PROMPT.format(
            skills=self.state.resume_info.get("skills", "未明确"),
            projects=self.state.resume_info.get("projects", "未明确"),
            work_experience=self.state.resume_info.get("work_experience", "未明确"),
            education=self.state.resume_info.get("education", "未明确"),
        )

    def _get_difficulty_focus(self) -> str:
        """获取难度档位对应的考察侧重说明"""
        cfg = self.state.get_difficulty_config()
        focus = cfg.get("focus_emphasis", "")
        style = cfg.get("question_style", "")
        hint = cfg.get("evaluation_hint", "")
        return f"考察侧重：{focus}\n出题风格：{style}\n评价侧重：{hint}"

    def _retrieve_for_round(self, round_config: Dict):
        """根据轮次配置检索相关知识库"""
        categories = round_config.get("categories", [])
        depth = round_config.get("depth", "")

        # 难度档对应的检索关键词（校招偏基础，社招偏进阶/工程）
        difficulty_kw = {
            "校招": "基础 原理 理解",
            "社招": "实战 优化 落地 经验",
            "两者兼顾": "综合 进阶 深度",
        }.get(self.state.difficulty, "")

        # 用轮次主题作为查询词
        query = f"{round_config.get('focus', '')} {depth} {difficulty_kw} 面试题"

        # 如果有分类，优先按分类检索
        if categories:
            # 取第一个分类做过滤
            docs = self.retriever.search(query, k=5, category=categories[0])
            if len(docs) < 3:
                # 结果不够，去掉分类限制再搜
                more_docs = self.retriever.search(query, k=5)
                seen_sources = {d["source"] for d in docs}
                for d in more_docs:
                    if d["source"] not in seen_sources:
                        docs.append(d)
                        seen_sources.add(d["source"])
                docs = docs[:5]
        else:
            docs = self.retriever.search(query, k=5)

        return docs

    def _get_asked_questions_str(self) -> str:
        """获取已问问题的字符串，用于避免重复"""
        if not self.state.qa_history:
            return "（暂无）"

        questions = [f"{i+1}. {qa.question}" for i, qa in enumerate(self.state.qa_history)]
        return "\n".join(questions[-10:])  # 只显示最近10个，避免太长

    def _call_llm(self, messages: List[Dict], max_tokens: Optional[int] = None, retries: int = 3) -> str:
        """调用LLM API（带重试机制，避免偶发网络/限流导致返回空）"""
        import time

        last_error = ""
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
                last_error = "LLM 返回空内容"
            except Exception as e:
                last_error = str(e)

            # 重试前短暂等待
            if attempt < retries - 1:
                time.sleep(1)

        return f"[API调用出错: {last_error}]"

    def _is_valid_response(self, text: str) -> bool:
        """判断 LLM 返回内容是否为有效题目（非空且非错误信息）"""
        if not text:
            return False
        if text.startswith("[API调用出错"):
            return False
        return True
