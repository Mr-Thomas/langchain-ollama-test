from pydantic import Field
from typing import Optional, List, Any
from langchain.llms.base import BaseLLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from zhipuai import ZhipuAI
import os
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
_ = load_dotenv(find_dotenv())


class ZhipuAILLM(BaseLLM):
    api_key: str = Field(..., description="ZhipuAI 的 API 密钥")
    client: Optional[ZhipuAI] = None  # 声明 client 为可选字段

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = ZhipuAI(api_key=self.api_key)  # 初始化 client

    def _generate(self, prompt: str, stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None,
                  **kwargs: Any) -> str:
        return self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any) -> str:
        messages = self._create_messages(prompt)
        response = self.client.chat.completions.create(
            model="GLM-4-0520",
            messages=messages,
            temperature=0.1
        )
        if response and response.choices:
            return response.choices[0].message.content
        else:
            raise ValueError("从 ZhipuAI API 未收到有效响应")

    def _create_messages(self, prompt: str) -> List[dict]:
        return [
            {"role": "system", "content": "你是一个法律服务顾问，你的任务是为用户提供专业、准确的回答。"},
            {"role": "user", "content": prompt}
        ]

    def _llm_type(self) -> str:
        return "zhipuai-llm"

# 实例化 ZhipuAILLM 类
# zhipuai_llm = ZhipuAILLM(api_key=os.environ.get("ZHIPUAI_API_KEY"))
#
# # 使用实例生成回答
# prompt = "请告诉我关于量子计算机的最新研究。"
# try:
#     answer = zhipuai_llm._generate(prompt)
#     print(answer)
# except ValueError as e:
#     print(f"Error: {e}")
