# 字符串模板
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("你是一个起名大师，帮我取一个具有{county}特色的男孩名字")
print(prompt.format(county="法国"))
print("============================================================================================================")

# 对话模板
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个起名大师，你的名字叫{name}。"),
        ("human", "你好{name}，你感觉如何？"),
        ("ai", "你好!，我的状态非常好！"),
        ("human", "{user_input}")
    ]
)
print(chat_template.format_messages(name="小明", user_input="你叫什么名字？"))
print("============================================================================================================")

from langchain.prompts import ChatMessagePromptTemplate

prompt = "愿{subject}与你同在！"
chat_message_prompt = ChatMessagePromptTemplate.from_template(role="assistant", template=prompt)
print(chat_message_prompt.format(subject="天使"))

print("============================================================================================================")

# 自定义模板
from langchain_core.prompts import StringPromptTemplate


def hello_world() -> str:
    return "Hello World!"


PROMPT = """
        你是一个非常有经验和天赋的程序员，现在给你如下函数名称，你会按照如下格式，输出这段代码的名称、源代码、中文解释。
        函数名称：{function_name}
        源代码：
        {source_code}
        代码解释：
        """

import inspect


def get_source_code(function_name):
    # 获取源代码
    return inspect.getsource(function_name)


# 自定义模板
class CustomPromptTemplate(StringPromptTemplate):
    def format(self, **kwargs) -> str:
        source_code = get_source_code(kwargs["function_name"])
        prompt = PROMPT.format(function_name=kwargs["function_name"], source_code=source_code)
        return prompt


# input_variables 是 LangChain 框架中 PromptTemplate 或其派生类（如 StringPromptTemplate）的一个固定语法，用于定义模板中允许的占位符变量
# input_variables 是一个列表，表示模板中可以插入的变量名称。所有这些变量必须通过 .format() 方法传入，才能正确渲染模板。
# input_variables 是 LangChain 的设计约定，用于验证模板的输入是否符合预期。如果模板使用了未声明的变量，或传入了多余的变量，LangChain 会抛出错误。
custom_prompt = CustomPromptTemplate(input_variables=["function_name"])
pm = custom_prompt.format(function_name=hello_world)
# print(pm)

from zhipuai import ZhipuAI
import os
from dotenv import load_dotenv, find_dotenv

# find_dotenv() 寻找并定位 .env 文件的路径
# load_dotenv() 读取该 .env 文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

client = ZhipuAI(
    api_key=os.environ.get("ZHIPUAI_API_KEY")
)  # 请填写您自己的APIKey


def completion_api(prompt: str):
    response = client.chat.completions.create(
        # 获取 GPT 模型调用结果
        # 请求参数：
        #     prompt: 对应的提示词
        #     model: 调用的模型，默认为 gpt-3.5-turbo，也可以按需选择 gpt-4 等其他模型
        #     temperature: 模型输出的温度系数，控制输出的随机程度，取值范围是 0~2。温度系数越低，输出内容越一致。
        model="GLM-4-0520",  # 填写需要调用的模型编码
        messages=[
            {"role": "system", "content": "你是一个资深程序专家，你的任务是为用户提供专业、准确的代码解读。"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )
    if len(response.choices) > 0:
        return response.choices[0].message.content
    return "generate answer error"


response = completion_api(prompt=pm)
print(response)


# 流式输出
def completion_api_stream(prompt: str):
    response = client.chat.completions.create(
        # 获取 GPT 模型调用结果
        # 请求参数：
        #     prompt: 对应的提示词
        #     model: 调用的模型，默认为 gpt-3.5-turbo，也可以按需选择 gpt-4 等其他模型
        #     temperature: 模型输出的温度系数，控制输出的随机程度，取值范围是 0~2。温度系数越低，输出内容越一致。
        model="GLM-4-0520",  # 填写需要调用的模型编码
        messages=[
            {"role": "system", "content": "你是一个资深程序专家，你的任务是为用户提供专业、准确的代码解读。"},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    for chunk in response:
        print(chunk.choices[0].delta.content, end="", flush=False)


completion_api_stream(prompt=pm)
