from langchain.prompts import load_prompt

# 加载yaml格式的prompt模板
prompt = load_prompt(path="./simple_prompt.yaml", encoding="utf-8")
print(prompt.format(name="LangChain", what="LLM"))

# 加载json格式的prompt模板
prompt = load_prompt(path="./simple_prompt.json", encoding="utf-8")
print(prompt.format(name="LangChain", what="LLM"))

# 根据输入相似度选择示例(最大边际相关性)[相关且多样]
# · MMR是一种在信息检索中常用的方法,它的目标是在相关性和多样性之间找到一个平衡
# · MMR会首先找出与输入最相似(即余弦相似度最大)的样本
# · 然后在迭代添加样本的过程中,对于与已选择样本过于接近(即相似度过高)的样本进行惩罚
# · MMR既能确保选出的样本与输入高度相关,又能保证选出的样本之间有足够的多样性
# · 关注如何在相关性和多样性之间找到一个平衡

# conda install -c pytorch faiss-cpu=1.8.0

# 使用MMR来检索相关示例
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import ZhipuAIEmbeddings
import os
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate

_ = load_dotenv(find_dotenv())

# 假设已存在示例
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
    {"input": "高兴", "output": "悲伤"}
]

# 构建提示词模板
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

# 调用MMR示例选择器
mmr_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples=examples,
    embeddings=ZhipuAIEmbeddings(api_key=os.environ.get("ZHIPUAI_API_KEY")),
    vectorstore_cls=FAISS,
    k=2
)

# 使用小样本模板
mmrShot_prompt = FewShotPromptTemplate(
    example_selector=mmr_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"]
)
print(mmrShot_prompt.format(adjective="worried"))

print("\n=================================================================")

# 根据输入相似度选择示例(最大余弦相似度)[相关]
# · 一种常见的相似度计算方法
# · 它通过计算两个向量(在这里,向量可以代表文本、句子或词语)之间的余弦值来衡量它们的相似度
# · 余弦值越接近1,表示两个向量越相似
# · 主要关注的是如何准确衡量两个向量的相似度

# 根据最大余弦相似度检索相关示例
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=ZhipuAIEmbeddings(api_key=os.environ.get("ZHIPUAI_API_KEY")),
    vectorstore_cls=Chroma,
    k=1
)
fewShot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"]
)

print(fewShot_prompt.format(adjective="worried"))
