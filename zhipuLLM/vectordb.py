from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from zhipuai import ZhipuAI
from dotenv import load_dotenv, find_dotenv
import os

# find_dotenv() 寻找并定位 .env 文件的路径
# load_dotenv() 读取该 .env 文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

client = ZhipuAI(
    api_key=os.environ.get("ZHIPUAI_API_KEY")
)  # 请填写您自己的APIKey


class VectorDB:
    embedding = ZhipuAIEmbeddings()
    persist_directory = '../test/db'
    slice = 120

    def __init__(self, sliced_text: list = None):
        assert sliced_text is not None
        self.vectordb = Chroma.from_texts(
            texts=sliced_text[:self.slice],  # 为了速度，只选择前 120 个切分的 doc 进行生成；使用千帆时因QPS限制，建议选择前 5 个doc
            embedding=self.embedding,
            persist_directory=self.persist_directory  # 允许我们将persist_directory目录保存到磁盘上
        )

    def persist(self):
        self.vectordb.persist()
        print(f"向量库中存储的数量：{self.vectordb._collection.count()}")

    def sim_search(self, query, k=3):
        # 最大余弦相似度检索
        sim_docs = self.vectordb.similarity_search(query, k=k)
        for i, sim_doc in enumerate(sim_docs, start=1):
            print(f"检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")
        return sim_docs

    def mmr_search(self, query, k=3):
        # 最大边际相关性搜索，获取与查询最相关的前 k 个文档
        mmr_docs = self.vectordb.max_marginal_relevance_search(query, k=k)
        for i, sim_doc in enumerate(mmr_docs, start=1):
            print(f"MMR 检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")
        return mmr_docs
