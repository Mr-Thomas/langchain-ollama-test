import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.embeddings import ZhipuAIEmbeddings

# find_dotenv() 寻找并定位 .env 文件的路径
# load_dotenv() 读取该 .env 文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

embed = ZhipuAIEmbeddings(api_key=os.environ.get("ZHIPUAI_API_KEY"))

# embed_documents
embeddings = embed.embed_documents(["你好", "你好啊", "你叫什么名字", "我叫王大锤", "很高兴认识你啊大锤"])
print(embeddings[:6])

# embed_query
embed_query = embed.embed_query("提到了什么名字")
print(embed_query[:5])

# 嵌入向量缓存
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

lfs = LocalFileStore("./embedding_cache")
cache_backed_embeddings = CacheBackedEmbeddings.from_bytes_store(embed, lfs, namespace=embed.model)
print(list(lfs.yield_keys()))

# 加载文档 并切分
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

docs = WebBaseLoader(
    "https://python.langchain.com/v0.2/docs/concepts/#prompt-templates").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(docs)
print(documents[:2])

# 切分后 存入缓存
from langchain_community.vectorstores import FAISS

faiss_store = FAISS.from_documents(documents[:2], cache_backed_embeddings)
print(list(lfs.yield_keys()))
