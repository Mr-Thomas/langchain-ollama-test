from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

ollama_emb = OllamaEmbeddings(model="qwen2:1.5b")
vectordb = Chroma(
    embedding_function=ollama_emb,
    persist_directory="./db"
)

print(f"向量库中存储的数量：{vectordb._collection.count()}")

docs = vectordb.similarity_search("langchain是什么?", k=3)
print(f"检索到的内容数：{len(docs)}")

for i, doc in enumerate(docs):
    print(f"检索到的第{i}个内容: \n {doc.page_content}", end="\n-----------------------------------------------------\n")
