from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

# RAG+Langchain 基于外部知识，增强大模型回复

loader = WebBaseLoader(
    "https://python.langchain.com/v0.2/docs/concepts/#prompt-templates")
docs = loader.load()

# 文档切分，trunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                               chunk_overlap=50)
all_splits = text_splitter.split_documents(docs)

print(f"split==> {all_splits[0]}")

ollama_emb = OllamaEmbeddings(model="qwen2:1.5b")

# 为文档片段生成向量
vector = Chroma.from_documents(documents=all_splits, embedding=ollama_emb, persist_directory="./db")
# save db to disk
vector.persist()
# 打印出第一个文档片段的向量，以验证向量生成是否成功
print(f"vector==> {vector}")

retriever = vector.as_retriever()

ollama = Ollama(base_url='http://localhost:11434', model="qwen2:1.5b")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question base only on the document
    <context>
    {context}
    </context>
    Question:{input}
    """
)

qa_chain = RetrievalQA.from_chain_type(ollama, retriever=retriever,
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": prompt})

result = qa_chain({"query": "langchain是什么？"})
print(f"result==> {result}")
