from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.llms import Ollama

# RAG+Langchain 基于外部知识，增强大模型回复

loader = WebBaseLoader(
    "https://python.langchain.com/v0.2/docs/concepts/#prompt-templates")
docs = loader.load()

# 文档切分，trunks
text_splitter = RecursiveCharacterTextSplitter()
all_splits = text_splitter.split_documents(docs)

print(f"split==> {all_splits[0]}")

# 使用 OllamaEmbeddings 实例 ollama_emb 生成每个文档片段的向量。
# ollama_emb = OllamaEmbeddings(model="qwen2:1.5b")
embeddings = OllamaEmbeddings(base_url="http://192.168.100.207:11434", model="qwen2:72b-instruct")
print(f"ollama_emb==> {embeddings}")

# 使用 Chroma.from_documents 方法创建一个向量存储，并将这些向量存储在本地目录 ./db 中。
vector = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="./db")

# 调用 vector.persist() 将向量存储持久化到磁盘。
vector.persist()
# 打印出第一个文档片段的向量，以验证向量生成是否成功
print(f"vector==> {vector}")

# 使用 vector.as_retriever() 创建一个检索器。
retriever = vector.as_retriever()

# 定义一个 ChatPromptTemplate，用于在生成回答时提供上下文。
prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question base only on the document
    <context>
    {context}
    </context>
    Question:{input}
    """
)

# 使用 Ollama 实例创建一个大型语言模型接口。
# ollama = Ollama(base_url='http://localhost:11434', model="qwen2:1.5b")
ollama = Ollama(base_url="http://192.168.100.198:11434", model="qwen2:1.5b")

# 创建一个文档处理链 document_chain，用于处理文档和生成回答。
document_chain = create_stuff_documents_chain(ollama, prompt)

# Returns:
#     An LCEL Runnable. The Runnable return is a dictionary containing at the very
#     least a `context` and `answer` key.
# 创建一个检索链 retriever_chain，将检索器和文档处理链结合起来。
retriever_chain = create_retrieval_chain(retriever, document_chain)

resp = retriever_chain.invoke({"input": "what is langchain ?"})
print(f"""answer==> {resp["answer"]}""")
print(f"resp==> {resp}")
