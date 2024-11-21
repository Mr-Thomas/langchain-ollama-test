"""
RAG 增强流程架构
    1、数据处理与向量存储：
        文档被分片后生成嵌入向量，存储到向量数据库（Chroma）。
    2、用户查询：
        用户输入查询，生成查询嵌入向量。
    3、语义检索：
        使用查询嵌入从向量数据库中检索相关文档。
    4、上下文组合与生成回答：
        将检索到的文档上下文与用户问题组合，通过生成模型（ZhipuAI）生成回答。
"""

from zhipuChain import slice_text, slice_text_by_sentences
from langchain_community.vectorstores import Chroma
from rapidocr_onnxruntime import RapidOCR
import os
import fitz  # PyMuPDF
from zhipuai import ZhipuAI
from dotenv import load_dotenv, find_dotenv
from ZhipuLLM import ZhipuAILLM
from langchain_community.embeddings import ZhipuAIEmbeddings

_ = load_dotenv(find_dotenv())
client = ZhipuAI(
    api_key=os.environ.get("ZHIPUAI_API_KEY")
)  # 请填写您自己的APIKey


# PDF-to-Text
def extract_text_from_pdf(pdf_path) -> str:
    pdf_document = fitz.open(pdf_path)
    text = ""
    count = 0
    for page in pdf_document:
        page_text = page.get_text()
        if page_text.strip():  # 如果页面有可提取文本
            text += page_text
        else:  # 否则转图片进行 OCR
            text += ocr_page(page, count)
            count += 1
    pdf_document.close()
    return text


# OCR
def ocr_page(page, count) -> str:
    pix = page.get_pixmap()
    image_filename = f"../documents/image/temp_page_{count}.jpg"
    pix.save(image_filename)
    ocr = RapidOCR()
    result, _ = ocr(image_filename)
    if result:
        return "\n".join(line[1] for line in result)
    return ""


def build_context_from_results(search_results, score_threshold=0.5) -> str:
    """
    根据得分筛选并合并上下文
    """
    filtered_results = [res for res in search_results if res[1] >= score_threshold]
    unique_contexts = list({result[0].metadata["content"] for result in filtered_results})
    return "\n".join(unique_contexts)


embedding = ZhipuAIEmbeddings(model="embedding-3", api_key=os.environ.get("ZHIPUAI_API_KEY"))

# 初始化 Chroma 向量数据库
persist_directory = "../test/db"
chroma = Chroma(persist_directory=persist_directory, embedding_function=embedding)


# 文档插入时调用嵌入对象的方法
def insert_documents(sliced_text: list):
    chroma.add_texts(
        texts=sliced_text,
        metadatas=[{"content": doc} for doc in sliced_text],
        embeddings=embedding.embed_documents(sliced_text),
    )
    chroma.persist()


text = extract_text_from_pdf("../documents/起诉状.pdf")
texts = [text]
sliced_text = slice_text(texts)
# sliced_text = slice_text_by_sentences(text)

insert_documents(sliced_text)

# 实例化 ZhipuAILLM 类
zhipuai_llm = ZhipuAILLM(api_key=os.environ.get("ZHIPUAI_API_KEY"))

query = "请帮我分析起诉状，列出所有原、被告等所有当事人"

# 查询相关文档
query_vector = embedding.embed_query(query)
# 输入： 查询嵌入向量（query_vector），需要先通过嵌入函数手动生成。
# 输出： 一个元组列表，每个元组包含： Document 对象：检索到的文档。 相似度分数：与查询嵌入的相似度得分。
search_results = chroma.similarity_search_with_score(query_vector, k=5)

# 提取检索到的文档内容
context = build_context_from_results(search_results)
# context = "\n".join([result[0].metadata["content"] for result in search_results])

# 构建生成模型的输入
prompt = f"根据以下内容回答问题：\n{context}\n\n问题：{query}"

# 调用生成模型
answer = zhipuai_llm._generate(prompt)
print(answer)





# 方式二
# 构造 LangChain 的检索器
retriever = chroma.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# 构造 RAG 生成链
class RAGChain:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def query(self, query_text):
        # 1. 检索相关文档【自动完成查询文本到嵌入向量的转换，并进行语义检索】
        # 输入： 查询文本（query_text），不需要手动计算嵌入。
        # 输出： 一个 Document 对象列表，每个对象包含以下字段：page_content：文档内容。metadata：文档的元数据信息（如标题、来源等）。
        docs = self.retriever.get_relevant_documents(query_text)
        # 2. 构造上下文
        context = "\n".join([doc.page_content for doc in docs])
        # 3. 构造提示语
        prompt = f"根据以下内容回答问题：\n{context}\n\n问题：{query_text}"
        # 4. 调用 LLM 生成答案
        return self.llm._generate(prompt)


# 初始化 RAGChain
rag_chain = RAGChain(retriever=retriever, llm=zhipuai_llm)
# 获取回答
answer = rag_chain.query(query_text=query)
# 输出结果
print("生成的回答：", answer)
