import fitz
from langchain_community.embeddings import ZhipuAIEmbeddings
import os

from pymupdf import Document
from zhipuai import ZhipuAI
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredMarkdownLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import re
from vectordb import VectorDB

# find_dotenv() 寻找并定位 .env 文件的路径
# load_dotenv() 读取该 .env 文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

client = ZhipuAI(
    api_key=os.environ.get("ZHIPUAI_API_KEY")
)  # 请填写您自己的APIKey


# 获取folder_path下所有文件路径，储存在file_paths里
def generate_path(folder_path: str = '../documents/') -> list:
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


def generate_loaders_text(file_paths: list) -> list:
    texts = []
    for file_path in file_paths:
        file_type = file_path.split('.')[-1]
        if file_type == 'pdf':
            with fitz.open(file_path) as pdf:
                # 遍历每一页
                for page in pdf:
                    # 提取文本
                    text = page.get_text()
                    text = text.replace('\n', '')
                    text = text.replace('\u3000', '')
                    # 添加到texts列表中
                    texts.append(text)
    return texts


# PyMuPDFLoader、UnstructuredMarkdownLoader加载文件
def generate_loaders(file_paths: list) -> list:
    loaders = []
    for file_path in file_paths:
        file_type = file_path.split('.')[-1]
        if file_type == 'pdf':
            loaders.append(PyMuPDFLoader(file_path))
        elif file_type == 'md':
            loaders.append(UnstructuredMarkdownLoader(file_path))
    return loaders


def exec_load(loaders: list) -> list:
    texts = []
    for loader in loaders:
        content = loader.load()
        texts.extend(content)
    return texts


def slice_text(texts: list):
    """
    按照 句号（。）、问号（？）、感叹号（！）和省略号（……） 对文本 text 进行分割。
    """
    text_list = []
    for text in texts:
        sentences = re.split(r"(。|？|！|\...\...)", text)
        chunks = [sentence + (punctuation if punctuation else '') for sentence, punctuation in
                  zip(sentences[::2], sentences[1::2])]
        text_list.extend(chunks)
    return text_list


def slice_text_by_sentences(text: str, max_length: int = 500) -> list:
    """
    按句子分片，确保每片不超过 max_length 字符
    """
    sentences = text.split("。")
    slices, current_slice = [], ""
    for sentence in sentences:
        if len(current_slice) + len(sentence) <= max_length:
            current_slice += sentence + "。"
        else:
            slices.append(current_slice.strip())
            current_slice = sentence + "。"
    if current_slice:
        slices.append(current_slice.strip())
    return slices


def slice_docs(texts):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return text_splitter.split_documents(texts)


if __name__ == '__main__':
    file_paths = generate_path()

    # loaders = generate_loaders(file_paths)

    # texts = exec_load(loaders=loaders)

    # sliced_docs = slice_docs(texts)

    texts = generate_loaders_text(file_paths)

    sliced_text = slice_text(texts)

    vdb = VectorDB(sliced_text)

    vdb.persist()

    vdb.mmr_search("中华人民共和国民事诉讼法第九条具体内容是什么")

# embedding = ZhipuAIEmbeddings()

# vector = Chroma.from_texts(texts=sliced_text[:200], embedding=embedding, persist_directory="../test/db")

# vector.persist()

# simi_docs = vector.similarity_search("中华人民共和国民事诉讼法第九条具体内容是什么", k=3)
# for i, simi_doc in enumerate(simi_docs):
#     print(f"""{i}==> {simi_doc}""")


# prompt = ChatPromptTemplate.from_template(
#     """
#     从文档中总结回答以下问题
#     <context>
#     {context}
#     </context>
#     问题:{input}
#     """
# )
#
# document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
#
# retriever = vector.as_retriever()
#
# retriever_chain = create_retrieval_chain(retriever, document_chain)
#
# resp = retriever_chain.invoke({"input": "中华人民共和国民法典第十一的具体内容是什么 ?"})
# print(f"""answer==> {resp["answer"]}""")
# print(f"resp==> {resp}")
