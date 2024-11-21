"""
向量化以及检索
"""

from zhipuChain import slice_text
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
from rapidocr_onnxruntime import RapidOCR
import os
import fitz  # PyMuPDF


# pdf转图片
def pdf_to_images(pdf_path, output_folder) -> list:
    output_file_path = []
    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)
    # 遍历PDF中的每一页
    for page_number in range(len(pdf_document)):
        # 获取PDF页面
        page = pdf_document[page_number]

        # 将PDF页面渲染为图片
        # matrix参数设置为fitz.Identity，表示不进行变换
        # pix默认为RGB模式
        pix = page.get_pixmap()
        # 构建输出图片的文件名
        image_filename = f"{output_folder}/page_{page_number + 1}.jpg"
        # 保存图片
        pix.save(image_filename)
        output_file_path.append(image_filename)
    # 关闭PDF文件
    pdf_document.close()
    return output_file_path


def load_jpg_file_list(jpg_files: list):
    work_dir = "../documents"
    ocr = RapidOCR()
    docs = ""
    for jpg_file in jpg_files:
        result, _ = ocr(os.path.join(work_dir, jpg_file))
        if result:
            # 从OCR结果中提取文本信息，line[1]表示每行识别结果的文本部分
            ocr_result = [line[1] for line in result]
            docs += "\n".join(ocr_result)
        docs += "\n"
    return docs


paths = pdf_to_images("../documents/起诉状.pdf", "../documents/image")
text = load_jpg_file_list(paths)
texts = [text]

sliced_text = slice_text(texts)

embedding = ZhipuAIEmbeddings()

vector = Chroma.from_texts(texts=sliced_text, embedding=embedding, persist_directory="../test/db")

# 向量库中的数据持久化
vector.persist()

simi_docs = vector.similarity_search("请帮我分析起诉状，列出所有原、被告等所有当事人", k=5)
print(simi_docs)

retriever = vector.as_retriever()

result = retriever.invoke("请帮我分析起诉状，列出所有原、被告等所有当事人")

print(result)
