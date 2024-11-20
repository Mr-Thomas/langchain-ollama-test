from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
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


# 判断是纯图pdf和普通pdf
def is_image_pdf(pdf_path):
    # 打开PDF文件
    doc = fitz.open(pdf_path)
    # 检查每一页是否只包含图片
    for page in doc:
        # 尝试提取文本
        text = page.get_text("text")
        # 如果能提取到文本，则不是纯图片
        if text:
            return False
    # 如果没有提取到文本，则假定是纯图片
    return True


def generate_loaders_text(file_path: str) -> list:
    texts = []
    file_format = file_path.split('.')[-1]
    if file_format == 'pdf' or file_format == 'txt':
        with fitz.open(file_path) as pdf:
            # 遍历每一页
            for page in pdf:
                pp = page.get_images(full=True)
                # 提取文本
                text = page.get_text()
                # 添加到texts列表中
                texts.append(text)
    return texts


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


# 加载jpg文件
def load_jpg_file(jpg_file: str):
    work_dir = "../documents"
    ocr = RapidOCR()
    result, _ = ocr(os.path.join(work_dir, jpg_file))
    docs = ""
    if result:
        # 从OCR结果中提取文本信息，line[1]表示每行识别结果的文本部分
        ocr_result = [line[1] for line in result]
        docs += "\n".join(ocr_result)
    return docs


if __name__ == '__main__':
    # texts = generate_loaders_text("../documents/起诉状.pdf")
    # flag = is_image_pdf("../documents/答辩状.pdf")

    paths = pdf_to_images("../documents/起诉状.pdf", "../documents/image")
    text = load_jpg_file_list(paths)

    # text = load_jpg_file("../documents/Dingtalk_20240814101941.jpg")

    ollama = Ollama(base_url="http://192.168.100.207:11434", model="qwen2:72b-instruct", keep_alive=0)

    # template = """
    #     处理{input}内容：
    #         提取法条名称和条数，响应格式为["《law_name》law_number"]
    #     """

    template = """
        处理{input}内容：
            提取原被告姓名、地址、小区名称、逾期费用、物业费，
            分别用plaintiff_name,plaintiff_address,defendant_name,defendant_address,cell_name,overdue_charge,property_fee表示，响应格式为json
        """

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | ollama
    resp = chain.invoke(input=text)
    print(resp)

    # for text in texts:
    #     chain = prompt | ollama
    #     resp = chain.invoke(input=text)
    #     print(resp)
