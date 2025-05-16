import base64
from io import BytesIO
# from IPython.display import HTML, display
from PIL import Image
from langchain_ollama import OllamaLLM
from rapidocr_onnxruntime import RapidOCR
import os


def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# def plt_img_base64(img_base64):
#     """
#     Display base64 encoded string as image
#
#     :param img_base64:  Base64 string
#     """
#     # Create an HTML img tag with the base64 string as the source
#     image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
#     # Display the image by rendering the HTML
#     display(HTML(image_html))

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
    # file_path = "../documents/Dingtalk_20240814101941.jpg"
    # pil_image = Image.open(file_path)
    # image_b64 = convert_to_base64(pil_image)
    # # deepseek-r1 不是多模态 不支持图文
    # llm = OllamaLLM(base_url="127.0.0.1:11434", model="deepseek-r1:1.5b", keep_alive=0)
    # llm_with_image_context = llm.bind(images=[image_b64])
    # resp = llm_with_image_context.invoke("""提取原被告姓名、地址、小区名称、逾期费用、物业费，
    #         分别用plaintiff_name,plaintiff_address,defendant_name,defendant_address,cell_name,overdue_charge,property_fee表示，响应格式为json""")
    # print(resp)

    file_path = "../documents/Dingtalk_20240814101941.jpg"
    text = load_jpg_file(file_path)
    llm = OllamaLLM(base_url="192.168.11.198:11434", model="deepseek-r1:14b", keep_alive=0)
    response = llm.invoke(f"""
    根据以下内容提取关键信息:
    {text}
    提取字段为原被告姓名、地址、小区名称、逾期费用、物业费，分别用plaintiff_name,plaintiff_address,defendant_name,defendant_address,cell_name,overdue_charge,property_fee表示，响应格式为json
    """)
    print(response)
