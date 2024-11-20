import base64
from io import BytesIO
# from IPython.display import HTML, display
from PIL import Image
from langchain_community.llms import Ollama


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


if __name__ == '__main__':
    file_path = "../documents/Dingtalk_20240814101941.jpg"
    pil_image = Image.open(file_path)
    image_b64 = convert_to_base64(pil_image)
    llm = Ollama(base_url="http://192.168.100.207:11434", model="qwen2:72b-instruct", keep_alive=0)
    llm_with_image_context = llm.bind(images=[image_b64])
    resp = llm_with_image_context.invoke("""提取原被告姓名、地址、小区名称、逾期费用、物业费，
            分别用plaintiff_name,plaintiff_address,defendant_name,defendant_address,cell_name,overdue_charge,property_fee表示，响应格式为json""")
    print(resp)
