from zhipuai import ZhipuAI
import os
from dotenv import load_dotenv, find_dotenv

# find_dotenv() 寻找并定位 .env 文件的路径
# load_dotenv() 读取该 .env 文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

client = ZhipuAI(
    api_key=os.environ.get("ZHIPUAI_API_KEY")
)  # 请填写您自己的APIKey


# # 通过自定义 prompt 来提升模型回答效果
def completion_api(prompt: str):
    response = client.chat.completions.create(
        # 获取 GPT 模型调用结果
        # 请求参数：
        #     prompt: 对应的提示词
        #     model: 调用的模型，默认为 gpt-3.5-turbo，也可以按需选择 gpt-4 等其他模型
        #     temperature: 模型输出的温度系数，控制输出的随机程度，取值范围是 0~2。温度系数越低，输出内容越一致。
        model="GLM-4-0520",  # 填写需要调用的模型编码
        messages=[
            {"role": "system", "content": "你是一个法律服务顾问，你的任务是为用户提供专业、准确的建议。"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )
    if len(response.choices) > 0:
        return response.choices[0].message.content
    return "generate answer error"


if __name__ == '__main__':
    response = completion_api(prompt="你好！你叫什么名字？")
    print(response)

    text = f"""答辩状
        尊敬的审判长：
        我是本案被告王庆彬，于[收到起诉状的日期]收到原告山东万泰昇杰现代服务业有限公
        司提起的民事起诉状，经过认真阅读和分析，现依法提出答辩意见如下：
        一、关于物业费的支付问题
        原告要求我支付物业费共计人民币 7642.99 元。对此，我有以下答辩意见：
        1. 根据《中华人民共和国物权法》第七十一条规定，业主应当按照约定交纳物业服务
        费。然而，我并未实际享受原告所提供的物业管理服务，因此不应承担相应费用。
        2. 原告未能提供证据证明其提供的服务符合《中华人民共和国合同法》第三百九十二
        条所规定的质量要求，且未向我明示服务内容及收费标准，因此我有理由拒绝支付物业费。
        3. 我从未收到过任何正式的缴费通知或者催缴物业费的相关文件，因此对于所谓的拖
        欠物业费用一事并不知情。
        二、关于逾期付款损失的问题
        原告请求我支付逾期付款损失暂计人民币 512.86 元。对此，我认为：
        1. 原告所主张的损失计算方式未向我明确告知，且该计算方式是否合理及合法有待商
        榷。
        2. 根据《中华人民共和国合同法》第一百一十四条规定，逾期付款的违约金应当合理
        确定。若按照原告所述以全国银行间同业拆借中心公布的贷款市场报价利率的 1.5 倍计算逾
        期付款损失，可能会高于法定标准或实际造成的损失。
        3. 依据《中华人民共和国民事诉讼法》第六十四条规定，当事人对自己提出的主张有
        责任提供证据。原告应提供充分证据证明其损失的实际发生及合理数额。
        三、关于诉讼费用的承担问题
        原告请求我承担本案的诉讼费用。鉴于我上述答辩理由成立，请求法院依法判决原告自
        行承担诉讼费用。
        综上所述，请求贵院依法审理，并判决驳回原告的全部诉讼请求。
        此致
        敬礼！
        答辩人：王庆"""

    prompt = f"""
    按步骤处理以下"{text}"里的文本内容：
        1、提取出相关法条
        2、对步骤1提取的内容首尾添加<span></span>标签
        3、处理后的"{text}"输出为标准HTML格式
    """
    response = completion_api(prompt=prompt)
    print(response)
