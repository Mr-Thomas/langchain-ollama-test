import os
from zhipuai import ZhipuAI
from dotenv import load_dotenv, find_dotenv

# find_dotenv() 寻找并定位 .env 文件的路径
# load_dotenv() 读取该 .env 文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

client = ZhipuAI(
    api_key=os.environ.get("ZHIPUAI_API_KEY")
)  # 请填写您自己的APIKey


def zhipu_embedding(text: str):
    response = client.embeddings.create(
        model="Embedding-3",
        input=text,
    )
    return response


if __name__ == '__main__':
    # response为`zhipuai.types.embeddings.EmbeddingsResponded`类型，
    # 我们可以调用`object`、`data`、`model`、`usage`来查看response的embedding类型、embedding、embedding model及使用情况。
    text = '要生成 embedding 的输入文本，字符串形式。'
    response = zhipu_embedding(text=text)
    print(response)
