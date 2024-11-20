from langchain.text_splitter import RecursiveCharacterTextSplitter

# langchain递归切分

text = """百度百科是百度公司推出的一部内容开放、自由的网络百科全书。其测试版于2006年4月20日上线，正式版在2008年4月21日发布，截至2023年4月，百度百科已经收录了超2700万个词条，参与词条编辑的网友超过770万人，几乎涵盖了所有已知的知识领域。
“世界很复杂，百度更懂你”，百度百科旨在创造一个涵盖各领域知识的中文信息收集平台。百度百科强调用户的参与和奉献精神，充分调动互联网用户的力量，汇聚上亿用户的头脑智慧，积极进行交流和分享。同时，百度百科实现与百度搜索、百度知道的结合，从不同的层次上满足用户对信息的需求。
2024年4月，百度百科团队决定于2024年6月30日起，将百度百科APP的全部功能升级、迁移至百度APP内。使用百度APP中的【百度百科】小程序，体验更全面的产品功能，享受更高质量的浏览体验。"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=35,
    chunk_overlap=0,
    length_function=len
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"段 {i + 1}: {len(chunk)}:{chunk}")
