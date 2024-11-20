"""
range(3) 就是 0 ，1，2 ，每次递增 1
range(3,6) 就是 3 ，4 ，5 ，也是每次递增 1
range(0,10,2) , 它的意思是：从 0 数到 10（不取 10 ），每次间隔为 2

切片索引从0开始
sentences[1:4] 将返回一个包含第2、3、4个元素的列表
sentences[1::4] 将返回一个包含第2、6、10个元素的列表。
"""
sentences = [i for i, e in enumerate(range(11))]
print(sentences)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(sentences[1:4])  # [1, 2, 3]
print(sentences[1::4])  # [1, 5, 9]
print(sentences[1::2])  # [1, 3, 5, 7, 9]
print(sentences[::4])  # [0, 4, 8]
print(sentences[1:])  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

sentences = zip(sentences[::2], sentences[1::2])
sentences = [sentence for i, sentence in enumerate(sentences)]
print(sentences)  # [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

# 按照句子来切分
import re

text = """百度百科是百度公司推出的一部内容开放、自由的网络百科全书。其测试版于2006年4月20日上线，正式版在2008年4月21日发布，截至2023年4月，百度百科已经收录了超2700万个词条，参与词条编辑的网友超过770万人，几乎涵盖了所有已知的知识领域。
“世界很复杂，百度更懂你”，百度百科旨在创造一个涵盖各领域知识的中文信息收集平台。百度百科强调用户的参与和奉献精神，充分调动互联网用户的力量，汇聚上亿用户的头脑智慧，积极进行交流和分享。同时，百度百科实现与百度搜索、百度知道的结合，从不同的层次上满足用户对信息的需求。
2024年4月，百度百科团队决定于2024年6月30日起，将百度百科APP的全部功能升级、迁移至百度APP内。使用百度APP中的【百度百科】小程序，体验更全面的产品功能，享受更高质量的浏览体验。"""

sentences = re.split(r"(。|？|！|\...\...)", text)
print(f"sentences==>{len(sentences)}:{sentences}")

sentenceList = [sentence for sentence, punctuation in enumerate(zip(sentences[::2], sentences[1::2]))]
print(f"sentenceList==>{len(sentenceList)}:{sentenceList}")

punctuationList = [punctuation for sentence, punctuation in enumerate(zip(sentences[::2], sentences[1::2]))]
print(f"punctuationList==>{len(punctuationList)}:{punctuationList}")

chunks = [sentence + (punctuation if punctuation else '') for sentence, punctuation in
          zip(sentences[::2], sentences[1::2])]

print(f"chunks==>{len(chunks)}:{chunks}")

for i, chunk in enumerate(chunks):
    print(f"块 {i + 1}: {len(chunk)}:{chunk}")
