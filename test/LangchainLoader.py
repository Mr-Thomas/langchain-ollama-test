from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = WebBaseLoader(
    "https://python.langchain.com/v0.2/docs/concepts/#prompt-templates")
docs = loader.load()

# 文档切分，trunks
text_splitter = RecursiveCharacterTextSplitter()
all_splits = text_splitter.split_documents(docs)

for i, all_split in enumerate(all_splits):
    print(f"块{i}:{all_split}")
