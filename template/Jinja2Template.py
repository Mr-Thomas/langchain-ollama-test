from langchain.prompts import PromptTemplate

# f-string 提示词模板
f_string = """
    给我讲一个关于{name}的{what}故事！
"""
prompt = PromptTemplate.from_template(f_string, template_format="f-string")
print(prompt.format(name="狗剩", what="猫科"))

print(" =================================================================================== ")

# jinja2灵活、高效的python模板引擎
jinja_template = "给我讲一个关于{{name}}的{{what}}故事！"
prompt = PromptTemplate.from_template(jinja_template, template_format="jinja2")
print(prompt.format(name="狗剩", what="猫科"))

print(" =================================================================================== ")

# 组合提示词模板
# Pipeline Prompt组成提示词管道模板
from langchain.prompts.pipeline import PipelinePromptTemplate

full_template = """{Character}
{behavior}
{prohibit}
"""
full_prompt = PromptTemplate.from_template(full_template)

# 第一层
Character_template = """你是{person}，你有着{xingge}性格;"""
Character_prompt = PromptTemplate.from_template(Character_template)

# 第二层
behavior_template = """你遵从以下行为：
{behavior_list}
"""
behavior_prompt = PromptTemplate.from_template(behavior_template)

# 第三层
prohibit_template = """你不允许有以下行为： 
{prohibit_list}
"""
prohibit_prompt = PromptTemplate.from_template(prohibit_template)

# 将三层组合起来
input_prompts = [
    ("Character", Character_prompt),
    ("behavior", behavior_prompt),
    ("prohibit", prohibit_prompt)
]
pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt,
                                         pipeline_prompts=input_prompts)
pm = pipeline_prompt.format(person="猫科", xingge="逗逼", behavior_list="睡猫窝", prohibit_list="吃猫砂")
print(pm)
