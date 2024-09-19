
# %%
import langchain
from langchain.llms import OpenAI
# %%
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

# %%
import os   
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# %%
prompt = PromptTemplate(        
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
# %%
llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.0})
# %%
llm_chain = LLMChain(prompt=prompt, llm=llm)
# %%
llm_chain.run("healty spud")
# %%
from langchain.llms import HuggingFacePipeline
import torch        
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
#%%
prompt = PromptTemplate(
    input_variables=["name"],
    template="can you tell me about football player {name}?",
)
# %%
model_id = 'google/flan-t5-large'
# %%
tokenizer = AutoTokenizer.from_pretrained(model_id)
# %%
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map = 'auto')
# %%
pipeline = pipeline('text2text-generation', model=model, tokenizer=tokenizer, max_length=128)

# %%
llm = HuggingFacePipeline(pipeline=pipeline)
# %%
llm_chain = LLMChain(prompt=prompt, llm=llm)
# %%
llm_chain.run("messi")
# %%
