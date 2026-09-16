from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()


# Create Hugging Face pipeline

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True
)


# Wrap Hugging Face pipeline with LangChain

model = HuggingFacePipeline(
    pipeline=pipe
)


# Create prompt

prompt = """<|im_start|>system
You are a helpful AI assistant.
<|im_end|>

<|im_start|>user
Tell me about LangChain.
<|im_end|>

<|im_start|>assistant
"""


# Invoke model

result = model.invoke(prompt)

print(result)