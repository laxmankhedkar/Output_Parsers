from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'Qwen/Qwen2.5-7B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm = llm)

template1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables=['topic']
)

# Prompt 2: Summarize the generated report
template2 = PromptTemplate(
    template='Write a 1-line summary of the following text:\n{text}',
    input_variables=['text']
)

# Creating Str output parser

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser 

result = chain.invoke({'topic':'black hole'})

# Print the final summary
print(result)