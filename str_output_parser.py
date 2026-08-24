from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env
load_dotenv()

# Connect to the Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

# Create a chat model
model = ChatHuggingFace(llm=llm)

# Prompt 1: Generate a detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# Prompt 2: Summarize the generated report
template2 = PromptTemplate(
    template='Write a 1-line summary of the following text:\n{text}',
    input_variables=['text']
)

# Fill the first prompt
prompt1 = template1.invoke({
    'topic': 'black hole'
})

# Send the first prompt to the LLM
result = model.invoke(prompt1)

# Put the first response into the second prompt
prompt2 = template2.invoke({
    'text': result.content
})

# Send the second prompt to the LLM
result1 = model.invoke(prompt2)

# Print the final summary
print(result1.content)