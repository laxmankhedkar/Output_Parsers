# importing required libraries
from langchain_huggingface import HuggingFacePipeline 
from transformers import pipeline
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


# Create a Hugging Face pipeline for text generation
pipe = pipeline(
    "text-generation",
    model = "Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens = 100,
    temperature = 0.7
)


# Wrap the Hugging Face pipeline in a LangChain HuggingFacePipeline
model = HuggingFacePipeline(pipeline=pipe)

chat_history = []

# Create chatbot loop 
while True:
    user_input =  input('You:  ')
    chat_history.append(f"You: {user_input}")

    if user_input.lower() in ['exit', 'quit']:
        break

    result = model.invoke(user_input)
    print(f"AI: {result}")


