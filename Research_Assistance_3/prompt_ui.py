from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, GenerationConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# -----------------------------
# Generation Configuration
# -----------------------------

generation_config = GenerationConfig(
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True
)


# -----------------------------
# Hugging Face Pipeline
# -----------------------------

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    generation_config=generation_config
)


# -----------------------------
# LangChain Wrapper
# -----------------------------

model = HuggingFacePipeline(
    pipeline=pipe
)


# -----------------------------
# Chatbot
# -----------------------------

print("AI Chatbot started!")
print("Type 'exit' or 'quit' to stop.\n")


while True:

    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("AI: Goodbye!")
        break

    prompt = f"""You are a helpful AI assistant.

User: {user_input}
Assistant:"""

    result = model.invoke(prompt)

    print("AI:", result)