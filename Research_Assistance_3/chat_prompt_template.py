from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Tell me about {domain} and the topic {topic}.")
])

prompt = chat_template.format_messages(
    domain="cricket",
    topic="Dusra"
)

print(prompt)