from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Load the api key 
load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Creating Schema 
schema = [
    ResponseSchema(name='fact_1', description='Fact1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact3 about the topic')
]

# Fixed lowercase variable name to match call later
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 facts about {topic}.\n{format_instructions}',
    input_variables=['topic'],
    # Fixed method call from get_format_instruction() to get_format_instructions()
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# Added .content step to extract text from ChatHuggingFace wrapper before parsing
chain = template | model | (lambda x: x.content) | parser

result = chain.invoke({'topic': 'black hole'})

print(result)
