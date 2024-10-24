from dotenv import load_dotenv
load_dotenv()

from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI


from langchain_community.graphs import Neo4jGraph
graph = Neo4jGraph()

graph.refresh_schema()

chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(temperature=0, model_name="gpt-4o"),
    qa_llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
)

import readline
import atexit
import os

HISTORY_FILE = ".command_history"

# Load history if it exists
if os.path.exists(HISTORY_FILE):
    readline.read_history_file(HISTORY_FILE)

# Register a function to save history on exit
atexit.register(readline.write_history_file, HISTORY_FILE)

while True:
    try:
        user_input = input("> ")
        readline.add_history(user_input)
        response = chain.invoke(user_input)
        print(response['result'])
    except EOFError:
        print("\nExiting...")
        break
