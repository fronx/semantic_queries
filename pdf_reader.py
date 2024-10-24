import sys

if len(sys.argv) < 2:
    print("Error: Please provide the path of a PDF file.")
    sys.exit(1)

file_path = sys.argv[1]

from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader(file_path)
documents = loader.load()

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
llm_transformer = LLMGraphTransformer(llm=llm)

import os

from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
load_dotenv()
graph = Neo4jGraph()

graph_documents = llm_transformer.convert_to_graph_documents(documents)
graph.add_graph_documents(graph_documents)
