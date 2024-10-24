import dspy
from dspy.retrieve.qdrant_rm import QdrantRM
from qdrant_client import QdrantClient
import os

from dotenv import load_dotenv
load_dotenv()

model = dspy.OpenAI(model='gpt-4o')

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_KEY"),
)

qdrant_retriever_model = QdrantRM("tweets", qdrant_client, k=7)
dspy.settings.configure(lm=model, rm=qdrant_retriever_model)


from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---

import json
import pandas as pd
import numpy as np
from qdrant_client.http import models
from qdrant_client.http.models import Batch

def load_paragraphs():
    tweets_path = os.path.join('./data-fronx-fronxer', "tweets_with_embeddings.csv")
    return pd.read_csv(tweets_path, converters={'embedding': json.loads})

def store_tweets(tweets_df, batch_size=50):
    num_batches = (len(tweets_df) + batch_size - 1) // batch_size
    batches = np.array_split(tweets_df, num_batches)

    for i, tweets in enumerate(batches, start=1):
        print(i, '/', num_batches)
        filtered_tweets = tweets[['id', 'full_text', 'account']]
        payloads = filtered_tweets.to_dict(orient='records')
        vectors = tweets['embedding'].to_list()

        qdrant_client.upsert(
            collection_name="tweets",
            points=Batch(
                ids=tweets['id'].to_list(),
                payloads=payloads,
                vectors=vectors,
            ),
        )

def create_collection():
    qdrant_client.create_collection('tweets', models.VectorParams(size=1536, distance=models.Distance.COSINE))

def get_relevant_tweets(query: str, top_k: int = 20):
    try:
        # 
        encoded_query = client.embeddings.create(input=[query],
        engine="text-embedding-ada-002").data[0]['embedding']

        result = qdrant_client.search(
            collection_name='tweets',
            query_vector=encoded_query,
            limit=top_k,
        )
        return result

    except Exception as e:
        print({e})

# retrieve = dspy.Retrieve(k=7)
# question = "What is consciousness"
# topK_passages = get_relevant_tweets(question)

class GenerateAnswer(dspy.Signature):
    context = dspy.InputField(desc="may contain relevant thoughts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="an answer that mirrors the style of the context")

class RAG(dspy.Module):
    def __init__(self, num_passages=20):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = [ point.payload['full_text'] for point in get_relevant_tweets(question) ]
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

from dspy.teleprompt import BootstrapFewShot

# Validation logic: check that the predicted answer is correct.
# Also check that the retrieved context does actually contain that answer.
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM

# Set up a basic teleprompter, which will compile our RAG program.
teleprompter = BootstrapFewShot(metric=validate_context_and_answer)

trainset = [
    dspy.Example(question="What is the essential character of flying things?", answer="They are nasty little buggers"),
    dspy.Example(question="?", answer="They are nasty little buggers"),
]

# Compile!
compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

