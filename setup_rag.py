from dotenv import load_dotenv
import time
import os
import json
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import openai


load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create a Pinecone index
pc.create_index(
    name="rag",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

# Load the review data
data = json.load(open("reviews.json"))

processed_data = []
client = OpenAI()

def get_embedding(text):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=text, model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error: {e}")
            if "RateLimitError" in str(e):
                wait_time = 2 ** attempt
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise

# Create embeddings for each review
for review in data["reviews"]:
    embedding = get_embedding(review['review'])
    processed_data.append(
        {
            "values": embedding,
            "id": review["professor"],
            "metadata": {
                "review": review["review"],
                "subject": review["subject"],
                "stars": review["stars"],
            }
        }
    )

# Insert the embeddings into the Pinecone index
index = pc.Index("rag")
upsert_response = index.upsert(
    vectors=processed_data,
    namespace="ns1",
)
print(f"Upserted count: {upsert_response['upserted_count']}")

# Print index statistics
print(index.describe_index_stats())
