"""rabbithole.vecstore module"""

import os

import pinecone

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west4-gcp")
index = pinecone.Index("rabbithole")
