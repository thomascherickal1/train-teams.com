import os
from langchain import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.deeplake import DeepLake
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.chat_models.openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory

from .utils import save

import config as cfg

class Retriever:
    def __init__(self):
        self.text_retriever = None
        self.text_deeplake_schema = None
        self.embeddings = None
        self.memory = ConversationBufferWindowMemory(k=20, return_messages=True).csv


def create_and_add_embeddings(self, file):
    # Create a directory named "data" if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Initialize embeddings using OpenAIEmbeddings
    self.embeddings = OpenAIEmbeddings(
        openai_api_key=cfg.OPENAI_API_KEY,
        chunk_size=cfg.OPENAI_EMBEDDINGS_CHUNK_SIZE,
    )

    # Load documents from the provided file using PyMuPDFLoader
    loader = PyMuPDFLoader(file)
    documents = loader.load()

    # Split text into chunks using CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        chunk_size=cfg.CHARACTER_SPLITTER_CHUNK_SIZE,
        chunk_overlap=0,
    )
    docs = text_splitter.split_documents(documents)

    # Create a DeepLake schema for text documents
    self.text_deeplake_schema = DeepLake(
        dataset_path=cfg.TEXT_VECTORSTORE_PATH,
        embedding_function=self.embeddings,
        overwrite=True,
    )

    # Add the split documents to the DeepLake schema
    self.text_deeplake_schema.add_documents(docs)

    # Create a text retriever from the DeepLake schema with search type "similarity"
    self.text_retriever = self.text_deeplake_schema.as_retriever(
        search_type="similarity"
    )

    # Configure search parameters for the text retriever
    self.text_retriever.search_kwargs["distance_metric"] = "cos"
    self.text_retriever.search_kwargs["fetch_k"] = 15
    self.text_retriever.search_kwargs["maximal_marginal_relevance"] = True
    self.text_retriever.search_kwargs["k"] = 3

def retrieve_text(self, query):
    # Create a DeepLake schema for text documents in read-only mode
    self.text_deeplake_schema = DeepLake(
        dataset_path=cfg.TEXT_VECTORSTORE_PATH,
        read_only=True,
        embedding_function=self.embeddings,
    )

    # Define a prompt template for giving instruction to the model
    prompt_template = """You are an advanced AI capable of analyzing text from 
    documents and providing detailed answers to user queries. Your goal is to 
    offer comprehensive responses to eliminate the need for users to revisit 
    the document. If you lack the answer, please acknowledge it rather than 
    making up information.
    {context}
    Question: {question} 
    Answer:
    """

    # Create a PromptTemplate with the "context" and "question"
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Define chain type
    chain_type_kwargs = {"prompt": PROMPT}

    # Initialize the ChatOpenAI model
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=cfg.OPENAI_API_KEY,
    )

    # Create a RetrievalQA instance of the model
    qa = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=self.text_retriever,
        return_source_documents=False,
        verbose=False,
        chain_type_kwargs=chain_type_kwargs,
        memory=self.memory,
    )

    # Query the model with the user's question
    response = qa({"query": query})

    # Return response from llm
    return response["result"]



def save(query, qa):
    # Use the get_openai_callback function 
    with get_openai_callback() as cb:
        # Query the qa object with the user's question
        response = qa({"query": query}, return_only_outputs=True)
        
        # Return the answer from the llm's response
        return response["result"]