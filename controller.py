from retriever.retrieval import Retriever

# Create a Controller class to manage document embedding and retrieval
class Controller:
    def __init__(self):
        self.retriever = None
        self.query = ""

    def embed_document(self, file):
        # Embed a document if 'file' is provided
        if file is not None:
            self.retriever = Retriever()
            # Create and add embeddings for the provided document file
            self.retriever.create_and_add_embeddings(file.name)

    def retrieve(self, query):
        # Retrieve text based on the user's query
        texts = self.retriever.retrieve_text(query)
        return texts  