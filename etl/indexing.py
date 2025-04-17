import os
import logging
from dotenv import load_dotenv, find_dotenv
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv(find_dotenv())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentIndexer:
    def __init__(self):
        self.connection_string = (
            f"postgresql+psycopg://"
            f"{os.getenv('POSTGRES_USER')}:"
            f"{os.getenv('POSTGRES_PASSWORD')}@"
            f"{os.getenv('POSTGRES_HOST')}:"
            f"{os.getenv('POSTGRES_PORT')}/"
            f"{os.getenv('POSTGRES_DB')}"
        )
        self.collection_name = os.getenv('COLLECTION_NAME')
        self.model_name = 'sentence-transformers/all-MiniLM-L6-v2'

    def get_documents_path(self):
        """Get the path to the documents directory."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "docs")

    def create_embeddings(self):
        """Create and return embedding pipeline."""
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"}
        )

    def load_documents(self, docs_path):
        """Load documents from the specified path."""
        loader = PyPDFDirectoryLoader(docs_path)
        return loader.load()

    def split_documents(self, documents):
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=256,
            chunk_overlap=50
        )
        return text_splitter.split_documents(documents)

    def create_vector_store(self):
        """Create and return vector store instance."""
        try:
            return PGVector(
                embeddings=self.create_embeddings(),
                collection_name=self.collection_name,
                connection=self.connection_string,
                use_jsonb=True,
            )
        except Exception as e:
            logger.error(f"An error occurred during vector store creation: {e}")
            return None

    def index_documents(self):
        """Main indexing process."""
        logger.info("Starting document indexing process...")
      
        docs_path = self.get_documents_path()
        logger.info(f"Using documents path: {docs_path}")
            
        documents = self.load_documents(docs_path)
        logger.info(f"Loaded {len(documents)} documents")
            
        chunks = self.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
            
        vector_store = self.create_vector_store()
        vector_store.add_documents(chunks)
        logger.info("Successfully added documents to vector store")
        
        return True
            

def main():
    indexer = DocumentIndexer()
    success = indexer.index_documents()
    
    if success:
        logger.info("Indexing completed successfully!")
    else:
        logger.error("Indexing failed!")

if __name__ == "__main__":
    main()