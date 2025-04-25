import os
from dotenv import load_dotenv, find_dotenv
from langchain_postgres import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_unstructured import UnstructuredLoader

from pipeline.pipeline import process_all_pdfs
from pipeline.constant import INPUT_DIR, OUTPUT_DIR
from pipeline.utils import setup_logging, save_combined_output

load_dotenv(find_dotenv())

logger = setup_logging()

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
    

    def create_embeddings(self):
        """Create and return embedding pipeline."""
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"}
        )

    def load_documents(self, docs_path):
        """Load documents from the specified path."""
        loader = UnstructuredLoader(docs_path)
        return loader.load()

    def split_documents(self, documents):
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500,
            chunk_overlap=100,
            separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                "",
            ],
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
        try:
            logger.info("Processing PDF documents...")
            texts, markdowns = process_all_pdfs(INPUT_DIR)
            logger.info("PDF documents processed successfully")

            save_combined_output(texts, markdowns)
            logger.info(f"Combined output saved successfully on {OUTPUT_DIR}")
        except Exception as e:
            logger.error(f"An error occurred during processing: {e}")
            
        
        documents = self.load_documents(OUTPUT_DIR + "/combined_output.txt")
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