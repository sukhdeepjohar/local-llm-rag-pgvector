from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core import StorageContext, Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
# from llama_index.readers.web import SimpleWebPageReader 
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_llm(model_path):
    """Initialize the Llama model with adjusted context window settings."""
    try:
        return LlamaCpp(
            model_path=model_path,
            context_window=4096,
            n_batch=128,
            n_ctx=4096,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=True,
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

def create_qa_system(documents_path, vector_store, llm):
    """Create the question-answering system with updated Settings configuration."""
    try:
        # Configure settings
        Settings.llm = llm
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="intfloat/multilingual-e5-small",
        )
        # You can use the below settings to reduce your chunk size to accomodate in shorted context windows
        # Settings.chunk_size = 256
        # Settings.chunk_overlap = 20
        
        # Setup storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Load documents
        documents = SimpleDirectoryReader(documents_path).load_data()
        logger.info(f"Loaded {len(documents)} documents")
        
        # Create index
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context
        )
        
        # Configure and return query engine
        # return index.as_query_engine()	
        return index.as_query_engine( similarity_top_k=25)
        # restrictive response if you have a very short context window
        # return index.as_query_engine( similarity_top_k=2, response_mode="compact")
    except Exception as e:
        logger.error(f"Failed to create QA system: {e}")
        raise

def setup_vector_store(db_config):
    """Setup PostgreSQL vector store connection."""
    try:
        return PGVectorStore.from_params(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database'],
            embed_dim=384,
        )
    except Exception as e:
        logger.error(f"Failed to connect to vector store: {e}")
        raise

def interactive_qa_session(query_engine):
    """Run an interactive Q&A session."""
    print("\nWelcome to the Interactive Q&A System!")
    print("Type 'exit', 'quit', or press Ctrl+C to end the session")
    print("-----------------------------------------------")
    
    try:
        while True:
            # Get query from user
            query = input("\nEnter your question: ").strip()
            
            # Check for exit commands
            if query.lower() in ['exit', 'quit', '']:
                print("\nEnding Q&A session. Goodbye!")
                break
                
            try:
                # Process query and get response
                print("\nProcessing your question...")
                response = query_engine.query(query)
                
                # Print response
                print("\nResponse:", response)
                print("\n-----------------------------------------------")
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"\nSorry, there was an error processing your question: {str(e)}")
                print("Please try another question.")
                
    except KeyboardInterrupt:
        print("\n\nSession terminated by user. Goodbye!")
    except Exception as e:
        logger.error(f"Session error: {e}")
        print(f"\nAn error occurred: {str(e)}")

def main():
    # Configuration
    # model_path = './models/llama-2-7b-chat.Q4_K_M.gguf'
    model_path = './models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf'
    # model_path = './models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf'
    db_config = {
        'host': 'localhost',
        'port': '5432',
        'user': 'postgres',
        'password': 'password',
        'database': 'vector_db'
    }
    documents_path = './data'
    
    try:
        # Initialize components
        print("Initializing system...")
        llm = initialize_llm(model_path)
        vector_store = setup_vector_store(db_config)
        query_engine = create_qa_system(documents_path, vector_store, llm)
        
        # Start interactive session
        interactive_qa_session(query_engine)
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
