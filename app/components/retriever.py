from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """Answer the following medical question in 2-3 lines maximum using only the information provided in the context.

Context:
{context}

Question:
{input}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE,input_variables=["context" , "question"])

def create_qa_chain():
    try:
        logger.info("Loading vector store for context")
        db = load_vector_store()
        if db is None:
            raise CustomException("Vector store not present or empty")

        logger.info("Loading LLM from Groq")
        llm = load_llm()
        if llm is None:
            raise CustomException("LLM not loaded")
        
        # 1. Setup Retriever
        retriever = db.as_retriever(search_kwargs={'k': 1}) # K=1 is very low; 3 is safer
        
        # 2. Setup Prompt
        prompt = ChatPromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)
        
        # 3. Create the "Stuff Documents" Chain (The LLM + Prompt part)
        # This replaces the old RetrievalQA logic
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        
        # 4. Create the Final Retrieval Chain (The Retriever + Document Chain)
        qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        logger.info("Successfully created the QA chain")
        return qa_chain
    
    except Exception as e:
        # Pass both message and the original exception to your CustomException
        error_message = f"Failed to make a QA chain | Error: {str(e)}"
        logger.error(error_message)
        raise CustomException(error_message)