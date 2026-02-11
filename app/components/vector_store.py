import os

from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DB_CHROMA_PATH

from langchain_chroma import Chroma

logger = get_logger(__name__)


def _is_non_empty_dir(path: str) -> bool:
    return os.path.exists(path) and os.path.isdir(path) and len(os.listdir(path)) > 0


def load_vector_store():
    try:
        embedding_model = get_embedding_model()

        if _is_non_empty_dir(DB_CHROMA_PATH):
            logger.info("Loading existing Chroma vectorstore...")
            return Chroma(
                persist_directory=DB_CHROMA_PATH,
                embedding_function=embedding_model,
                collection_name="ragbot",
            )

        logger.warning("No vectorstore found. Persist directory is empty or missing.")
        return None

    except Exception as e:
        error_message = CustomException("Failed to load vectorstore", e)
        logger.error(str(error_message))
        raise


def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No chunks were found..")

        logger.info("Generating your new Chroma vectorstore...")

        embedding_model = get_embedding_model()

        db = Chroma.from_documents(
            documents=text_chunks,
            embedding=embedding_model,
            persist_directory=DB_CHROMA_PATH,
            collection_name="ragbot",
        )

        logger.info(f"Vectorstore saved successfully at: {DB_CHROMA_PATH}")
        return db

    except Exception as e:
        error_message = CustomException("Failed to create new vectorstore", e)
        logger.error(str(error_message))
        raise
