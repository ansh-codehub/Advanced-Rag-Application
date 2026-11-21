from functools import lru_cache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import config

@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    """
    Create (or reuse) the OpenAI chat model used for every LLM call.

    The chat model does two jobs in this project: writing alternative versions
    of the user's question, and writing the final answer from the retrieved
    context.

    Returns:
        A configured LangChain ChatOpenAI model.

    Raises:
        ValueError: If GROQ_API_KEY is missing.
    """    

    config.check_openai_key()

    return ChatOpenAI(
        model = config.GROQ_CHAT_MODEL,
        temperature = config.LLM_TEMPERATURE,
        api_key = config.GROQ_API_KEY
    )

@lru_cache(maxsize=1)
def get_embedding_model() -> OpenAIEmbeddings:
    """
    Create (or reuse) the OpenAI embedding model used to build the vector index.

    An embedding turns a piece of text into a list of numbers (a vector).
    Texts with similar meaning get similar vectors, which is what makes
    "search by meaning" possible.

    Returns:
        A configured LangChain OpenAIEmbeddings model.

    Raises:
    """
    config.check_openai_key()

    return OpenAIEmbeddings(
        model = config.HF_EMBEDDING_MODEL,
        api_key = config.GROQ_API_KEY
    )