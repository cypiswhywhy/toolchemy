from .common import ILLMClient, LLMClientBase, ChatMessage, ChatMessages, ModelConfig, ModelResponseError, LLMCacheDoesNotExist, prepare_chat_messages
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
try:
    from .gemini_client import GeminiClient
except ImportError:  # google-genai is optional - install the 'gemini' extra for Gemini support
    GeminiClient = None
from .dummy_model_client import DummyModelClient
from .factory import create_llm


__all__ = [
    "ILLMClient", "LLMClientBase",
    "OllamaClient",
    "OpenAIClient",
    "GeminiClient",
    "DummyModelClient",
    "create_llm",
    "ModelConfig",
    "ModelResponseError",
    "LLMCacheDoesNotExist",
    "prepare_chat_messages",
    "ChatMessage", "ChatMessages"]
