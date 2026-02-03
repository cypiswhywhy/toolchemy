import logging
import unittest
from unittest.mock import patch

from toolchemy.ai.clients import ModelConfig
from toolchemy.ai.clients.factory import create_llm, URI_OPENAI, URI_GEMINI


class TestCreateLLM(unittest.TestCase):
    @patch('toolchemy.ai.clients.factory.OpenAIClient')
    def test_create_openai_client_explicit_uri(self, mock_openai):
        api_key = "sk-test-key"
        name = "gpt-4"

        result = create_llm(name=name, uri=URI_OPENAI, api_key=api_key)

        self.assertEqual(result, mock_openai.return_value)

        mock_openai.assert_called_once_with(
            model_name=name,
            api_key=api_key,
            default_model_config=None,
            system_prompt=None,
            no_cache=False
        )

    @patch('toolchemy.ai.clients.factory.OpenAIClient')
    def test_create_openai_client_implicit_uri(self, mock_openai):
        api_key = "sk-test-key"
        name = "gpt-4.1"

        result = create_llm(name=name, api_key=api_key)

        self.assertEqual(result, mock_openai.return_value)

    @patch('toolchemy.ai.clients.factory.GeminiClient')
    def test_create_gemini_client_explicit_uri(self, mock_gemini):
        api_key = "gemini-key"
        name = "gemini-pro"

        result = create_llm(name=name, uri=URI_GEMINI, api_key=api_key)

        self.assertEqual(result, mock_gemini.return_value)
        mock_gemini.assert_called_once_with(
            default_model_name=name,
            default_model_config=None,
            api_key=api_key,
            system_prompt=None,
            disable_cache=False,
            log_level=logging.INFO,
        )

    @patch('toolchemy.ai.clients.factory.GeminiClient')
    def test_create_gemini_client_implicit_uri(self, mock_gemini):
        api_key = "gemini-key"
        name = "gemini-1.5-flash"

        result = create_llm(name=name, uri=None, api_key=api_key)

        self.assertEqual(result, mock_gemini.return_value)

    @patch('toolchemy.ai.clients.factory.OllamaClient')
    def test_create_ollama_client(self, mock_ollama):
        uri = "http://localhost:11434"
        name = "gpt-oss:120b"
        system_prompt = "Be helpful."

        result = create_llm(name=name, uri=uri, api_key="dummy", system_prompt=system_prompt)

        self.assertEqual(result, mock_ollama.return_value)
        mock_ollama.assert_called_once_with(
            uri=uri,
            model_name=name,
            default_model_config=None,
            system_prompt=system_prompt,
            truncate_log_messages_to=2000,
            disable_cache=False,
            log_level=logging.INFO,
        )

    @patch('toolchemy.ai.clients.factory.OllamaClient')
    def test_create_ollama_client_with_default_model_config(self, mock_ollama):
        uri = "http://localhost:11434"
        name = "gpt-oss:120b"
        system_prompt = "Be helpful."
        expected_model_config = ModelConfig(max_new_tokens=8000)

        result = create_llm(name=name, uri=uri, api_key="dummy", system_prompt=system_prompt, default_model_config=expected_model_config)

        self.assertEqual(result, mock_ollama.return_value)
        mock_ollama.assert_called_once_with(
            uri=uri,
            model_name=name,
            default_model_config=expected_model_config,
            system_prompt=system_prompt,
            truncate_log_messages_to=2000,
            disable_cache=False,
            log_level=logging.INFO,
        )

    def test_openai_missing_api_key_raises_error(self):
        with self.assertRaises(ValueError) as context:
            create_llm(name="gpt-4", uri=URI_OPENAI, api_key="")

        self.assertIn("must pass the 'api_key' explicitly", str(context.exception))

    def test_gemini_missing_api_key_raises_error(self):
        with self.assertRaises(ValueError) as context:
            create_llm(name="gemini-pro", uri=URI_GEMINI, api_key=None)

        self.assertIn("must pass the 'api_key' explicitly", str(context.exception))

    def test_inference_failure_unknown_name(self):
        with self.assertRaises(ValueError) as context:
            create_llm(name="claude-3-opus", uri=None, api_key="key")

        self.assertIn("Cannot assume the LLM provider", str(context.exception))

    def test_inference_failure_gpt_oss(self):
        with self.assertRaises(ValueError) as context:
            create_llm(name="gpt-oss-model", uri=None, api_key="key")

        self.assertIn("Cannot assume the LLM provider", str(context.exception))


if __name__ == '__main__':
    unittest.main()
