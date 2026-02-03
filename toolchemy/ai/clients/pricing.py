KEY_INPUT_TOKENS = "input_tokens_cost"
KEY_OUTPUT_TOKENS = "output_tokens_cost"



class Pricing:
    pricing_per_1_mln = {
        "gpt-5.2": {
            KEY_INPUT_TOKENS: 1.75,
            KEY_OUTPUT_TOKENS: 14.00,
        },
        "gpt-5.2-pro": {
            KEY_INPUT_TOKENS: 21.00,
            KEY_OUTPUT_TOKENS: 168.00,
        },
        "gpt-5-mini": {
            KEY_INPUT_TOKENS: 0.25,
            KEY_OUTPUT_TOKENS: 2.00,
        },
        "gpt-4.1": {
            KEY_INPUT_TOKENS: 3.00,
            KEY_OUTPUT_TOKENS: 12.00,
        },
        "gpt-4.1-mini": {
            KEY_INPUT_TOKENS: 0.8,
            KEY_OUTPUT_TOKENS: 3.2,
        },
        "gpt-4.1-nano": {
            KEY_INPUT_TOKENS: 0.2,
            KEY_OUTPUT_TOKENS: 0.8,
        },
        "o4-mini": {
            KEY_INPUT_TOKENS: 4.00,
            KEY_OUTPUT_TOKENS: 16.00,
        },
        "mistral-small3.2:24b": {
            KEY_INPUT_TOKENS: 0.01,
            KEY_OUTPUT_TOKENS: 0.03,
        },
        "gpt-oss:120b": {
            KEY_INPUT_TOKENS: 0.01,
            KEY_OUTPUT_TOKENS: 0.03,
        },
        "gemma3:27b": {
            KEY_INPUT_TOKENS: 0.01,
            KEY_OUTPUT_TOKENS: 0.03,
        },
        "qwen3:32b-q8_0": {
            KEY_INPUT_TOKENS: 0.01,
            KEY_OUTPUT_TOKENS: 0.03,
        },
        "dummy-model": {
            KEY_INPUT_TOKENS: 0.1,
            KEY_OUTPUT_TOKENS: 1.0,
        }
    }

    @classmethod
    def estimate(cls, model_name: str, input_tokens: int, output_tokens: int) -> float:
        if model_name not in cls.pricing_per_1_mln:
            raise ValueError(f"Model '{model_name}' not supported for pricing estimation")

        model_pricing = cls.pricing_per_1_mln[model_name]
        input_cost = model_pricing[KEY_INPUT_TOKENS] * (input_tokens / 1_000_000)
        output_cost = model_pricing[KEY_OUTPUT_TOKENS] * (output_tokens / 1_000_000)
        return input_cost + output_cost
