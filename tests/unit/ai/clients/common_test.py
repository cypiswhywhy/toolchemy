from toolchemy.ai.clients import ModelConfig, prepare_chat_messages
from toolchemy.utils.utils import ff


def test_model_config():
    expected_model_name = "dummy"
    expected_max_tokens = 100
    expected_temperature = 0.05
    expected_top_p = 0.85
    expected_presence_penalty = 0.5

    model_config = ModelConfig(model_name=expected_model_name, max_new_tokens=expected_max_tokens,
                               temperature=expected_temperature, top_p=expected_top_p,
                               presence_penalty=expected_presence_penalty)

    expected_model_config_str = f"{expected_model_name}__{expected_max_tokens}__{ff(expected_presence_penalty)}__{ff(expected_temperature)}__{ff(expected_top_p)}"

    assert str(model_config) == expected_model_config_str


def test_prepare_chat_messages():
    prompt = "foo"
    expected_messages = [{"role": "user", "content": prompt}]

    assert prepare_chat_messages(prompt) == expected_messages


def test_prepare_chat_messages_with_system_prompt():
    prompt = "foo"
    system_prompt = "system_prompt"
    expected_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    messages = prepare_chat_messages(prompt, system_prompt=system_prompt)

    assert messages == expected_messages


def test_prepare_chat_messages_with_system_prompt_enveloped():
    prompt = "foo"
    system_prompt = "system_prompt"
    expected_messages = {"messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]}

    messages = prepare_chat_messages(prompt, system_prompt=system_prompt, envelope=True)

    assert messages == expected_messages


def test_prepare_chat_messages_with_history():
    prompt1 = "foo"
    response1 = "foo-response"
    prompt2 = "bar"
    expected_messages = [
        {"role": "user", "content": prompt1},
        {"role": "assistant", "content": response1},
        {"role": "user", "content": prompt2},
    ]

    messages = prepare_chat_messages(prompt=prompt2, messages_history=expected_messages[:2])

    assert messages == expected_messages


def test_prepare_chat_messages_with_history_and_system_prompt():
    system_prompt = "system_prompt"
    prompt1 = "foo"
    response1 = "foo-response"
    prompt2 = "bar"
    expected_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt1},
        {"role": "assistant", "content": response1},
        {"role": "user", "content": prompt2},
    ]

    messages = prepare_chat_messages(prompt=prompt2, system_prompt=system_prompt, messages_history=expected_messages[:3])

    assert messages == expected_messages


def test_prepare_chat_messages_with_history_and_implicite_system_prompt():
    system_prompt = "system_prompt"
    prompt1 = "foo"
    response1 = "foo-response"
    prompt2 = "bar"
    expected_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt1},
        {"role": "assistant", "content": response1},
        {"role": "user", "content": prompt2},
    ]

    messages = prepare_chat_messages(prompt=prompt2, system_prompt=system_prompt, messages_history=expected_messages[1:3])

    assert messages == expected_messages
