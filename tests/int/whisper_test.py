import os

import pytest

from toolchemy.ai.clients.whisper_client import WhisperClient
from toolchemy.utils.locations import Locations

WHISPER_URL_ENV_VAR = "TOOLCHEMY_WHISPER_URL"


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get(WHISPER_URL_ENV_VAR),
                    reason=f"needs a live Whisper server; set {WHISPER_URL_ENV_VAR} (e.g. tcp://host:10300) to run")
def test_whisper():
    locations = Locations()
    audio_path = locations.in_resources("tests/ai/output_pl.mp3")

    whisper_client = WhisperClient(url=os.environ[WHISPER_URL_ENV_VAR])
    transcription = whisper_client.transcribe(audio_path)

    assert transcription == "Dzisiaj była bardzo ładna pogoda, więc poszedłem z córką na sanki.  I było fajnie."
