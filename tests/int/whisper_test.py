from toolchemy.ai.clients.whisper_client import WhisperClient
from toolchemy.utils.locations import Locations


def test_whisper():
    locations = Locations()
    audio_path = locations.in_resources("tests/ai/output_pl.mp3")

    whisper_client = WhisperClient(url=f"tcp://hal:10300")
    transcription = whisper_client.transcribe(audio_path)

    assert transcription == "Dzisiaj była bardzo ładna pogoda, więc poszedłem z córką na sanki.  I było fajnie."
