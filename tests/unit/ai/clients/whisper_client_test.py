import asyncio
import subprocess
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from wyoming.asr import Transcript

from toolchemy.ai.clients.whisper_client import WhisperClient

AUDIO_RATE = 16000
AUDIO_WIDTH = 2
AUDIO_CHANNELS = 1


@pytest.fixture
def wav_path(tmp_path: Path) -> str:
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(AUDIO_CHANNELS)
        wav.setsampwidth(AUDIO_WIDTH)
        wav.setframerate(AUDIO_RATE)
        wav.writeframes(b"\x00\x01" * AUDIO_RATE)
    return str(path)


def _wyoming_client(url: str = "tcp://whisper:10300") -> WhisperClient:
    with patch("toolchemy.ai.clients.whisper_client.AsyncClient.from_uri"):
        return WhisperClient(url=url)


@pytest.mark.parametrize("url,expected_endpoint", [
    ("http://whisper:8080", "http://whisper:8080/transcribe"),
    ("http://whisper:8080/", "http://whisper:8080/transcribe"),
    ("http://whisper:8080/transcribe", "http://whisper:8080/transcribe"),
])
def test_http_endpoints_resolve_to_the_transcribe_path(url: str, expected_endpoint: str):
    assert WhisperClient(url=url)._endpoint == expected_endpoint


def test_an_unknown_protocol_is_rejected():
    with pytest.raises(ValueError, match="Unknown protocol"):
        WhisperClient(url="ftp://whisper:10300")


def test_convert_to_wav_reports_ffmpegs_own_error(tmp_path: Path):
    client = _wyoming_client()
    failure = subprocess.CalledProcessError(returncode=1, cmd=["ffmpeg"], stderr=b"Invalid data found\n")

    with patch("toolchemy.ai.clients.whisper_client.subprocess.run", side_effect=failure):
        with pytest.raises(RuntimeError, match="Invalid data found") as exc_info:
            client._convert_to_wav(str(tmp_path / "broken.mp3"), audio_rate=AUDIO_RATE, audio_channels=AUDIO_CHANNELS)

    assert isinstance(exc_info.value.__cause__, subprocess.CalledProcessError)


def test_convert_to_wav_runs_ffmpeg_with_check_so_a_failure_cannot_pass_silently(tmp_path: Path):
    client = _wyoming_client()

    with patch("toolchemy.ai.clients.whisper_client.subprocess.run") as run_mock:
        client._convert_to_wav(str(tmp_path / "audio.mp3"), audio_rate=AUDIO_RATE, audio_channels=AUDIO_CHANNELS)

    assert run_mock.call_args.kwargs["check"] is True


def _transcribe_with_events(client: WhisperClient, wav_path: str, events: list) -> str:
    wyoming = AsyncMock()
    wyoming.read_event.side_effect = events
    client._whisper_client_wyoming = wyoming

    with patch.object(client, "_convert_to_wav", return_value=wav_path):
        return asyncio.run(client._transcribe_wyoming(wav_path))


def test_wyoming_transcription_returns_the_first_non_empty_transcript(wav_path: str):
    client = _wyoming_client()
    events = [MagicMock(), Transcript(text="hello there").event()]

    assert _transcribe_with_events(client, wav_path, events) == "hello there"


def test_wyoming_transcription_raises_when_the_stream_closes_before_a_transcript(wav_path: str):
    client = _wyoming_client()
    events = [MagicMock(), None]

    with pytest.raises(RuntimeError, match="closed the stream before returning a transcript"):
        _transcribe_with_events(client, wav_path, events)


def test_http_transcription_rejects_a_missing_file(tmp_path: Path):
    client = WhisperClient(url="http://whisper:8080")

    with pytest.raises(ValueError, match="not found"):
        client._transcribe_http(str(tmp_path / "absent.mp3"))


def test_http_transcription_raises_on_a_non_200_response(wav_path: str):
    client = WhisperClient(url="http://whisper:8080")
    response = MagicMock(status_code=503)

    with patch("toolchemy.ai.clients.whisper_client.requests.post", return_value=response):
        with pytest.raises(RuntimeError, match="Status Code: 503"):
            client._transcribe_http(wav_path)


def test_transcribe_over_http_strips_the_returned_text(wav_path: str):
    client = WhisperClient(url="http://whisper:8080")
    response = MagicMock(status_code=200)
    response.json.return_value = {"text": "  padded transcription  "}

    with patch("toolchemy.ai.clients.whisper_client.requests.post", return_value=response):
        assert client.transcribe(wav_path) == "padded transcription"


def test_transcribe_raises_when_the_backend_returns_nothing(wav_path: str):
    client = WhisperClient(url="http://whisper:8080")
    response = MagicMock(status_code=200)
    response.json.return_value = {}

    with patch("toolchemy.ai.clients.whisper_client.requests.post", return_value=response):
        with pytest.raises(RuntimeError, match="Transcription failed"):
            client.transcribe(wav_path)
