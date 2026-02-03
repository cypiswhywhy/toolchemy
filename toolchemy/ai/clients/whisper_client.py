import asyncio
import requests
import sys
import os
import subprocess
import tempfile
from wyoming.client import AsyncClient
from wyoming.audio import AudioChunk, AudioStart, AudioStop, AudioChunkConverter
from wyoming.asr import Transcript
from wyoming.ping import Ping
from wyoming.info import Describe
import wave

from toolchemy.utils.logger import get_logger


class WhisperClient:
    def __init__(self, url: str):
        self._logger = get_logger()
        self._endpoint = url
        self._whisper_client_wyoming = None
        if self._endpoint.startswith("http"):
            if not self._endpoint.endswith("transcribe"):
                if not self._endpoint.endswith("/"):
                    self._endpoint += "/"
                self._endpoint += "transcribe"
        elif self._endpoint.startswith("tcp"):
            self._whisper_client_wyoming = AsyncClient.from_uri(self._endpoint)
        else:
            raise ValueError(f"Unknown protocol for the whisper server endpoint: '{self._endpoint}'")
        self._logger.info(f"Whisper client initialized (endpoint: '{self._endpoint}')")

    def transcribe(self, audio_path: str) -> str:
        transcription = None

        if self._endpoint.startswith("tcp"):
            transcription = asyncio.run(self._transcribe_wyoming(audio_path))

        if self._endpoint.startswith("http"):
            transcription = self._transcribe_http(audio_path)

        if transcription is None:
            raise RuntimeError(f"Transcription failed...")

        return transcription.strip()

    def _transcribe_http(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            raise ValueError(f"Error: File '{audio_path}' not found.")

        with open(audio_path, "rb") as audio_file:
            files = {"file": audio_file}

            self._logger.info(f"Sending '{audio_path}' to Whisper server...")
            response = requests.post(self._endpoint, files=files)

        if response.status_code == 200:
            result = response.json()
            result_transcription = result.get("text")
            self._logger.info(f"Transcription: '{result_transcription}'")
            return result_transcription

        err_msg = f"Error: Failed to transcribe. Status Code: {response.status_code}"
        self._logger.error(err_msg)
        raise RuntimeError(err_msg)

    async def _transcribe_wyoming(self, audio_path: str, audio_rate: int = 16000, audio_width: int = 2,
                                  audio_channels: int = 1, chunk_size: int = 1024) -> str:
        wav_path = self._convert_to_wav(audio_path, audio_rate=audio_rate, audio_channels=audio_channels)

        await self._whisper_client_wyoming.connect()
        await self._whisper_client_wyoming.write_event(Ping(text="test").event())

        await self._whisper_client_wyoming.write_event(Describe().event())

        info_event = await self._whisper_client_wyoming.read_event()
        self._logger.info(f"info event: {info_event}")

        with wave.open(wav_path, 'rb') as wav:
            assert wav.getframerate() == audio_rate
            assert wav.getsampwidth() == audio_width
            assert wav.getnchannels() == audio_channels

            await self._whisper_client_wyoming.write_event(AudioStart(audio_rate, audio_width, audio_channels).event())

            audio_bytes = wav.readframes(chunk_size)

            converter = AudioChunkConverter(
                rate=audio_rate, width=audio_width, channels=audio_channels,
            )

            while audio_bytes:
                chunk = converter.convert(AudioChunk(audio_rate, audio_width, audio_channels, audio_bytes))
                await self._whisper_client_wyoming.write_event(chunk.event())
                audio_bytes = wav.readframes(chunk_size)

            await self._whisper_client_wyoming.write_event(AudioStop().event())

            while True:
                event = await asyncio.wait_for(self._whisper_client_wyoming.read_event(), timeout=30)
                if event is None:
                    break
                transcript = Transcript.from_event(event)
                if transcript.text:
                    self._logger.info(f"Transcription: {transcript.text}")
                    break
        await self._whisper_client_wyoming.disconnect()

        return transcript.text

    def _convert_to_wav(self, input_path: str, audio_rate: int, audio_channels: int) -> str:
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.close()

        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-ar", str(audio_rate),
            "-ac", str(audio_channels),
            "-f", "wav",
            "-sample_fmt", "s16",
            temp_wav.name
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return temp_wav.name


def main(argv: list):
    if len(argv) < 2:
        raise ValueError("Usage: python transcribe_audio.py <audio_file.mp3/wav>")

    file_path = argv[1]
    client = WhisperClient(url="tcp://hal:10300")
    transcription = client.transcribe(file_path)
    print(f"> '{transcription}'")


if __name__ == "__main__":
    main(sys.argv)
