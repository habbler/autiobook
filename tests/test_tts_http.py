"""tests for the http tts engine across local and hosted backends."""

import argparse
import base64
import http.client
import io
import json
import urllib.error
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf  # type: ignore

from autiobook.config import (
    GEMINI_API_BASE,
    SAMPLE_RATE,
    TTS_DIALECT_GEMINI,
    TTS_DIALECT_OPENAI,
    TTS_DIALECT_QWEN,
    TTS_DIRECTION_PREFIX,
    TTS_FORMAT_MP3,
    TTS_FORMAT_PCM,
    TTS_FORMAT_WAV,
)
from autiobook.tts_http import HTTPTTSConfig, HTTPTTSEngine

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
LOCAL_BASE = "http://localhost:8080/v1"
# comfortably more than one STREAM_READ_BYTES buffer of s16 samples
LONG_TAKE_SAMPLES = 20000


def _wav_bytes(seconds: float = 0.01) -> bytes:
    """a real wav payload so decoding is exercised, not mocked."""
    buf = io.BytesIO()
    samples = np.zeros(int(SAMPLE_RATE * seconds), dtype="float32")
    sf.write(buf, samples, SAMPLE_RATE, format="WAV")
    return buf.getvalue()


def _pcm_bytes(count: int = 8) -> bytes:
    return np.arange(count, dtype=np.int16).tobytes()


def _dropped() -> Exception:
    """what urlopen raises when the server closes the socket mid-request."""
    return http.client.RemoteDisconnected(
        "Remote end closed connection without response"
    )


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        f"{OPENROUTER_BASE}/audio/speech", code, "err", {}, io.BytesIO(b"upstream")
    )


def _mock_response(payload: bytes):
    resp = MagicMock()
    resp.read.return_value = payload
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    # SSE parsing iterates the response line by line
    resp.__iter__ = lambda s: iter(payload.splitlines(keepends=True))
    return resp


def _stream_response(*chunks):
    """a response body read incrementally: each chunk in turn, then EOF."""
    resp = MagicMock()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    resp.read1.side_effect = [*chunks, b""]
    return resp


def _engine(**kwargs) -> HTTPTTSEngine:
    return HTTPTTSEngine(HTTPTTSConfig(model="tts-1", **kwargs))


def _bodies(mock_url) -> list[dict]:
    return [json.loads(c[0][0].data) for c in mock_url.call_args_list]


class TestDialectResolution:
    """auto-detection of the request subset a backend understands."""

    def test_openrouter_host_resolves_openai(self):
        assert _engine(api_base=OPENROUTER_BASE).dialect == TTS_DIALECT_OPENAI

    def test_openai_host_resolves_openai(self):
        engine = _engine(api_base="https://api.openai.com/v1")
        assert engine.dialect == TTS_DIALECT_OPENAI

    def test_local_host_resolves_qwen(self):
        assert _engine(api_base=LOCAL_BASE).dialect == TTS_DIALECT_QWEN

    def test_explicit_dialect_wins_over_host(self):
        engine = _engine(api_base=OPENROUTER_BASE, dialect=TTS_DIALECT_QWEN)
        assert engine.dialect == TTS_DIALECT_QWEN

    def test_response_format_follows_dialect(self):
        assert _engine(api_base=OPENROUTER_BASE).response_format == TTS_FORMAT_PCM
        assert _engine(api_base=LOCAL_BASE).response_format == TTS_FORMAT_WAV

    def test_explicit_response_format_wins(self):
        engine = _engine(api_base=OPENROUTER_BASE, response_format=TTS_FORMAT_MP3)
        assert engine.response_format == TTS_FORMAT_MP3


class TestAuth:
    """bearer auth reaches every tts request, not just the llm ones."""

    def test_api_key_sent_on_speech(self):
        engine = _engine(api_base=OPENROUTER_BASE, api_key="sk-or-test")
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_pcm_bytes())
        ) as mock_url:
            engine.synthesize("hello")
            req = mock_url.call_args[0][0]
            assert req.get_header("Authorization") == "Bearer sk-or-test"

    def test_no_key_sends_no_auth_header(self):
        engine = _engine(api_base=LOCAL_BASE)
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_wav_bytes())
        ) as mock_url:
            engine.synthesize("hello")
            req = mock_url.call_args[0][0]
            assert req.get_header("Authorization") is None

    def test_api_key_sent_on_voices_listing(self):
        engine = _engine(api_base=LOCAL_BASE, api_key="sk-test")
        payload = json.dumps({"tts-1": ["ryan"]}).encode()
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(payload)
        ) as mock_url:
            assert engine.list_voices() == ["ryan"]
            req = mock_url.call_args[0][0]
            assert req.get_header("Authorization") == "Bearer sk-test"


class TestOpenAIDialectRequests:
    """hosted providers accept a narrow body and bill every round trip."""

    def test_body_limited_to_supported_fields(self):
        engine = _engine(
            api_base=OPENROUTER_BASE,
            api_key="sk-or-test",
            speaker="Zephyr",
            temperature=0.7,
            top_k=20,
            repetition_penalty=1.1,
            seed=1234,
        )
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_pcm_bytes())
        ) as mock_url:
            engine.synthesize("hello")
            body = _bodies(mock_url)[0]

        assert body == {
            "model": "tts-1",
            "input": "hello",
            "voice": "Zephyr",
            "response_format": TTS_FORMAT_PCM,
        }

    def test_single_request_no_sse_probe(self):
        """the qwen sse probe would be a second billed synthesis."""
        engine = _engine(api_base=OPENROUTER_BASE, api_key="sk-or-test")
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_pcm_bytes())
        ) as mock_url:
            engine.synthesize("hello")

        assert mock_url.call_count == 1
        assert "stream_format" not in _bodies(mock_url)[0]

    def test_pcm_response_decoded(self):
        engine = _engine(api_base=OPENROUTER_BASE, api_key="sk-or-test")
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_pcm_bytes(4))
        ):
            audio, sr = engine.synthesize("hello")

        assert sr == SAMPLE_RATE
        assert audio.dtype == np.float32
        np.testing.assert_allclose(
            audio, np.arange(4, dtype=np.float32) / 32768.0, rtol=1e-6
        )

    def test_instructions_sent_as_field_by_default(self):
        engine = _engine(api_base=OPENROUTER_BASE)
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_pcm_bytes())
        ) as mock_url:
            engine.synthesize("hello", instruct="speaks joyfully")
            body = _bodies(mock_url)[0]

        assert body["instructions"] == "speaks joyfully"
        assert body["input"] == "hello"

    def test_prefix_direction_folds_into_input(self):
        """providers that drop unknown fields only take direction inline."""
        engine = _engine(api_base=OPENROUTER_BASE, direction=TTS_DIRECTION_PREFIX)
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_pcm_bytes())
        ) as mock_url:
            engine.synthesize("hello", instruct="speaks joyfully")
            body = _bodies(mock_url)[0]

        assert "instructions" not in body
        assert "speaks joyfully" in body["input"]
        assert body["input"].endswith("hello")


class TestQwenDialectRequests:
    """the local server keeps its sse probe and sampler fields."""

    def test_sse_probe_first_then_falls_back(self):
        """a server that cannot stream still answers the plain wav retry."""
        engine = _engine(api_base=LOCAL_BASE)
        sse_bodies: list[dict] = []

        def record_sse(url, body, api_key=""):
            sse_bodies.append(body)
            raise RuntimeError("no sse")

        with patch("autiobook.tts_http._post_sse", side_effect=record_sse):
            with patch(
                "urllib.request.urlopen", return_value=_mock_response(_wav_bytes())
            ) as mock_url:
                engine.synthesize("hello")

        assert sse_bodies[0]["stream_format"] == "sse"
        assert "stream_format" not in _bodies(mock_url)[0]

    def test_sampler_fields_sent(self):
        engine = _engine(
            api_base=LOCAL_BASE,
            temperature=0.7,
            top_k=20,
            repetition_penalty=1.1,
            seed=99,
        )
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_wav_bytes())
        ) as mock_url:
            with patch(
                "autiobook.tts_http._post_sse", side_effect=RuntimeError("no sse")
            ):
                engine.synthesize("hello")
            body = _bodies(mock_url)[0]

        assert body["temperature"] == 0.7
        assert body["top_k"] == 20
        assert body["repetition_penalty"] == 1.1
        assert body["seed"] == 99
        assert body["language"] == "en"
        assert body["response_format"] == TTS_FORMAT_WAV


class TestPresetVoices:
    """--preset-voices needs a voice list even without a discovery endpoint."""

    def test_known_model_voices_without_request(self):
        engine = _engine(api_base=OPENROUTER_BASE)
        engine.config.model = "google/gemini-3.1-flash-tts-preview"
        with patch("urllib.request.urlopen") as mock_url:
            voices = engine.list_voices()

        mock_url.assert_not_called()
        assert voices[0] == "Zephyr"
        assert "Sulafat" in voices

    def test_configured_voices_override_known(self):
        engine = _engine(api_base=OPENROUTER_BASE, voices=["Puck", "Kore"])
        engine.config.model = "google/gemini-3.1-flash-tts-preview"
        assert engine.list_voices() == ["Puck", "Kore"]

    def test_unknown_hosted_model_reports_how_to_fix(self):
        engine = _engine(api_base=OPENROUTER_BASE)
        engine.config.model = "some/unlisted-tts"
        with pytest.raises(RuntimeError, match="--tts-voices"):
            engine.list_voices()

    def test_qwen_still_queries_the_endpoint(self):
        engine = _engine(api_base=LOCAL_BASE)
        payload = json.dumps({"tts-1": ["ryan", "vivian"]}).encode()
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(payload)
        ) as mock_url:
            assert engine.list_voices() == ["ryan", "vivian"]
            assert "/audio/voices" in mock_url.call_args[0][0].full_url


class TestUnsupportedFeatures:
    """hosted backends have no server-side voice creation."""

    def test_clone_voice_explains_the_alternative(self):
        engine = _engine(api_base=OPENROUTER_BASE, api_key="sk-or-test")
        with pytest.raises(RuntimeError, match="--preset-voices"):
            engine.clone_voice("hello", np.zeros(10, dtype="float32"), "ref")


class TestStaleServerVoice:
    """server voice ids live in the model process, so a swap forgets them."""

    def _engine_and_ref(self, tmp_path):
        from autiobook.tts_http import _voice_cache

        _voice_cache.clear()
        ref = tmp_path / "Narrator.wav"
        sf.write(str(ref), np.zeros(SAMPLE_RATE // 10, dtype="float32"), SAMPLE_RATE)
        return _engine(api_base=LOCAL_BASE), ref

    def test_reregisters_and_retries(self, tmp_path):
        created = []

        def fake_create(self, ref_audio, ref_text, refresh=False):
            created.append(refresh)
            return f"voice_{len(created)}"

        calls = []

        def fake_synth(self, text, voice="", instruct=""):
            calls.append(voice)
            if voice == "voice_1":
                raise RuntimeError(
                    'http 400: {"error":{"message":"unknown voice '
                    '\'voice_1\'","type":"invalid_request_error"}}'
                )
            return np.zeros(10, dtype="float32")

        engine, ref = self._engine_and_ref(tmp_path)
        with patch.object(HTTPTTSEngine, "_get_or_create_voice", fake_create):
            with patch.object(HTTPTTSEngine, "_synthesize_one", fake_synth):
                out, sr = engine.clone_voice(["one", "two"], str(ref), "ref text")

        assert len(out) == 2 and sr == SAMPLE_RATE
        # the stale id is retried once against a freshly registered voice
        assert created == [False, True]
        assert calls == ["voice_1", "voice_2", "voice_2"]

    def test_other_errors_are_not_retried(self, tmp_path):
        def fake_synth(self, text, voice="", instruct=""):
            raise RuntimeError("http 500: server on fire")

        engine, ref = self._engine_and_ref(tmp_path)
        with patch.object(
            HTTPTTSEngine, "_get_or_create_voice", lambda *a, **k: "voice_1"
        ):
            with patch.object(HTTPTTSEngine, "_synthesize_one", fake_synth):
                with pytest.raises(RuntimeError, match="on fire"):
                    engine.clone_voice("one", str(ref), "ref text")

    def test_refresh_bypasses_the_cache(self, tmp_path):
        from autiobook.tts_http import _voice_cache

        engine, ref = self._engine_and_ref(tmp_path)
        posted = []

        def fake_post(url, fields, files, api_key=""):
            posted.append(url)
            return {"id": f"voice_{len(posted)}"}

        with patch("autiobook.tts_http._post_multipart", fake_post):
            first = engine._get_or_create_voice(str(ref), "ref text")
            cached = engine._get_or_create_voice(str(ref), "ref text")
            fresh = engine._get_or_create_voice(str(ref), "ref text", refresh=True)

        assert (first, cached, fresh) == ("voice_1", "voice_1", "voice_2")
        assert len(posted) == 2
        # the new id replaces the stale one for later calls
        assert _voice_cache[(LOCAL_BASE, str(ref), "ref text")] == "voice_2"


class TestStreaming:
    """progressive playback, by sse locally and chunked reads when hosted."""

    def test_hosted_backends_always_stream(self):
        """no batch size to configure: the response body is the stream."""
        assert _engine(api_base=OPENROUTER_BASE).streaming is True

    def test_qwen_streams_only_when_asked(self):
        assert _engine(api_base=LOCAL_BASE).streaming is False
        assert _engine(api_base=LOCAL_BASE, stream_batch_size=16).streaming is True

    def test_hosted_stream_reads_body_incrementally(self, stub_provider):
        engine = HTTPTTSEngine(
            HTTPTTSConfig(
                api_base=stub_provider,
                api_key="sk-or-test",
                dialect=TTS_DIALECT_OPENAI,
                model="tts-1",
            )
        )
        chunks: list[bytes] = []
        audio, sr = engine.design_voice_stream(
            "long", "warm voice", on_chunk=chunks.append, voice="Puck"
        )

        # the point of streaming: audio is handed over before the take ends
        assert len(chunks) > 1
        assert b"".join(chunks) == _pcm_bytes(LONG_TAKE_SAMPLES)
        assert len(audio) == LONG_TAKE_SAMPLES
        assert sr == SAMPLE_RATE
        sent = StubProvider.requests[0]["body"]
        assert sent["voice"] == "Puck"
        assert sent["response_format"] == TTS_FORMAT_PCM
        assert "stream_format" not in sent

    def test_hosted_stream_forces_pcm_over_configured_format(self, stub_provider):
        """the player is fed raw samples; a container would arrive header-first."""
        engine = HTTPTTSEngine(
            HTTPTTSConfig(
                api_base=stub_provider,
                api_key="sk-or-test",
                dialect=TTS_DIALECT_OPENAI,
                response_format=TTS_FORMAT_MP3,
                model="tts-1",
            )
        )
        engine.design_voice_stream("hello", "warm voice")
        assert StubProvider.requests[0]["body"]["response_format"] == TTS_FORMAT_PCM

    def test_qwen_still_uses_sse(self):
        engine = _engine(api_base=LOCAL_BASE, stream_batch_size=8)
        with patch(
            "autiobook.tts_http._post_sse",
            return_value=(_pcm_bytes(2), {}, {}),
        ) as mock_sse:
            engine.design_voice_stream("hello", "warm voice")

        body = mock_sse.call_args[0][1]
        assert body["stream_format"] == "sse"
        assert body["stream_batch_size"] == 8


class TestTransientRetry:
    """a dropped socket must not end a multi-hour synthesis run."""

    def _synthesize(self, side_effect, api_base=OPENROUTER_BASE):
        """synthesize one take against a scripted sequence of urlopen results."""
        engine = _engine(api_base=api_base)
        with patch("time.sleep") as sleep:
            with patch("urllib.request.urlopen", side_effect=side_effect) as mock_url:
                audio, _ = engine.synthesize("hello")
        return audio, mock_url, sleep

    def test_dropped_connection_is_retried(self):
        audio, mock_url, _ = self._synthesize(
            [_dropped(), _mock_response(_pcm_bytes(4))]
        )

        assert mock_url.call_count == 2
        assert len(audio) == 4

    def test_backoff_doubles_between_attempts(self):
        from autiobook.config import TTS_RETRY_DELAY

        _, _, sleep = self._synthesize(
            [_dropped(), _dropped(), _mock_response(_pcm_bytes(4))]
        )

        delays = [c[0][0] for c in sleep.call_args_list]
        assert delays == [TTS_RETRY_DELAY, TTS_RETRY_DELAY * 2]

    def _failing_synthesis(self, side_effect, match: str) -> int:
        """synthesize against a failure; return how many requests were sent."""
        engine = _engine(api_base=OPENROUTER_BASE)
        with patch("time.sleep"):
            with patch("urllib.request.urlopen", side_effect=side_effect) as mock_url:
                with pytest.raises(Exception, match=match):
                    engine.synthesize("hello")
        return int(mock_url.call_count)

    def test_gives_up_after_max_retries(self):
        from autiobook.config import TTS_MAX_RETRIES

        sent = self._failing_synthesis(_dropped(), "Remote end closed")
        assert sent == TTS_MAX_RETRIES + 1

    def test_server_error_is_retried(self):
        audio, mock_url, _ = self._synthesize(
            [_http_error(503), _mock_response(_pcm_bytes(4))]
        )

        assert mock_url.call_count == 2
        assert len(audio) == 4

    def test_rate_limit_is_retried(self):
        _, mock_url, _ = self._synthesize(
            [_http_error(429), _mock_response(_pcm_bytes(4))]
        )

        assert mock_url.call_count == 2

    def test_client_error_fails_fast(self):
        """a bad body or unknown voice answers the same way however often it is
        sent, and every send is billed."""
        assert self._failing_synthesis(_http_error(400), "http 400") == 1

    def test_voice_listing_is_retried(self):
        engine = _engine(api_base=LOCAL_BASE)
        payload = json.dumps({"tts-1": ["ryan"]}).encode()
        with patch("time.sleep"):
            with patch(
                "urllib.request.urlopen",
                side_effect=[_dropped(), _mock_response(payload)],
            ) as mock_url:
                assert engine.list_voices() == ["ryan"]
        assert mock_url.call_count == 2

    def test_qwen_probe_falls_back_after_a_dropped_sse(self):
        """the sse probe's failure route already exists; a reset socket takes it."""
        engine = _engine(api_base=LOCAL_BASE)
        with patch("autiobook.tts_http._post_sse", side_effect=_dropped()):
            with patch(
                "urllib.request.urlopen", return_value=_mock_response(_wav_bytes())
            ) as mock_url:
                audio, sr = engine.synthesize("hello")

        assert sr == SAMPLE_RATE and len(audio) > 0
        assert mock_url.call_count == 1

    def test_stream_retried_before_any_audio(self):
        engine = _engine(api_base=OPENROUTER_BASE)
        chunks: list[bytes] = []
        with patch("time.sleep"):
            with patch(
                "urllib.request.urlopen",
                side_effect=[_dropped(), _stream_response(_pcm_bytes(4))],
            ) as mock_url:
                engine.design_voice_stream("hello", "warm", on_chunk=chunks.append)

        assert mock_url.call_count == 2
        assert b"".join(chunks) == _pcm_bytes(4)

    def test_stream_not_replayed_once_the_player_has_audio(self):
        """a retry mid-take would play its opening seconds twice."""
        engine = _engine(api_base=OPENROUTER_BASE)
        broken = _stream_response()
        broken.read1.side_effect = [_pcm_bytes(2), _dropped()]
        chunks: list[bytes] = []
        with patch("time.sleep"):
            with patch(
                "urllib.request.urlopen",
                side_effect=[broken, _stream_response(_pcm_bytes(4))],
            ) as mock_url:
                with pytest.raises(Exception, match="Remote end closed"):
                    engine.design_voice_stream("hello", "warm", on_chunk=chunks.append)

        assert mock_url.call_count == 1
        assert chunks == [_pcm_bytes(2)]


class StubProvider(BaseHTTPRequestHandler):
    """minimal hosted tts backend: bearer auth, raw pcm, no voices endpoint."""

    requests: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        StubProvider.requests.append(
            {
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "body": body,
            }
        )
        if body.get("response_format") != TTS_FORMAT_PCM:
            self.send_error(400, "unsupported response_format")
            return
        # "long" asks for more audio than one socket read returns, so a
        # streaming client observably receives it in pieces
        payload = _pcm_bytes(LONG_TAKE_SAMPLES if body["input"] == "long" else 4)
        self.send_response(200)
        self.send_header("Content-Type", "audio/pcm")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self.send_error(404, "not found")

    def log_message(self, *args):
        pass


@pytest.fixture
def stub_provider():
    """a real http server, so headers and framing are exercised end to end."""
    StubProvider.requests = []
    server = HTTPServer(("127.0.0.1", 0), StubProvider)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    yield f"http://{host}:{port}/v1"
    server.shutdown()
    server.server_close()


class TestAgainstStubProvider:
    """full request/response cycle over a socket, no mocks."""

    def _engine(self, base: str) -> HTTPTTSEngine:
        return HTTPTTSEngine(
            HTTPTTSConfig(
                api_base=base,
                api_key="sk-or-test",
                dialect=TTS_DIALECT_OPENAI,
                model="google/gemini-3.1-flash-tts-preview",
                speaker="Zephyr",
            )
        )

    def test_synthesis_round_trip(self, stub_provider):
        audio, sr = self._engine(stub_provider).synthesize("hello")

        assert sr == SAMPLE_RATE
        assert len(audio) == 4
        sent = StubProvider.requests[0]
        assert sent["path"] == "/v1/audio/speech"
        assert sent["auth"] == "Bearer sk-or-test"
        assert sent["body"]["voice"] == "Zephyr"

    def test_one_request_per_chunk(self, stub_provider):
        self._engine(stub_provider).synthesize(["one", "two"])
        assert len(StubProvider.requests) == 2

    def test_voices_resolve_without_an_endpoint(self, stub_provider):
        """the stub 404s /audio/voices, as openrouter does."""
        voices = self._engine(stub_provider).list_voices()
        assert "Zephyr" in voices
        assert StubProvider.requests == []


class TestConfigFromArgs:
    """cli/env wiring reaches the engine."""

    def _args(self, **kwargs):
        defaults = dict(
            api_base=None,
            api_key=None,
            tts_api_base=None,
            tts_api_key=None,
            tts_dialect="auto",
            tts_direction="field",
            tts_voices="",
            tts_model="",
        )
        return argparse.Namespace(**{**defaults, **kwargs})

    def test_api_key_reaches_tts_config(self):
        from autiobook.utils import get_tts_config

        config = get_tts_config(
            self._args(api_base=OPENROUTER_BASE, api_key="sk-or-test")
        )
        assert config.api_base == OPENROUTER_BASE
        assert config.api_key == "sk-or-test"

    def test_tts_endpoint_overrides_shared_one(self):
        from autiobook.utils import get_tts_config

        config = get_tts_config(
            self._args(
                api_base=LOCAL_BASE,
                api_key="sk-llm",
                tts_api_base=OPENROUTER_BASE,
                tts_api_key="sk-or-test",
            )
        )
        assert config.api_base == OPENROUTER_BASE
        assert config.api_key == "sk-or-test"

    def test_tts_endpoint_alone_selects_http_engine(self):
        from autiobook.tts_http import HTTPTTSConfig as Cfg
        from autiobook.utils import get_tts_config

        config = get_tts_config(self._args(tts_api_base=OPENROUTER_BASE))
        assert isinstance(config, Cfg)

    def test_voices_flag_parsed(self):
        from autiobook.utils import get_design_config

        config = get_design_config(
            self._args(api_base=OPENROUTER_BASE, tts_voices="Zephyr, Puck ,Kore")
        )
        assert config.voices == ["Zephyr", "Puck", "Kore"]

    def test_tts_model_overrides_every_mode(self):
        """one hosted model serves design and cloning; the qwen ids do not."""
        from autiobook.utils import get_clone_config, get_design_config, get_tts_config

        args = self._args(api_base=OPENROUTER_BASE, tts_model="google/gemini-tts")
        assert get_tts_config(args).model == "google/gemini-tts"
        assert get_design_config(args).model == "google/gemini-tts"
        assert get_clone_config(args).model == "google/gemini-tts"

    def test_per_mode_model_still_wins(self):
        from autiobook.utils import get_clone_config

        args = self._args(
            api_base=OPENROUTER_BASE,
            tts_model="google/gemini-tts",
            tts_clone_model="other/clone-tts",
        )
        assert get_clone_config(args).model == "other/clone-tts"


class TestPresetAuditionConfig:
    """dramatize's preset path must not rebuild the config from scratch."""

    def _design_config(self) -> HTTPTTSConfig:
        return HTTPTTSConfig(
            api_base=OPENROUTER_BASE,
            api_key="sk-or-test",
            model="google/gemini-3.1-flash-tts-preview",
            direction=TTS_DIRECTION_PREFIX,
            voices=["Zephyr"],
        )

    def test_hosted_config_keeps_auth_and_model(self):
        from autiobook.utils import preset_audition_config

        config = preset_audition_config(self._design_config())

        assert config.api_key == "sk-or-test"
        assert config.model == "google/gemini-3.1-flash-tts-preview"
        assert config.direction == TTS_DIRECTION_PREFIX
        assert config.voices == ["Zephyr"]

    def test_qwen_config_switches_to_the_instructable_model(self):
        from autiobook.config import DEFAULT_MODEL
        from autiobook.utils import preset_audition_config

        design = replace(
            self._design_config(), api_base=LOCAL_BASE, model="design-model"
        )
        assert preset_audition_config(design).model == DEFAULT_MODEL

    def test_local_config_falls_back_to_a_fresh_http_config(self):
        """a local engine config has no endpoint to inherit."""
        from autiobook.utils import preset_audition_config

        config = preset_audition_config(object(), LOCAL_BASE)
        assert isinstance(config, HTTPTTSConfig)
        assert config.api_base == LOCAL_BASE


GEMINI_MODEL = "gemini-3.8-flash-tts"


def _gemini_engine(**kwargs) -> HTTPTTSEngine:
    defaults = dict(
        api_base=GEMINI_API_BASE,
        api_key="g-test",
        model=GEMINI_MODEL,
        speaker="Kore",
    )
    return HTTPTTSEngine(HTTPTTSConfig(**{**defaults, **kwargs}))


def _interaction(audio: bytes) -> bytes:
    """an interactions api reply carrying one take, as the api frames it."""
    return json.dumps(
        {
            "id": "v1_test",
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {"type": "audio", "data": base64.b64encode(audio).decode()}
                    ],
                }
            ],
        }
    ).encode()


def _gemini_events(*chunks: bytes) -> bytes:
    """an interactions sse stream: audio deltas framed by step and turn events."""
    events = [
        ("interaction.created", {"interaction": {"status": "in_progress"}}),
        ("step.start", {"index": 0, "step": {"type": "model_output"}}),
        *(
            (
                "step.delta",
                {
                    "index": 0,
                    "delta": {"type": "audio", "data": base64.b64encode(c).decode()},
                },
            )
            for c in chunks
        ),
        ("step.stop", {"index": 0}),
        ("interaction.completed", {"interaction": {"status": "completed"}}),
    ]
    return "".join(
        f"event: {name}\ndata: {json.dumps({**data, 'event_type': name})}\n\n"
        for name, data in events
    ).encode()


class TestGeminiDialect:
    """google's interactions api: style metadata, key header, json audio."""

    def test_google_host_resolves_gemini(self):
        assert _gemini_engine().dialect == TTS_DIALECT_GEMINI

    def test_defaults_to_raw_pcm(self):
        assert _gemini_engine().response_format == TTS_FORMAT_PCM

    def test_body_shape(self):
        engine = _gemini_engine()
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(_interaction(_pcm_bytes())),
        ) as mock_url:
            engine.synthesize(
                "Hello there.", instruct="whispers softly", speaker="Puck"
            )

        (body,) = _bodies(mock_url)
        assert body == {
            "model": GEMINI_MODEL,
            "input": [
                {
                    "type": "user_input",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello there.",
                            "annotations": [
                                {"type": "speech_metadata", "style": "whispers softly"}
                            ],
                        }
                    ],
                }
            ],
            "response_format": {
                "type": "audio",
                "mime_type": "audio/l16",
                "sample_rate": SAMPLE_RATE,
            },
            "generation_config": {"speech_config": [{"voice": "Puck"}]},
            "store": False,
        }

    def test_no_direction_sends_no_style(self):
        engine = _gemini_engine()
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(_interaction(_pcm_bytes())),
        ) as mock_url:
            engine.synthesize("Hello there.")

        (content,) = _bodies(mock_url)[0]["input"][0]["content"]
        assert "annotations" not in content
        assert _bodies(mock_url)[0]["generation_config"]["speech_config"] == [
            {"voice": "Kore"}
        ]

    def test_prefix_direction_never_reaches_the_transcript(self):
        """3.8 speaks its input verbatim, so a prefixed direction would be read."""
        engine = _gemini_engine(direction=TTS_DIRECTION_PREFIX)
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(_interaction(_pcm_bytes())),
        ) as mock_url:
            engine.synthesize("Hello there.", instruct="whispers softly")

        (content,) = _bodies(mock_url)[0]["input"][0]["content"]
        assert content["text"] == "Hello there."
        assert content["annotations"][0]["style"] == "whispers softly"

    def test_key_header_not_bearer(self):
        engine = _gemini_engine()
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(_interaction(_pcm_bytes())),
        ) as mock_url:
            engine.synthesize("hello")

        req = mock_url.call_args[0][0]
        assert req.full_url == f"{GEMINI_API_BASE}/interactions"
        assert req.get_header("X-goog-api-key") == "g-test"
        assert req.get_header("Authorization") is None

    def test_pcm_reply_decoded(self):
        engine = _gemini_engine()
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(_interaction(_pcm_bytes(6))),
        ):
            audio, sr = engine.synthesize("hello")

        assert sr == SAMPLE_RATE
        np.testing.assert_allclose(audio, np.arange(6) / 32768.0)

    def test_wav_reply_decoded(self):
        engine = _gemini_engine(response_format=TTS_FORMAT_WAV)
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(_interaction(_wav_bytes(0.01))),
        ) as mock_url:
            audio, _ = engine.synthesize("hello")

        assert _bodies(mock_url)[0]["response_format"]["mime_type"] == "audio/wav"
        assert len(audio) == int(SAMPLE_RATE * 0.01)

    def test_last_audio_block_is_the_take(self):
        reply = json.loads(_interaction(_pcm_bytes(2)))
        reply["steps"].insert(0, {"type": "thought", "signature": "x"})
        reply["steps"].append(
            {
                "type": "model_output",
                "content": [
                    {"type": "audio", "data": base64.b64encode(_pcm_bytes(5)).decode()}
                ],
            }
        )
        engine = _gemini_engine()
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(json.dumps(reply).encode()),
        ):
            audio, _ = engine.synthesize("hello")
        assert len(audio) == 5

    def test_reply_without_audio_fails(self):
        engine = _gemini_engine()
        payload = json.dumps({"status": "failed", "steps": []}).encode()
        with patch("urllib.request.urlopen", return_value=_mock_response(payload)):
            with pytest.raises(RuntimeError, match="no audio.*failed"):
                engine.synthesize("hello")

    def test_mp3_is_refused_before_sending(self):
        engine = _gemini_engine(response_format=TTS_FORMAT_MP3)
        with patch("urllib.request.urlopen") as mock_url:
            with pytest.raises(RuntimeError, match="mp3"):
                engine.synthesize("hello")
        assert mock_url.call_count == 0

    def test_stream_ignores_an_unservable_synthesis_format(self):
        """streams always ask for pcm, so mp3 for synthesis must not block them."""
        engine = _gemini_engine(response_format=TTS_FORMAT_MP3)
        events = _gemini_events(_pcm_bytes(2))
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(events)
        ) as mock_url:
            audio, _ = engine.design_voice_stream("hello", "calm")
        assert len(audio) == 2
        assert _bodies(mock_url)[0]["response_format"]["mime_type"] == "audio/l16"

    def test_hosted_capabilities(self):
        engine = _gemini_engine()
        assert engine.seeded is False
        assert engine.streaming is True

    def test_known_voices_without_request(self):
        with patch("urllib.request.urlopen") as mock_url:
            voices = _gemini_engine().list_voices()
        assert "Kore" in voices and len(voices) == 30
        assert mock_url.call_count == 0

    def test_design_and_clone_point_at_preset_voices(self):
        engine = _gemini_engine()
        with pytest.raises(RuntimeError, match="--preset-voices"):
            engine.design_voice("hello", "a gruff old sailor")
        with pytest.raises(RuntimeError, match="--preset-voices"):
            engine.clone_voice("hello", np.zeros(10, dtype="float32"), "ref")

    def test_stream_parses_audio_deltas(self):
        engine = _gemini_engine(response_format=TTS_FORMAT_WAV)
        chunks: list[bytes] = []
        events = _gemini_events(_pcm_bytes(3), _pcm_bytes(4))
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(events)
        ) as mock_url:
            audio, sr = engine.design_voice_stream(
                "hello", "calm", on_chunk=chunks.append, voice="Puck"
            )

        assert chunks == [_pcm_bytes(3), _pcm_bytes(4)]
        assert len(audio) == 7 and sr == SAMPLE_RATE
        req = mock_url.call_args[0][0]
        assert req.full_url == f"{GEMINI_API_BASE}/interactions?alt=sse"
        assert req.get_header("Accept") == "text/event-stream"
        body = _bodies(mock_url)[0]
        assert body["stream"] is True
        # the player takes raw samples whatever format synthesis is set to
        assert body["response_format"]["mime_type"] == "audio/l16"
        assert body["generation_config"]["speech_config"] == [{"voice": "Puck"}]

    def test_stream_error_event_fails(self):
        engine = _gemini_engine()
        events = b'event: error\ndata: {"error": {"code": 400}}\n\n'
        with patch("urllib.request.urlopen", return_value=_mock_response(events)):
            with pytest.raises(RuntimeError, match="tts stream error"):
                engine.design_voice_stream("hello", "calm")

    def test_transient_failure_is_retried(self):
        engine = _gemini_engine()
        with (
            patch(
                "urllib.request.urlopen",
                side_effect=[
                    _dropped(),
                    _mock_response(_interaction(_pcm_bytes(2))),
                ],
            ) as mock_url,
            patch("time.sleep"),
        ):
            audio, _ = engine.synthesize("hello")
        assert mock_url.call_count == 2
        assert len(audio) == 2


class GeminiStub(BaseHTTPRequestHandler):
    """minimal interactions api: key header auth, json or sse audio replies."""

    requests: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        GeminiStub.requests.append(
            {
                "path": self.path,
                "key": self.headers.get("x-goog-api-key"),
                "body": body,
            }
        )
        if self.headers.get("x-goog-api-key") != "g-test":
            self.send_error(401, "API key not valid")
            return
        if body.get("stream"):
            payload = _gemini_events(_pcm_bytes(3), _pcm_bytes(5))
            content_type = "text/event-stream"
        else:
            payload = _interaction(_pcm_bytes(4))
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def gemini_stub():
    GeminiStub.requests = []
    server = HTTPServer(("127.0.0.1", 0), GeminiStub)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    yield f"http://{host}:{port}/v1beta"
    server.shutdown()
    server.server_close()


class TestAgainstGeminiStub:
    """full request/response cycle over a socket, no mocks."""

    def _engine(self, base: str, key: str = "g-test") -> HTTPTTSEngine:
        return _gemini_engine(api_base=base, api_key=key, dialect=TTS_DIALECT_GEMINI)

    def test_synthesis_round_trip(self, gemini_stub):
        audio, sr = self._engine(gemini_stub).synthesize("hello", instruct="calm")

        assert sr == SAMPLE_RATE
        assert len(audio) == 4
        sent = GeminiStub.requests[0]
        assert sent["path"] == "/v1beta/interactions"
        assert sent["key"] == "g-test"

    def test_stream_round_trip(self, gemini_stub):
        chunks: list[bytes] = []
        audio, _ = self._engine(gemini_stub).design_voice_stream(
            "hello", "calm", on_chunk=chunks.append
        )
        assert len(chunks) == 2
        assert len(audio) == 8
        assert GeminiStub.requests[0]["path"] == "/v1beta/interactions?alt=sse"

    def test_bad_key_fails_fast_with_the_reason(self, gemini_stub):
        with pytest.raises(RuntimeError, match="http 401"):
            self._engine(gemini_stub, key="wrong").synthesize("hello")
        assert len(GeminiStub.requests) == 1


class TestGeminiConfigFromArgs:
    """the gemini dialect needs neither an endpoint nor a tts-specific key."""

    def _args(self, **kwargs):
        defaults = dict(
            api_base=None,
            api_key=None,
            tts_api_base=None,
            tts_api_key=None,
            tts_dialect="auto",
            tts_direction="field",
            tts_voices="",
            tts_model=GEMINI_MODEL,
        )
        return argparse.Namespace(**{**defaults, **kwargs})

    def test_dialect_alone_selects_google(self, monkeypatch):
        from autiobook.utils import get_tts_config

        monkeypatch.setenv("GEMINI_API_KEY", "g-env")
        config = get_tts_config(self._args(tts_dialect=TTS_DIALECT_GEMINI))

        assert isinstance(config, HTTPTTSConfig)
        assert config.api_base == GEMINI_API_BASE
        assert config.api_key == "g-env"
        assert config.model == GEMINI_MODEL

    def test_google_key_preferred_over_the_shared_one(self, monkeypatch):
        """the shared key usually belongs to the llm's provider."""
        from autiobook.utils import get_tts_config

        monkeypatch.setenv("GEMINI_API_KEY", "g-env")
        config = get_tts_config(
            self._args(
                api_base=OPENROUTER_BASE,
                api_key="sk-or-llm",
                tts_api_base=GEMINI_API_BASE,
            )
        )
        assert config.api_key == "g-env"

    def test_tts_key_wins(self, monkeypatch):
        from autiobook.utils import get_tts_config

        monkeypatch.setenv("GEMINI_API_KEY", "g-env")
        config = get_tts_config(
            self._args(tts_dialect=TTS_DIALECT_GEMINI, tts_api_key="g-flag")
        )
        assert config.api_key == "g-flag"

    def test_shared_key_used_without_a_google_one(self, monkeypatch):
        from autiobook.utils import get_tts_config

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        config = get_tts_config(
            self._args(tts_dialect=TTS_DIALECT_GEMINI, api_key="g-shared")
        )
        assert config.api_key == "g-shared"

    def test_other_dialects_ignore_the_google_key(self, monkeypatch):
        from autiobook.utils import get_tts_config

        monkeypatch.setenv("GEMINI_API_KEY", "g-env")
        config = get_tts_config(
            self._args(api_base=OPENROUTER_BASE, api_key="sk-or-test")
        )
        assert config.api_key == "sk-or-test"


def _gemini_http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        f"{GEMINI_API_BASE}/voices", code, "err", {}, io.BytesIO(b"{}")
    )


class TestGeminiVoiceApi:
    """stored custom voices: design, existence, deletion, quota counting."""

    def _created(self, **extra) -> bytes:
        return json.dumps(
            {
                "id": "voice_abc",
                "type": "prompted",
                "sample_audio": {
                    "mime_type": "audio/wav",
                    "data": base64.b64encode(b"RIFF").decode(),
                },
                **extra,
            }
        ).encode()

    def test_design_request(self):
        engine = _gemini_engine()
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(self._created())
        ) as mock_url:
            voice_id, sample = engine.create_designed_voice(
                "A gruff sailor.", "mistborn: Clubs", gender="male"
            )

        assert voice_id == "voice_abc"
        assert sample == b"RIFF"
        req = mock_url.call_args[0][0]
        assert req.full_url == f"{GEMINI_API_BASE}/voices"
        assert req.get_header("X-goog-api-key") == "g-test"
        assert _bodies(mock_url)[0] == {
            "store": True,
            "voice": {
                "model": GEMINI_MODEL,
                "type": "prompted",
                "display_name": "mistborn: Clubs",
                "prompted": {"input": "A gruff sailor."},
                "gender": "male",
            },
        }

    def test_design_accepts_a_resource_name(self):
        reply = json.dumps({"name": "voices/voice_xyz"}).encode()
        with patch("urllib.request.urlopen", return_value=_mock_response(reply)):
            voice_id, sample = _gemini_engine().create_designed_voice("p", "n")
        assert voice_id == "voice_xyz"
        assert sample is None

    def test_design_without_an_id_fails(self):
        with patch("urllib.request.urlopen", return_value=_mock_response(b"{}")):
            with pytest.raises(RuntimeError, match="no voice id"):
                _gemini_engine().create_designed_voice("p", "n")

    def test_dropped_design_is_not_resent(self):
        """the voice may exist already; a resend would spend quota on a twin."""
        with (
            patch("urllib.request.urlopen", side_effect=_dropped()) as mock_url,
            patch("time.sleep"),
        ):
            with pytest.raises(http.client.RemoteDisconnected):
                _gemini_engine().create_designed_voice("p", "n")
        assert mock_url.call_count == 1

    def test_refused_design_is_retried(self):
        with (
            patch(
                "urllib.request.urlopen",
                side_effect=[_gemini_http_error(503), _mock_response(self._created())],
            ) as mock_url,
            patch("time.sleep"),
        ):
            voice_id, _ = _gemini_engine().create_designed_voice("p", "n")
        assert voice_id == "voice_abc"
        assert mock_url.call_count == 2

    def test_voice_exists(self):
        engine = _gemini_engine()
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(b'{"id": "voice_a"}')
        ) as mock_url:
            assert engine.voice_exists("voice_a") is True
        assert mock_url.call_args[0][0].full_url == f"{GEMINI_API_BASE}/voices/voice_a"

    def test_expired_voice_does_not_exist(self):
        with patch("urllib.request.urlopen", side_effect=_gemini_http_error(404)):
            assert _gemini_engine().voice_exists("voice_a") is False

    def test_other_failures_are_not_taken_as_expiry(self):
        with patch("urllib.request.urlopen", side_effect=_gemini_http_error(403)):
            with pytest.raises(RuntimeError, match="http 403"):
                _gemini_engine().voice_exists("voice_a")

    def test_delete(self):
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(b"{}")
        ) as mock_url:
            _gemini_engine().delete_voice("voice_a")
        req = mock_url.call_args[0][0]
        assert req.get_method() == "DELETE"
        assert req.full_url == f"{GEMINI_API_BASE}/voices/voice_a"

    def test_deleting_a_gone_voice_is_fine(self):
        with patch("urllib.request.urlopen", side_effect=_gemini_http_error(404)):
            _gemini_engine().delete_voice("voice_a")

    def test_count_pages_through_stored_voices(self):
        pages = [
            {
                "voices": [{"id": "voice_1", "type": "prompted"}] * 3,
                "next_page_token": "t2",
            },
            {"voices": [{"id": "voice_2", "type": "replicated"}]},
        ]
        with patch(
            "urllib.request.urlopen",
            side_effect=[_mock_response(json.dumps(p).encode()) for p in pages],
        ) as mock_url:
            assert _gemini_engine().count_stored_voices() == 4

        first, second = (c[0][0].full_url for c in mock_url.call_args_list)
        assert "type=prompted" in first and "type=replicated" in first
        assert "page_token=t2" in second

    def test_prebuilt_voices_are_not_counted(self):
        reply = {"voices": [{"id": "Kore", "type": "prebuilt"}]}
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(json.dumps(reply).encode()),
        ):
            assert _gemini_engine().count_stored_voices() == 0

    def test_other_backends_refuse(self):
        engine = _engine(api_base=OPENROUTER_BASE)
        with pytest.raises(RuntimeError, match="--tts-dialect gemini"):
            engine.create_designed_voice("p", "n")
        with pytest.raises(RuntimeError, match="--tts-dialect gemini"):
            engine.voice_exists("voice_a")
