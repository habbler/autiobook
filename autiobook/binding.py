"""voice binding for preset-voice audition: series voices and designed ones.

before a book's characters are cast from the backend's preset list, two
sources may already decide a voice:
  - the series: every series character speaks with the voice it was bound to
    in an earlier book. a designed voice google has since expired is designed
    again from its recorded prompt.
  - gemini voice design (--design-voices): a stored custom voice per named
    character, from its voice prompt. walk-ons and the stand-in extras keep
    preset voices, since a google project holds GEMINI_VOICE_QUOTA at most.
"""

from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, Callable

from .config import (
    BINDING_DESIGNED,
    BINDING_PRESET,
    DESIGN_MIN_MENTIONS,
    EXTRA_CAST_NAMES,
    GEMINI_VOICE_QUOTA,
    NARRATOR_NAME,
    WAV_EXT,
)
from .llm import Character
from .resume import get_command_dir
from .series import Series, VoiceBinding, book_text, mention_count
from .utils import prompt_choice

# where a standalone book keeps designed voices' preview clips
DESIGNED_DIR = "designed"


def _save_sample(
    workdir: Path,
    series: Series | None,
    name: str,
    voice_id: str,
    sample: bytes | None,
) -> str:
    """keep google's preview clip of a designed voice; returns its path,
    relative to the series file when there is one."""
    if not sample:
        return ""
    if series is not None:
        folder = series.voices_dir
    else:
        folder = get_command_dir(workdir, "audition") / DESIGNED_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}__{voice_id}{WAV_EXT}"
    path.write_bytes(sample)
    if series is not None:
        return path.relative_to(series.path.parent).as_posix()
    return str(path)


def _design(
    engine: Any,
    workdir: Path,
    series: Series | None,
    char: Character,
    prompt: str,
    gender: str = "",
    language_code: str = "",
) -> VoiceBinding:
    """create one stored voice and the binding that records how."""
    label = series.path.stem if series is not None else workdir.name
    voice_id, sample = engine.create_designed_voice(
        prompt,
        f"{label}: {char.name}",
        gender=gender,
        language_code=language_code,
    )
    today = date.today().isoformat()
    return VoiceBinding(
        voice_id=voice_id,
        kind=BINDING_DESIGNED,
        model=engine.config.model,
        design_prompt=prompt,
        sample=_save_sample(workdir, series, char.name, voice_id, sample),
        created=today,
        last_used=today,
    )


def bind_series_voices(
    workdir: Path, cast: list[Character], engine: Any, series: Series
) -> tuple[dict[str, str], dict[str, VoiceBinding]]:
    """voices the series already decides for this book's cast.

    returns (character -> voice id, replacement bindings). a preset the
    backend does not offer is left unbound with a warning rather than
    guessed at; a designed voice google has expired is designed again from
    its recorded prompt, which gets close to the original but not the same.
    """
    voices: dict[str, str] = {}
    rebuilt: dict[str, VoiceBinding] = {}
    presets: set[str] | None = None
    for char in cast:
        entry = series.match(char)
        if entry is None or entry.binding is None:
            continue
        binding = entry.binding
        if binding.kind == BINDING_PRESET:
            if presets is None:
                presets = set(engine.list_voices())
            if binding.voice_id not in presets:
                print(
                    f"series: {char.name}'s voice '{binding.voice_id}' is not "
                    "offered by this backend; leaving them uncast for this book"
                )
                continue
            voices[char.name] = binding.voice_id
            continue

        if not hasattr(engine, "voice_exists"):
            raise RuntimeError(
                f"series binds {char.name} to designed voice {binding.voice_id}, "
                "which needs gemini tts (--tts-dialect gemini)"
            )
        if engine.voice_exists(binding.voice_id):
            voices[char.name] = binding.voice_id
            continue
        if not binding.design_prompt:
            print(
                f"series: {char.name}'s designed voice {binding.voice_id} is "
                "gone and has no recorded prompt to rebuild it from"
            )
            continue
        print(
            f"series: {char.name}'s designed voice {binding.voice_id} has "
            "expired; designing it again from its recorded prompt. it will be "
            "close to the original, not identical"
        )
        fresh = _design(
            engine,
            workdir,
            series,
            char,
            binding.design_prompt,
            gender=entry.gender,
            language_code=entry.language_code,
        )
        rebuilt[char.name] = replace(fresh, first_book=binding.first_book)
        voices[char.name] = fresh.voice_id
    return voices, rebuilt


def design_candidates(
    workdir: Path, cast: list[Character], decided: dict[str, str]
) -> list[Character]:
    """characters worth a designed voice: named, and named often.

    the narrator always qualifies; the extras never do, since they voice
    whoever the script could not name. everyone else needs at least
    DESIGN_MIN_MENTIONS mentions in the book's text.
    """
    text = book_text(workdir)
    picked = []
    for char in cast:
        if decided.get(char.name) or char.name in EXTRA_CAST_NAMES:
            continue
        if char.name != NARRATOR_NAME:
            mentions = mention_count(text, char)
            if mentions < DESIGN_MIN_MENTIONS:
                print(
                    f"design: {char.name} is named {mentions} time(s), under "
                    f"{DESIGN_MIN_MENTIONS}; keeping a preset voice"
                )
                continue
        picked.append(char)
    return picked


def _design_inputs(series: Series | None, char: Character) -> tuple[str, str]:
    entry = series.match(char) if series is not None else None
    return (entry.gender, entry.language_code) if entry else ("", "")


def design_voices(
    workdir: Path,
    cast: list[Character],
    engine: Any,
    series: Series | None,
    decided: dict[str, str],
    directed: bool = False,
    on_design: Callable[[str, VoiceBinding], None] | None = None,
) -> tuple[dict[str, VoiceBinding], bool]:
    """design a stored voice for each character that warrants one.

    returns (character -> binding, quit requested). stops early, leaving
    the rest to preset casting, once the project's voice quota is spent.
    `on_design` is told of each kept voice as soon as it exists.
    """
    candidates = design_candidates(workdir, cast, decided)
    if not candidates:
        return {}, False
    stored = engine.count_stored_voices()
    remaining = GEMINI_VOICE_QUOTA - stored
    print(
        f"design: {len(candidates)} character(s) to design; the project holds "
        f"{stored}/{GEMINI_VOICE_QUOTA} stored voices"
    )
    designed: dict[str, VoiceBinding] = {}
    for char in candidates:
        if remaining <= 0:
            print(
                f"design: voice quota reached ({GEMINI_VOICE_QUOTA}); the rest "
                "get preset voices. delete unused voices to free slots"
            )
            break
        gender, language_code = _design_inputs(series, char)
        if not directed:
            print(f"design: {char.name}...")
            designed[char.name] = _design(
                engine,
                workdir,
                series,
                char,
                char.voice_prompt(),
                gender=gender,
                language_code=language_code,
            )
            if on_design is not None:
                on_design(char.name, designed[char.name])
            remaining -= 1
            continue
        binding, spent, quit_requested = _direct_design(
            engine, workdir, series, char, gender, language_code, remaining
        )
        remaining -= spent
        if binding is not None:
            designed[char.name] = binding
            if on_design is not None:
                on_design(char.name, binding)
        if quit_requested:
            return designed, True
    return designed, False


def _direct_design(
    engine: Any,
    workdir: Path,
    series: Series | None,
    char: Character,
    gender: str,
    language_code: str,
    remaining: int,
) -> tuple[VoiceBinding | None, int, bool]:
    """design voices for one character until the user keeps one.

    returns (kept binding or None, quota slots still occupied, quit). every
    rejected design is deleted at once: left stored, each would hold a
    quota slot for a year.
    """
    from .audition import _edit_description
    from .casting import _preview_path, _start_preview, _stop_playback

    print(f"\n=== {char.name} (voice design) ===")
    print(f"  {char.description}")
    prompt = char.voice_prompt()
    while remaining > 0:
        print(f"  designing from: {prompt!r}")
        binding = _design(engine, workdir, series, char, prompt, gender, language_code)
        preview = _preview_path(workdir, char.name, binding.voice_id)
        try:
            playback = _start_preview(engine, char, binding.voice_id, preview)
        except Exception as e:
            print(f"    failed to play a take: {e}")
            playback = None
        while True:
            ans = prompt_choice(
                "  [y]es / [n]ew design / [r]eplay / [e]dit prompt / "
                "[p]reset voices / [q]uit: "
            )
            _stop_playback(playback)
            if ans in ("y", "yes"):
                return binding, 1, False
            if ans in ("r", "replay"):
                from .casting import _play_wav_async

                if preview.exists():
                    playback = _play_wav_async(preview)
                else:
                    print("    still rendering; replay again in a moment")
                continue
            if ans in ("n", "new", "e", "edit", "p", "preset", "q", "quit", ""):
                break
            print("    unknown input")
        engine.delete_voice(binding.voice_id)
        if ans in ("p", "preset"):
            return None, 0, False
        if ans in ("q", "quit"):
            return None, 0, True
        if ans in ("e", "edit"):
            prompt = _edit_description(prompt).strip() or prompt
    print("  voice quota reached; falling back to preset voices")
    return None, 0, False
