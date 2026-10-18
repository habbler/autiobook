"""tests for series casts: shared characters and voice bindings across books."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autiobook.config import (
    BINDING_DESIGNED,
    BINDING_PRESET,
    DESIGN_MIN_MENTIONS,
    GEMINI_VOICE_QUOTA,
    SERIES_LINK_FILE,
)
from autiobook.dramatize import (
    _apply_cast_merge,
    _merge_character_into_cast,
    load_cast,
    run_cast_generation,
    save_cast,
)
from autiobook.llm import CastMerge, Character
from autiobook.series import (
    Series,
    SeriesEntry,
    VoiceBinding,
    lock_cast,
    mention_count,
    record_book,
    resolve_series,
    seed_cast,
)

KELSIER = Character(
    name="Kelsier",
    description="A charismatic thief.",
    voice="Male, thirties, warm tenor, quick.",
    aliases=["Kell"],
    audition_line="Smile, they hate that.",
)


def _char(name: str, **kw) -> Character:
    return Character(name=name, description=kw.pop("description", "d"), **kw)


def _series(tmp_path: Path, *entries: SeriesEntry) -> Series:
    return Series(path=tmp_path / "mistborn.json", entries=list(entries))


def _kelsier_entry(**binding) -> SeriesEntry:
    from copy import deepcopy

    return SeriesEntry(
        character=deepcopy(KELSIER),
        binding=VoiceBinding(**binding) if binding else None,
    )


def _workdir(tmp_path: Path, text: str, series: Series | None = None) -> Path:
    """a book workdir with one extracted chapter, optionally series-linked."""
    workdir = tmp_path / "book2"
    (workdir / "extract").mkdir(parents=True)
    (workdir / "extract" / "01_Chapter.txt").write_text(text, encoding="utf-8")
    (workdir / "extract" / "metadata.json").write_text(
        json.dumps({"title": "The Well of Ascension"})
    )
    if series is not None:
        series.save()
        (workdir / SERIES_LINK_FILE).write_text(
            json.dumps({"series": str(series.path)})
        )
    return workdir


class TestSeriesFile:
    def test_missing_file_is_an_empty_series(self, tmp_path):
        series = Series.load(tmp_path / "new.json")
        assert series.entries == []

    def test_round_trip(self, tmp_path):
        entry = _kelsier_entry(
            voice_id="voice_abc",
            kind=BINDING_DESIGNED,
            design_prompt="Male, thirties, warm tenor, quick.",
        )
        entry.gender = "male"
        _series(tmp_path, entry).save()

        loaded = Series.load(tmp_path / "mistborn.json")
        (got,) = loaded.entries
        assert got.character == KELSIER
        assert got.binding == entry.binding
        assert got.gender == "male"

    def test_save_leaves_no_temp_file(self, tmp_path):
        _series(tmp_path, _kelsier_entry()).save()
        assert [p.name for p in tmp_path.iterdir()] == ["mistborn.json"]

    def test_empty_binding_fields_are_not_written(self, tmp_path):
        _series(tmp_path, _kelsier_entry(voice_id="Puck")).save()
        raw = json.loads((tmp_path / "mistborn.json").read_text())
        assert raw["characters"][0]["binding"] == {
            "voice_id": "Puck",
            "kind": BINDING_PRESET,
        }

    @pytest.mark.parametrize(
        "char",
        [
            _char("kelsier"),
            _char("Kell"),
            _char("The Survivor", aliases=["Kelsier"]),
        ],
        ids=["name-any-case", "alias", "own-alias"],
    )
    def test_match(self, tmp_path, char):
        assert _series(tmp_path, _kelsier_entry()).match(char) is not None

    def test_no_match(self, tmp_path):
        assert _series(tmp_path, _kelsier_entry()).match(_char("Vin")) is None

    def test_used_voices(self, tmp_path):
        series = _series(tmp_path, _kelsier_entry(voice_id="Puck"))
        series.entries.append(SeriesEntry(character=_char("Vin")))
        assert series.used_voices() == {"Puck"}


class TestMentions:
    def test_name_and_aliases_count(self):
        text = "Kelsier smiled. Kell laughed. Kelsier's plan."
        assert mention_count(text, KELSIER) == 3

    def test_whole_words_only(self):
        assert mention_count("Kellogg ate cereal.", KELSIER) == 0

    def test_case_sensitive(self):
        """a name that is also a word would otherwise count every use of it."""
        assert mention_count("i will go. Will went.", _char("Will")) == 1

    def test_pronoun_aliases_never_count(self):
        char = _char("Vin", aliases=["she"])
        assert mention_count("she ran. she hid.", char) == 0


class TestLockCast:
    def test_series_form_wins(self, tmp_path):
        book = _char("Kelsier", description="A dead man.", voice="Raspy.")
        (locked,), names, notes = lock_cast([book], _series(tmp_path, _kelsier_entry()))
        assert locked.description == KELSIER.description
        assert locked.voice == KELSIER.voice
        assert locked.audition_line == KELSIER.audition_line
        assert names == {"kelsier"}
        assert notes

    def test_alias_match_is_renamed(self, tmp_path):
        (locked,), _, notes = lock_cast(
            [_char("Kell")], _series(tmp_path, _kelsier_entry())
        )
        assert locked.name == "Kelsier"
        assert "'Kell' is series character 'Kelsier'" in notes

    def test_two_book_characters_fold_into_one(self, tmp_path):
        cast = [_char("Kelsier"), _char("Vin"), _char("The Survivor", aliases=["Kell"])]
        out, _, _ = lock_cast(cast, _series(tmp_path, _kelsier_entry()))
        assert [c.name for c in out] == ["Kelsier", "Vin"]
        assert "The Survivor" in out[0].aliases

    def test_book_aliases_are_learned(self, tmp_path):
        book = _char("Kelsier", aliases=["Master Thief", "he"])
        (locked,), _, _ = lock_cast([book], _series(tmp_path, _kelsier_entry()))
        assert locked.aliases == ["Kell", "Master Thief"]

    def test_series_entry_is_not_mutated(self, tmp_path):
        series = _series(tmp_path, _kelsier_entry())
        lock_cast([_char("Kelsier", aliases=["Master Thief"])], series)
        assert series.entries[0].character.aliases == ["Kell"]

    def test_unknown_characters_pass_through(self, tmp_path):
        vin = _char("Vin")
        out, names, _ = lock_cast([vin], _series(tmp_path, _kelsier_entry()))
        assert out == [vin] and names == set()


class TestSeedCast:
    def test_named_twice_joins(self, tmp_path):
        cast, names, notes = seed_cast(
            [_char("Narrator")],
            _series(tmp_path, _kelsier_entry()),
            "Kell grinned. Kelsier ran.",
        )
        assert [c.name for c in cast] == ["Narrator", "Kelsier"]
        assert "kelsier" in names
        assert "'Kelsier' joins from the series" in notes

    def test_named_once_does_not(self, tmp_path):
        cast, _, _ = seed_cast(
            [_char("Narrator")], _series(tmp_path, _kelsier_entry()), "Kelsier."
        )
        assert [c.name for c in cast] == ["Narrator"]

    def test_already_present_is_not_duplicated(self, tmp_path):
        cast, _, _ = seed_cast(
            [_char("Kelsier")],
            _series(tmp_path, _kelsier_entry()),
            "Kelsier. Kelsier.",
        )
        assert len(cast) == 1


class TestRecordBook:
    def test_new_characters_join_with_their_voice(self, tmp_path):
        series = _series(tmp_path)
        added = record_book(series, [_char("Vin")], {"Vin": "Kore"}, {}, "Book 1")
        assert added == 1
        binding = series.entries[0].binding
        assert binding.voice_id == "Kore"
        assert binding.kind == BINDING_PRESET
        assert binding.first_book == "Book 1"
        assert binding.last_used

    def test_designed_ids_are_recognised(self, tmp_path):
        series = _series(tmp_path)
        record_book(series, [_char("Vin")], {"Vin": "voice_x"}, {}, "Book 1")
        assert series.entries[0].binding.kind == BINDING_DESIGNED

    def test_existing_binding_is_kept(self, tmp_path):
        series = _series(tmp_path, _kelsier_entry(voice_id="Puck", first_book="B1"))
        record_book(series, [_char("Kelsier")], {"Kelsier": "Puck"}, {}, "Book 2")
        assert series.entries[0].binding.first_book == "B1"

    def test_rebuilt_binding_replaces(self, tmp_path):
        series = _series(tmp_path, _kelsier_entry(voice_id="voice_old"))
        fresh = VoiceBinding(voice_id="voice_new", kind=BINDING_DESIGNED)
        record_book(
            series,
            [_char("Kelsier")],
            {"Kelsier": "voice_new"},
            {"Kelsier": fresh},
            "Book 2",
        )
        assert series.entries[0].binding.voice_id == "voice_new"

    def test_uncast_characters_still_join(self, tmp_path):
        series = _series(tmp_path)
        record_book(series, [_char("Vin")], {}, {}, "Book 1")
        assert series.entries[0].binding is None

    def test_aliases_carry_to_the_next_book(self, tmp_path):
        series = _series(tmp_path, _kelsier_entry())
        record_book(series, [_char("Kelsier", aliases=["Master Thief"])], {}, {}, "B2")
        assert "Master Thief" in series.entries[0].character.aliases


class TestResolveSeries:
    def test_flag_is_recorded(self, tmp_path):
        workdir = tmp_path / "book"
        path = resolve_series(workdir, str(tmp_path / "s.json"))
        assert path == (tmp_path / "s.json").resolve()
        assert resolve_series(workdir, None) == path

    def test_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTIOBOOK_SERIES", str(tmp_path / "env.json"))
        assert resolve_series(tmp_path / "b", None) == (tmp_path / "env.json").resolve()

    def test_standalone_book(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AUTIOBOOK_SERIES", raising=False)
        assert resolve_series(tmp_path / "b", None) is None

    def test_moving_a_book_says_so(self, tmp_path, capsys):
        workdir = tmp_path / "book"
        resolve_series(workdir, str(tmp_path / "a.json"))
        resolve_series(workdir, str(tmp_path / "b.json"))
        assert "moving this book" in capsys.readouterr().out


class TestCastMergeLocks:
    def _maps(self, char):
        return {char.name.lower(): char}, {
            a.lower(): char.name.lower() for a in char.aliases or []
        }

    def test_locked_fields_survive_the_llm(self):
        from copy import deepcopy

        existing = deepcopy(KELSIER)
        cast_map, alias_map = self._maps(existing)
        incoming = _char(
            "Kelsier", description="A ghost.", voice="Hollow.", aliases=["Survivor"]
        )
        _merge_character_into_cast(incoming, cast_map, alias_map, locked={"kelsier"})
        assert existing.description == KELSIER.description
        assert existing.voice == KELSIER.voice
        assert "Survivor" in existing.aliases

    def test_series_character_is_never_folded_away(self):
        from copy import deepcopy

        kelsier, vin = deepcopy(KELSIER), _char("Vin")
        cast_map = {"kelsier": kelsier, "vin": vin}
        merge = CastMerge(into="Vin", from_=["Kelsier"])
        assert _apply_cast_merge(merge, cast_map, {}, locked={"kelsier"}) is None
        assert "kelsier" in cast_map

    def test_series_character_may_absorb_others(self):
        from copy import deepcopy

        kelsier = deepcopy(KELSIER)
        cast_map = {"kelsier": kelsier, "survivor": _char("Survivor")}
        merge = CastMerge(into="Kelsier", from_=["Survivor"])
        assert _apply_cast_merge(merge, cast_map, {}, locked={"kelsier"})
        assert "survivor" not in cast_map


class TestCastPhase:
    def test_seeds_and_locks(self, tmp_path):
        series = _series(tmp_path, _kelsier_entry())
        workdir = _workdir(tmp_path, "Kelsier laughed. Kell ran. Vin hid.", series)
        reply = [
            _char("Kelsier", description="Rewritten.", voice="Rewritten."),
            _char("Vin", description="A street urchin."),
        ]
        with patch("autiobook.dramatize.generate_cast", return_value=(reply, [])):
            cast = run_cast_generation(workdir, model="m")

        by_name = {c.name: c for c in cast}
        assert by_name["Kelsier"].voice == KELSIER.voice
        assert "Vin" in by_name
        saved = {c.name: c for c in load_cast(workdir)}
        assert saved["Kelsier"].description == KELSIER.description

    def test_llm_sees_series_characters_as_known(self, tmp_path):
        series = _series(tmp_path, _kelsier_entry())
        workdir = _workdir(tmp_path, "Kelsier laughed. Kelsier ran.", series)
        with patch(
            "autiobook.dramatize.generate_cast", return_value=([], [])
        ) as mock_llm:
            run_cast_generation(workdir, model="m")
        assert "Kelsier" in mock_llm.call_args.kwargs["existing_cast_summary"]

    def test_late_alias_match_is_locked(self, tmp_path):
        """a series character the llm names by alias, missed by seeding."""
        series = _series(tmp_path, _kelsier_entry())
        workdir = _workdir(tmp_path, "The Survivor waited.", series)
        reply = [_char("The Survivor", aliases=["Kell"], description="x")]
        with patch("autiobook.dramatize.generate_cast", return_value=(reply, [])):
            cast = run_cast_generation(workdir, model="m")
        assert "Kelsier" in [c.name for c in cast]
        assert "The Survivor" not in [c.name for c in cast]

    def test_up_to_date_cast_still_locks(self, tmp_path):
        """linking a series to a finished book must still apply its locks."""
        workdir = _workdir(tmp_path, "Kelsier. Kelsier.")
        with patch(
            "autiobook.dramatize.generate_cast",
            return_value=([_char("Kelsier", voice="Book voice.")], []),
        ):
            run_cast_generation(workdir, model="m")

        series = _series(tmp_path, _kelsier_entry())
        series.save()
        (workdir / SERIES_LINK_FILE).write_text(
            json.dumps({"series": str(series.path)})
        )
        with patch("autiobook.dramatize.generate_cast") as mock_llm:
            run_cast_generation(workdir, model="m")
        mock_llm.assert_not_called()
        saved = {c.name: c for c in load_cast(workdir)}
        assert saved["Kelsier"].voice == KELSIER.voice

    def test_standalone_book_is_untouched(self, tmp_path):
        workdir = _workdir(tmp_path, "Kelsier. Kelsier.")
        reply = [_char("Kelsier", voice="Book voice.")]
        with patch("autiobook.dramatize.generate_cast", return_value=(reply, [])):
            cast = run_cast_generation(workdir, model="m")
        assert {c.name: c for c in cast}["Kelsier"].voice == "Book voice."


class FakeEngine:
    """the subset of HTTPTTSEngine that voice binding drives."""

    def __init__(self, presets=("Zephyr", "Puck", "Kore", "Leda"), stored=0):
        self.presets = list(presets)
        self.stored = stored
        self.live: set[str] = set()
        self.deleted: list[str] = []
        self.designs: list[str] = []
        self.config = MagicMock(model="gemini-3.8-flash-tts", speaker="Kore")
        self.streaming = False

    def list_voices(self):
        return list(self.presets)

    def create_designed_voice(self, prompt, display_name, gender="", language_code=""):
        self.designs.append(prompt)
        voice_id = f"voice_{len(self.designs)}"
        self.live.add(voice_id)
        return voice_id, b"RIFFsample"

    def voice_exists(self, voice_id):
        return voice_id in self.live

    def delete_voice(self, voice_id):
        self.live.discard(voice_id)
        self.deleted.append(voice_id)

    def count_stored_voices(self):
        return self.stored + len(self.live)

    def synthesize(self, text, instruct="", speaker=None):
        import numpy as np

        return np.zeros(240, dtype="float32"), 24000


class TestBindSeriesVoices:
    def _bind(self, tmp_path, engine, *entries, cast=None):
        from autiobook.binding import bind_series_voices

        series = _series(tmp_path, *entries)
        cast = cast or [_char("Kelsier")]
        return bind_series_voices(tmp_path, cast, engine, series)

    def test_preset_binding_holds(self, tmp_path):
        voices, rebuilt = self._bind(
            tmp_path, FakeEngine(), _kelsier_entry(voice_id="Puck")
        )
        assert voices == {"Kelsier": "Puck"} and rebuilt == {}

    def test_preset_the_backend_lacks_is_left_uncast(self, tmp_path, capsys):
        voices, _ = self._bind(tmp_path, FakeEngine(), _kelsier_entry(voice_id="ryan"))
        assert voices == {}
        assert "not offered by this backend" in capsys.readouterr().out

    def test_live_designed_voice_holds(self, tmp_path):
        engine = FakeEngine()
        engine.live.add("voice_old")
        voices, rebuilt = self._bind(
            tmp_path,
            engine,
            _kelsier_entry(voice_id="voice_old", kind=BINDING_DESIGNED),
        )
        assert voices == {"Kelsier": "voice_old"} and rebuilt == {}
        assert engine.designs == []

    def test_expired_designed_voice_is_rebuilt_from_its_prompt(self, tmp_path):
        engine = FakeEngine()
        voices, rebuilt = self._bind(
            tmp_path,
            engine,
            _kelsier_entry(
                voice_id="voice_old",
                kind=BINDING_DESIGNED,
                design_prompt="Warm tenor.",
                first_book="Book 1",
            ),
        )
        assert engine.designs == ["Warm tenor."]
        assert voices == {"Kelsier": "voice_1"}
        assert rebuilt["Kelsier"].first_book == "Book 1"
        assert rebuilt["Kelsier"].design_prompt == "Warm tenor."

    def test_expired_voice_without_a_prompt_is_reported(self, tmp_path, capsys):
        voices, _ = self._bind(
            tmp_path,
            FakeEngine(),
            _kelsier_entry(voice_id="voice_old", kind=BINDING_DESIGNED),
        )
        assert voices == {}
        assert "no recorded prompt" in capsys.readouterr().out

    def test_designed_voice_needs_gemini(self, tmp_path):
        engine = MagicMock(spec=["list_voices"])
        with pytest.raises(RuntimeError, match="needs gemini"):
            self._bind(
                tmp_path,
                engine,
                _kelsier_entry(voice_id="voice_old", kind=BINDING_DESIGNED),
            )


class TestDesignVoices:
    def _workdir(self, tmp_path, kelsier_mentions):
        return _workdir(tmp_path, " ".join(["Kelsier."] * kelsier_mentions))

    def test_candidates(self, tmp_path):
        from autiobook.binding import design_candidates

        workdir = self._workdir(tmp_path, DESIGN_MIN_MENTIONS)
        cast = [
            _char("Narrator"),
            _char("Extra Male"),
            _char("Kelsier"),
            _char("Clubs"),
        ]
        picked = design_candidates(workdir, cast, {})
        assert [c.name for c in picked] == ["Narrator", "Kelsier"]

    def test_decided_characters_are_skipped(self, tmp_path):
        from autiobook.binding import design_candidates

        workdir = self._workdir(tmp_path, DESIGN_MIN_MENTIONS)
        picked = design_candidates(
            workdir, [_char("Narrator"), _char("Kelsier")], {"Kelsier": "Puck"}
        )
        assert [c.name for c in picked] == ["Narrator"]

    def test_designs_each_from_its_voice_prompt(self, tmp_path):
        from autiobook.binding import design_voices

        workdir = self._workdir(tmp_path, DESIGN_MIN_MENTIONS)
        engine = FakeEngine()
        cast = [_char("Narrator", voice="Deep."), _char("Kelsier", voice="Tenor.")]
        designed, quit_requested = design_voices(workdir, cast, engine, None, {})
        assert engine.designs == ["Deep.", "Tenor."]
        assert designed["Kelsier"].kind == BINDING_DESIGNED
        assert designed["Kelsier"].design_prompt == "Tenor."
        assert Path(designed["Kelsier"].sample).read_bytes() == b"RIFFsample"
        assert not quit_requested

    def test_sample_is_kept_beside_the_series(self, tmp_path):
        from autiobook.binding import design_voices

        workdir = self._workdir(tmp_path, 0)
        series = _series(tmp_path)
        designed, _ = design_voices(
            workdir, [_char("Narrator")], FakeEngine(), series, {}
        )
        sample = designed["Narrator"].sample
        assert sample.startswith("mistborn_voices/")
        assert (tmp_path / sample).exists()

    def test_stops_at_the_quota(self, tmp_path, capsys):
        from autiobook.binding import design_voices

        workdir = self._workdir(tmp_path, DESIGN_MIN_MENTIONS)
        engine = FakeEngine(stored=GEMINI_VOICE_QUOTA - 1)
        designed, _ = design_voices(
            workdir, [_char("Narrator"), _char("Kelsier")], engine, None, {}
        )
        assert list(designed) == ["Narrator"]
        assert "quota reached" in capsys.readouterr().out


class TestDirectedDesign:
    def _run(self, tmp_path, answers, engine=None):
        from autiobook.binding import design_voices

        workdir = _workdir(tmp_path, "")
        engine = engine or FakeEngine()
        with (
            patch("autiobook.binding.prompt_choice", side_effect=answers),
            patch("autiobook.casting._start_preview", return_value=None),
            patch("autiobook.casting._stop_playback"),
        ):
            designed, quit_requested = design_voices(
                workdir, [_char("Narrator")], engine, None, {}, directed=True
            )
        return designed, quit_requested, engine

    def test_rejected_designs_are_deleted(self, tmp_path):
        designed, _, engine = self._run(tmp_path, ["n", "n", "y"])
        assert designed["Narrator"].voice_id == "voice_3"
        assert engine.deleted == ["voice_1", "voice_2"]
        assert engine.live == {"voice_3"}

    def test_preset_fallback_frees_the_slot(self, tmp_path):
        designed, quit_requested, engine = self._run(tmp_path, ["p"])
        assert designed == {} and not quit_requested
        assert engine.live == set()

    def test_quit(self, tmp_path):
        designed, quit_requested, engine = self._run(tmp_path, ["q"])
        assert designed == {} and quit_requested
        assert engine.live == set()

    def test_edit_redesigns_from_the_new_prompt(self, tmp_path):
        with patch("autiobook.audition._edit_description", return_value="Brighter."):
            designed, _, engine = self._run(tmp_path, ["e", "y"])
        assert engine.designs[-1] == "Brighter."
        assert designed["Narrator"].design_prompt == "Brighter."


class TestPresetAssignment:
    def test_new_characters_avoid_series_voices(self, tmp_path):
        from autiobook.audition import _assign_round_robin

        assigned = _assign_round_robin(
            FakeEngine(),
            [_char("Narrator"), _char("Vin"), _char("Elend")],
            tmp_path,
            fixed={"Narrator": "Zephyr"},
            used={"Zephyr", "Puck"},
        )
        assert assigned == {"Narrator": "Zephyr", "Vin": "Kore", "Elend": "Leda"}

    def test_reuse_once_every_voice_is_held(self, tmp_path):
        from autiobook.audition import _assign_round_robin

        assigned = _assign_round_robin(
            FakeEngine(presets=("Zephyr", "Puck")),
            [_char("Vin"), _char("Elend")],
            tmp_path,
            used={"Zephyr", "Puck"},
        )
        assert set(assigned) == {"Vin", "Elend"}

    def test_fixed_survives_force(self, tmp_path):
        from autiobook.audition import _assign_round_robin
        from autiobook.casting import save_voices

        save_voices(tmp_path, {"Vin": "Leda"})
        assigned = _assign_round_robin(
            FakeEngine(),
            [_char("Vin"), _char("Kelsier")],
            tmp_path,
            force=True,
            fixed={"Kelsier": "Puck"},
            used={"Puck"},
        )
        assert assigned["Kelsier"] == "Puck"
        assert assigned["Vin"] != "Leda"

    def test_standalone_order_is_unchanged(self, tmp_path):
        from autiobook.audition import _assign_round_robin

        assigned = _assign_round_robin(
            FakeEngine(), [_char("Narrator"), _char("Vin")], tmp_path
        )
        assert assigned == {"Narrator": "Zephyr", "Vin": "Puck"}

    def test_unused_first(self):
        from autiobook.casting import unused_first

        assert unused_first(["a", "b", "c"], {"a"}) == ["b", "c", "a"]


class TestPresetAudition:
    """the whole preset audition with a series: bind, design, assign, record."""

    def _run(self, tmp_path, engine, design=False):
        from autiobook.audition import _run_preset
        from autiobook.tts_http import HTTPTTSConfig

        workdir = tmp_path / "book2"
        cast = [_char("Narrator", voice="Deep."), _char("Kelsier"), _char("Vin")]
        save_cast(workdir, cast)
        with patch("autiobook.tts_http.HTTPTTSEngine", return_value=engine):
            _run_preset(
                workdir,
                cast,
                HTTPTTSConfig(model="gemini-3.8-flash-tts"),
                workdir / "audition",
                design=design,
            )
        from autiobook.casting import load_voices

        return load_voices(workdir), Series.load(tmp_path / "mistborn.json")

    def test_series_voice_holds_and_book_is_recorded(self, tmp_path):
        series = _series(tmp_path, _kelsier_entry(voice_id="Zephyr"))
        _workdir(tmp_path, "Vin. " * DESIGN_MIN_MENTIONS, series)

        voices, saved = self._run(tmp_path, FakeEngine())

        assert voices["Kelsier"] == "Zephyr"
        # the narrator would take the first preset, but the series holds it
        assert voices["Narrator"] != "Zephyr"
        assert {e.character.name for e in saved.entries} == {
            "Kelsier",
            "Narrator",
            "Vin",
        }
        vin = saved.match(_char("Vin"))
        assert vin.binding.voice_id == voices["Vin"]
        assert vin.binding.first_book == "The Well of Ascension"

    def test_design_skips_bound_characters(self, tmp_path):
        series = _series(tmp_path, _kelsier_entry(voice_id="Zephyr"))
        _workdir(tmp_path, "Vin. " * DESIGN_MIN_MENTIONS, series)
        engine = FakeEngine()

        voices, saved = self._run(tmp_path, engine, design=True)

        assert engine.designs == ["Deep.", "d"]
        assert voices["Kelsier"] == "Zephyr"
        assert voices["Narrator"].startswith("voice_")
        assert voices["Vin"].startswith("voice_")
        vin = saved.match(_char("Vin"))
        assert vin.binding.kind == BINDING_DESIGNED
        assert vin.binding.design_prompt == "d"

    def test_rerun_does_not_redesign(self, tmp_path):
        series = _series(tmp_path)
        _workdir(tmp_path, "Vin. " * DESIGN_MIN_MENTIONS, series)
        engine = FakeEngine()
        first, _ = self._run(tmp_path, engine, design=True)
        designs = len(engine.designs)

        second, _ = self._run(tmp_path, engine, design=True)

        assert len(engine.designs) == designs
        assert second == first

    def test_designs_survive_a_crash_mid_run(self, tmp_path):
        """a voice designed before the failure must not be designed again."""
        series = _series(tmp_path)
        _workdir(tmp_path, "Vin. " * DESIGN_MIN_MENTIONS, series)
        engine = FakeEngine()
        real = engine.create_designed_voice

        def fail_second(*args, **kwargs):
            if engine.designs:
                raise RuntimeError("http 500")
            return real(*args, **kwargs)

        engine.create_designed_voice = fail_second
        with pytest.raises(RuntimeError):
            self._run(tmp_path, engine, design=True)

        from autiobook.casting import load_voices

        assert load_voices(tmp_path / "book2") == {"Narrator": "voice_1"}
        saved = Series.load(tmp_path / "mistborn.json")
        assert saved.match(_char("Narrator")).binding.voice_id == "voice_1"

        engine.create_designed_voice = real
        voices, _ = self._run(tmp_path, engine, design=True)
        assert voices["Narrator"] == "voice_1"
        assert engine.designs == ["Deep.", "d"]
