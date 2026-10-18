"""series casts: one set of characters and voice bindings shared by many books.

a series file holds every character met so far in a series and the voice each
was given. a book linked to a series:
  - cast: starts from the series characters its text names, and never
    rewrites one -- name, description, voice prompt and audition line are
    locked to the series; aliases the book teaches are still learned
  - audition: takes the bound voice for every series character, gives new
    characters voices the series has not used yet, then records them

the series file is the source of truth: edit it to recast or redescribe a
character for every later book. per-book files (characters.json,
voices.json) are regenerated from it.
"""

import json
import os
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any

from .config import (
    BINDING_DESIGNED,
    BINDING_PRESET,
    METADATA_FILE,
    SERIES_ENV,
    SERIES_LINK_FILE,
    SERIES_VERSION,
    TXT_EXT,
)
from .llm import Character
from .resume import load_state, save_state

# a series character joins a book's cast before the llm reads it once the
# text names it this often: the llm then reuses the series name instead of
# inventing a variant. twice, so a name that is also a word ("Hope") does not
# join on a single capitalized use.
SEED_MIN_MENTIONS = 2


@dataclass
class VoiceBinding:
    """the voice a series character speaks with, and how to rebuild it."""

    voice_id: str
    kind: str = BINDING_PRESET
    # for a designed voice: what it was designed from, kept so a voice google
    # has expired can be designed again from the same description
    model: str = ""
    design_prompt: str = ""
    # designed voices: google's preview clip, relative to the series file
    sample: str = ""
    first_book: str = ""
    created: str = ""
    last_used: str = ""

    @classmethod
    def from_json(cls, data: dict) -> "VoiceBinding":
        known = {f.name for f in fields(cls)}
        return cls(**{k: str(v) for k, v in data.items() if k in known})

    def to_json(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v}


@dataclass
class SeriesEntry:
    character: Character
    binding: VoiceBinding | None = None
    # optional voice design inputs (gemini takes "female"/"male"/"neutral"
    # and a bcp-47 tag); set by hand, never inferred
    gender: str = ""
    language_code: str = ""

    def forms(self) -> list[str]:
        """every name this character goes by that can be searched for."""
        return _forms(self.character)


@dataclass
class Series:
    path: Path
    entries: list[SeriesEntry] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "Series":
        """read a series file; a missing one is an empty series to start."""
        if not path.exists():
            return cls(path=path)
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = []
        for raw in data.get("characters", []):
            binding = raw.get("binding")
            entries.append(
                SeriesEntry(
                    character=Character(
                        name=raw["name"],
                        description=raw.get("description", ""),
                        aliases=raw.get("aliases") or None,
                        voice=raw.get("voice", ""),
                        audition_line=raw.get("audition_line", ""),
                    ),
                    binding=VoiceBinding.from_json(binding) if binding else None,
                    gender=raw.get("gender", ""),
                    language_code=raw.get("language_code", ""),
                )
            )
        return cls(path=path, entries=entries)

    def save(self) -> None:
        """write the series file atomically: it outlives every book's workdir,
        so a crash mid-write must not cost the whole series its bindings."""
        characters = []
        for e in self.entries:
            c = e.character
            raw: dict[str, Any] = {
                "name": c.name,
                "description": c.description,
                "voice": c.voice_prompt(),
                "aliases": c.aliases,
                "audition_line": c.audition_line,
            }
            if e.gender:
                raw["gender"] = e.gender
            if e.language_code:
                raw["language_code"] = e.language_code
            if e.binding:
                raw["binding"] = e.binding.to_json()
            characters.append(raw)
        payload = json.dumps(
            {"version": SERIES_VERSION, "characters": characters},
            indent=2,
            ensure_ascii=False,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.path)

    @property
    def voices_dir(self) -> Path:
        """where designed voices' preview clips are kept, beside the file."""
        return self.path.parent / f"{self.path.stem}_voices"

    def match(self, char: Character) -> SeriesEntry | None:
        """the series character `char` is, by name or alias.

        the canonical name is checked first, as the cast merge does: a book
        character re-emitted under its series name is the same person even
        if its aliases also touch another entry.
        """
        names = {e.character.name.casefold(): e for e in self.entries}
        aliases = {
            a.casefold(): e for e in self.entries for a in e.character.aliases or []
        }
        key = char.name.casefold()
        if key in names:
            return names[key]
        if key in aliases:
            return aliases[key]
        for a in char.aliases or []:
            hit = names.get(a.casefold()) or aliases.get(a.casefold())
            if hit:
                return hit
        return None

    def used_voices(self) -> set[str]:
        """every voice the series has already given someone."""
        return {e.binding.voice_id for e in self.entries if e.binding}


def _forms(char: Character) -> list[str]:
    from .dramatize import _usable_alias

    canon = char.name.casefold()
    return [char.name] + [a for a in char.aliases or [] if _usable_alias(a, canon)]


def mention_count(text: str, char: Character) -> int:
    """how often the text names a character, by name or alias.

    case-sensitive and whole-word: names are capitalized, and matching "will"
    for Will would count every future tense in the book.
    """
    total = 0
    for form in _forms(char):
        pattern = rf"(?<!\w){re.escape(form)}(?!\w)"
        total += len(re.findall(pattern, text))
    return total


def book_text(workdir: Path) -> str:
    """the book's extracted chapter text, for counting mentions."""
    extract_dir = workdir / "extract"
    return "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(extract_dir.glob(f"*{TXT_EXT}"))
    )


def book_label(workdir: Path) -> str:
    """the book's title, falling back to its workdir's name."""
    meta = workdir / "extract" / METADATA_FILE
    if meta.exists():
        try:
            title = json.loads(meta.read_text(encoding="utf-8")).get("title")
            if isinstance(title, str) and title.strip():
                return title.strip()
        except (json.JSONDecodeError, OSError):
            pass
    return workdir.name


def _absorb_aliases(target: Character, other: Character) -> None:
    """add `other`'s naming forms to `target`'s aliases."""
    from .dramatize import _usable_alias

    canon = target.name.casefold()
    merged = set(target.aliases or [])
    for form in [other.name, *(other.aliases or [])]:
        if _usable_alias(form, canon):
            merged.add(form)
    target.aliases = sorted(merged) or None


def lock_cast(
    cast: list[Character], series: Series
) -> tuple[list[Character], set[str], list[str]]:
    """restore every series character in a book's cast to its series form.

    returns (cast, locked names (casefolded), change notes). a book character
    found under an alias is renamed to its series name; two book characters
    that turn out to be one series character are folded together, keeping the
    first one's place in the cast.
    """
    out: list[Character] = []
    locked: dict[int, Character] = {}
    notes: list[str] = []
    for c in cast:
        entry = series.match(c)
        if entry is None:
            out.append(c)
            continue
        target = locked.get(id(entry))
        if target is None:
            target = deepcopy(entry.character)
            locked[id(entry)] = target
            out.append(target)
            if c.name != target.name:
                notes.append(f"'{c.name}' is series character '{target.name}'")
            elif (c.description, c.voice_prompt(), c.audition_line) != (
                target.description,
                target.voice_prompt(),
                target.audition_line,
            ):
                notes.append(f"'{c.name}' restored to its series description")
        else:
            notes.append(f"'{c.name}' folded into series character '{target.name}'")
        _absorb_aliases(target, c)
    return out, {c.name.casefold() for c in locked.values()}, notes


def seed_cast(
    cast: list[Character], series: Series, text: str
) -> tuple[list[Character], set[str], list[str]]:
    """lock a book's cast and add the series characters its text names.

    run before the cast llm reads the book, so it sees series characters as
    already known and keeps their names rather than coining new ones.
    """
    cast, locked, notes = lock_cast(cast, series)
    for entry in series.entries:
        if entry.character.name.casefold() in locked:
            continue
        if mention_count(text, entry.character) < SEED_MIN_MENTIONS:
            continue
        cast.append(deepcopy(entry.character))
        locked.add(entry.character.name.casefold())
        notes.append(f"'{entry.character.name}' joins from the series")
    return cast, locked, notes


def record_book(
    series: Series,
    cast: list[Character],
    voices: dict[str, str],
    new_bindings: dict[str, VoiceBinding],
    book: str,
) -> int:
    """add a book's cast and voices to the series; returns entries added.

    a series character keeps the binding it has unless `new_bindings` names a
    replacement (a designed voice rebuilt after google expired it). aliases
    the book taught are kept for the next book.
    """
    today = date.today().isoformat()
    added = 0
    for c in cast:
        entry = series.match(c)
        if entry is None:
            entry = SeriesEntry(character=deepcopy(c))
            series.entries.append(entry)
            added += 1
        else:
            _absorb_aliases(entry.character, c)
        voice_id = voices.get(c.name)
        if not voice_id:
            continue
        if c.name in new_bindings:
            entry.binding = new_bindings[c.name]
        elif entry.binding is None or entry.binding.voice_id != voice_id:
            kind = BINDING_DESIGNED if voice_id.startswith("voice_") else BINDING_PRESET
            entry.binding = VoiceBinding(
                voice_id=voice_id, kind=kind, first_book=book, created=today
            )
        entry.binding.last_used = today
    return added


def link_path(workdir: Path) -> Path:
    return workdir / SERIES_LINK_FILE


def resolve_series(workdir: Path | None, flag: str | None) -> Path | None:
    """the series this book belongs to, recorded in the workdir.

    precedence: --series, then $AUTIOBOOK_SERIES, then the link the book
    already holds. linking is recorded so a resumed run keeps its locks
    without the flag; a book is moved to another series by passing it again.
    """
    explicit = flag or os.getenv(SERIES_ENV) or ""
    if workdir is None:
        return Path(explicit).resolve() if explicit else None
    path = link_path(workdir)
    stored = load_state(path).get("series")
    if not explicit:
        return Path(stored) if isinstance(stored, str) and stored else None
    resolved = Path(explicit).resolve()
    if stored and Path(stored) != resolved:
        print(f"series: moving this book from {stored} to {resolved}")
    workdir.mkdir(parents=True, exist_ok=True)
    save_state(path, {"series": str(resolved)})
    return resolved


def load_series_for(workdir: Path) -> Series | None:
    """the series a book is linked to, or None for a standalone book."""
    stored = load_state(link_path(workdir)).get("series")
    if not isinstance(stored, str) or not stored:
        return None
    return Series.load(Path(stored))
