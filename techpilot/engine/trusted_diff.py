"""Build and apply file-write proposals without writing before approval."""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass
from pathlib import Path


class TrustedDiffError(ValueError):
    """The requested candidate content could not be constructed safely."""


class SourceSnapshotChanged(RuntimeError):
    """The real file changed between preview generation and execution."""


@dataclass(frozen=True, slots=True)
class FileWriteProposal:
    """Runtime-owned candidate content, trusted diff, and source snapshot."""

    path: Path
    display_path: str
    original_exists: bool
    original_content: str
    candidate_content: str
    source_snapshot: str
    trusted_diff: str

    @classmethod
    def for_edit(
        cls,
        path: Path,
        *,
        display_path: str,
        old_string: str,
        new_string: str,
    ) -> FileWriteProposal:
        if not isinstance(old_string, str) or not old_string or not isinstance(new_string, str):
            raise TrustedDiffError("edit_file requires non-empty old_string and string new_string")
        exists, content, snapshot = _read_source(path)
        if not exists:
            raise TrustedDiffError(f"{display_path} not found")
        occurrences = content.count(old_string)
        if occurrences == 0:
            raise TrustedDiffError(f"old_string not found in {display_path}")
        if occurrences > 1:
            raise TrustedDiffError(
                f"old_string appears {occurrences} times in {display_path}; "
                "include more surrounding context"
            )
        candidate = content.replace(old_string, new_string, 1)
        return cls._build(path, display_path, True, content, candidate, snapshot)

    @classmethod
    def for_write(
        cls,
        path: Path,
        *,
        display_path: str,
        content: str,
    ) -> FileWriteProposal:
        exists, original, snapshot = _read_source(path)
        return cls._build(path, display_path, exists, original, content, snapshot)

    @classmethod
    def _build(
        cls,
        path: Path,
        display_path: str,
        exists: bool,
        original: str,
        candidate: str,
        snapshot: str,
    ) -> FileWriteProposal:
        return cls(
            path=path,
            display_path=display_path,
            original_exists=exists,
            original_content=original,
            candidate_content=candidate,
            source_snapshot=snapshot,
            trusted_diff=_unified_diff(original, candidate, display_path, exists),
        )

    @property
    def has_changes(self) -> bool:
        return self.original_content != self.candidate_content or not self.original_exists

    def source_is_current(self) -> bool:
        """Return whether the real file still matches the approved source."""

        return _source_snapshot(self.path) == self.source_snapshot

    def apply(self) -> None:
        """Revalidate immediately, then write exactly the approved candidate."""

        if not self.source_is_current():
            raise SourceSnapshotChanged(f"Source snapshot changed for {self.display_path}")
        encoded = self.candidate_content.encode("utf-8")
        if self.original_exists:
            self.path.write_bytes(encoded)
            return

        # Exclusive creation closes the most important missing-file race: an
        # external creator is never silently overwritten after approval.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("xb") as handle:
                handle.write(encoded)
        except FileExistsError as error:
            raise SourceSnapshotChanged(
                f"Source snapshot changed for {self.display_path}"
            ) from error


def _read_source(path: Path) -> tuple[bool, str, str]:
    if not path.exists():
        return False, "", _missing_snapshot()
    if not path.is_file():
        raise TrustedDiffError(f"{path} is not a regular file")
    data = path.read_bytes()
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TrustedDiffError(f"{path} is not a UTF-8 text file") from error
    return True, content, _content_snapshot(data)


def _source_snapshot(path: Path) -> str:
    if not path.exists():
        return _missing_snapshot()
    if not path.is_file():
        return "non-file"
    return _content_snapshot(path.read_bytes())


def _missing_snapshot() -> str:
    return hashlib.sha256(b"missing\0").hexdigest()


def _content_snapshot(data: bytes) -> str:
    return hashlib.sha256(b"file\0" + data).hexdigest()


def _unified_diff(old: str, new: str, display_path: str, existed: bool) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    fromfile = f"a/{display_path}" if existed else "/dev/null"
    result = "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=fromfile,
            tofile=f"b/{display_path}",
            n=3,
        )
    )
    if not existed and not result:
        return f"--- /dev/null\n+++ b/{display_path}\n"
    return result
