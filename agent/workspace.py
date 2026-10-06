"""Confines every path the model supplies to one directory."""
from __future__ import annotations

from pathlib import Path


class WorkspaceError(ValueError):
    """A path supplied by the model is unusable or points outside the workspace."""


class Workspace:
    def __init__(self, root: str | Path) -> None:
        root_path = Path(root)
        if not root_path.is_dir():
            raise NotADirectoryError(f"Working directory does not exist: {root}")
        # Resolve once, so later comparisons are between real paths.
        self.root = root_path.resolve()

    def resolve(self, relative_path: str) -> Path:
        """Return the real path for `relative_path`, or raise if it leaves the workspace.

        `Path.resolve()` collapses `..` and follows symbolic links, so a link
        inside the workspace that points somewhere else is rejected too.
        """
        try:
            candidate = (self.root / relative_path).resolve()
        except (OSError, RuntimeError, ValueError) as exc:  # null byte, symlink loop, ...
            raise WorkspaceError(f'"{relative_path}" is not a usable path ({exc})') from exc
        if candidate != self.root and self.root not in candidate.parents:
            raise WorkspaceError(
                f'"{relative_path}" is outside the permitted working directory'
            )
        return candidate
