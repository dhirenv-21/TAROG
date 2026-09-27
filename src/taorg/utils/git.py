"""Git commit capture utility.

Captures the current git commit hash for experiment provenance.
Falls back gracefully when git is unavailable or the directory
is not a git repository.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def get_git_commit(repo_dir: Path | None = None) -> str:
    """Return the current git commit hash, or 'unknown'.

    Args:
        repo_dir: Directory to check. Defaults to cwd.

    Returns:
        Short git commit hash string, or 'unknown' if unavailable.
    """
    try:
        cwd = str(repo_dir) if repo_dir else None
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        logger.debug("git rev-parse returned %d", result.returncode)
        return "unknown"
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.debug("Could not capture git commit: %s", e)
        return "unknown"


def get_git_dirty(repo_dir: Path | None = None) -> bool:
    """Check whether the working tree has uncommitted changes.

    Args:
        repo_dir: Directory to check. Defaults to cwd.

    Returns:
        True if there are uncommitted changes, False otherwise.
        Returns True (conservative) if git is unavailable.
    """
    try:
        cwd = str(repo_dir) if repo_dir else None
        result = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"],
            capture_output=True,
            cwd=cwd,
            timeout=5,
        )
        return result.returncode != 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return True


def get_git_branch(repo_dir: Path | None = None) -> str:
    """Return the current git branch name, or 'unknown'.

    Args:
        repo_dir: Directory to check. Defaults to cwd.

    Returns:
        Branch name string, or 'unknown' if unavailable.
    """
    try:
        cwd = str(repo_dir) if repo_dir else None
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return "unknown"
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"
