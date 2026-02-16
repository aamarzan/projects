from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import ctypes

# =========================
# CONFIG
# =========================
REPO_URL = "https://github.com/aamarzan/projects.git"

AAM_DESKTOP = Path(r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop")
E_AHMED     = Path(r"E:\Dr. Ahmed")

REPO_PATH = AAM_DESKTOP / "projects_py_only"

SOURCES = [
    ("AAM_Desktop", AAM_DESKTOP),
    ("Dr_Ahmed",    E_AHMED),
]

EXCLUDE_DIRS = {
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".git",
    "node_modules",
    "projects_py_only",
}

# OneDrive “cloud-only” / placeholder detection (Windows file attributes)
FILE_ATTRIBUTE_OFFLINE = 0x00001000
FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x00040000
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000
INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF

kernel32 = ctypes.windll.kernel32
kernel32.GetFileAttributesW.argtypes = [ctypes.c_wchar_p]
kernel32.GetFileAttributesW.restype = ctypes.c_uint32


def get_attrs(p: Path) -> int | None:
    attrs = kernel32.GetFileAttributesW(str(p))
    if attrs == INVALID_FILE_ATTRIBUTES:
        return None
    return int(attrs)


def is_cloud_only(p: Path) -> bool:
    """
    Returns True if path looks like a OneDrive cloud-only placeholder (not stored locally).
    We primarily rely on OFFLINE attribute. This avoids triggering hydration/download.
    """
    attrs = get_attrs(p)
    if attrs is None:
        return False

    # Strong signal: OFFLINE (typical for cloud-only placeholders)
    if attrs & FILE_ATTRIBUTE_OFFLINE:
        return True

    # Extra safety: some placeholders are reparse points with recall flags
    if (attrs & FILE_ATTRIBUTE_REPARSE_POINT) and (attrs & (FILE_ATTRIBUTE_RECALL_ON_OPEN | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS)):
        return True

    return False


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"cwd: {cwd}\n"
            f"exit: {p.returncode}\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}"
        )
    return p


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def ensure_repo() -> None:
    REPO_PATH.mkdir(parents=True, exist_ok=True)

    git_dir = REPO_PATH / ".git"
    if not git_dir.exists():
        info(f"Initializing collector repo at: {REPO_PATH}")
        run(["git", "init"], cwd=REPO_PATH)

    # Ensure origin points to correct repo
    p = subprocess.run(["git", "remote", "get-url", "origin"], cwd=str(REPO_PATH), text=True, capture_output=True)
    if p.returncode != 0:
        run(["git", "remote", "add", "origin", REPO_URL], cwd=REPO_PATH)
    else:
        origin = p.stdout.strip()
        if origin != REPO_URL:
            run(["git", "remote", "set-url", "origin", REPO_URL], cwd=REPO_PATH)

    # Ensure main branch
    run(["git", "branch", "-M", "main"], cwd=REPO_PATH)

    # Fetch and hard reset to origin/main if it exists (prevents non-fast-forward)
    subprocess.run(["git", "fetch", "origin"], cwd=str(REPO_PATH), text=True, capture_output=True)
    ref_check = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", "refs/remotes/origin/main"],
        cwd=str(REPO_PATH),
        text=True,
        capture_output=True,
    )
    if ref_check.returncode == 0:
        run(["git", "checkout", "-B", "main", "origin/main"], cwd=REPO_PATH)
        run(["git", "reset", "--hard", "origin/main"], cwd=REPO_PATH)
    else:
        run(["git", "checkout", "-B", "main"], cwd=REPO_PATH)


def clean_worktree_keep_git() -> None:
    # Remove everything except .git
    for child in REPO_PATH.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except FileNotFoundError:
                pass


def write_gitignore_py_only() -> None:
    (REPO_PATH / ".gitignore").write_text("*\n!*/\n!**/*.py\n!.gitignore\n", encoding="utf-8")


def collect_py_files() -> tuple[int, int, int]:
    """
    Copies *.py from sources into collector repo, preserving structure.
    Skips excluded dirs and skips OneDrive cloud-only placeholders (won't download).
    Returns (copied, skipped_cloud, skipped_missing_src)
    """
    copied = 0
    skipped_cloud = 0
    skipped_missing = 0

    for name, src_root in SOURCES:
        if not src_root.exists():
            warn(f"Source not found, skipping: {src_root}")
            skipped_missing += 1
            continue

        # If the *source root* is cloud-only (rare, but possible), skip it completely
        if is_cloud_only(src_root):
            warn(f"Source is cloud-only (not local), skipping to avoid download: {src_root}")
            skipped_cloud += 1
            continue

        dest_root = REPO_PATH / name
        dest_root.mkdir(parents=True, exist_ok=True)

        info(f"Collecting *.py from: {src_root} -> {dest_root}")

        for root, dirs, files in os.walk(src_root, topdown=True):
            root_path = Path(root)

            # prune excluded dirs + prune cloud-only dirs (prevents OneDrive hydration)
            new_dirs = []
            for d in dirs:
                if d in EXCLUDE_DIRS:
                    continue
                dp = root_path / d
                if is_cloud_only(dp):
                    skipped_cloud += 1
                    continue
                new_dirs.append(d)
            dirs[:] = new_dirs

            for fn in files:
                if not fn.lower().endswith(".py"):
                    continue
                src_file = root_path / fn

                # If file is cloud-only placeholder, SKIP (prevents download)
                if is_cloud_only(src_file):
                    skipped_cloud += 1
                    continue

                rel = src_file.relative_to(src_root)
                dst_file = dest_root / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)

                # copy if missing or changed (size + mtime quick check)
                if dst_file.exists():
                    try:
                        s1 = src_file.stat()
                        s2 = dst_file.stat()
                        if (s1.st_size == s2.st_size) and (int(s1.st_mtime) == int(s2.st_mtime)):
                            continue
                    except FileNotFoundError:
                        pass

                shutil.copy2(src_file, dst_file)
                copied += 1

    return copied, skipped_cloud, skipped_missing


def list_changed_py_files() -> list[str]:
    p = run(["git", "status", "--porcelain"], cwd=REPO_PATH, check=False)
    if p.returncode != 0:
        raise RuntimeError(p.stderr)

    changed: list[str] = []
    for line in p.stdout.splitlines():
        if len(line) < 4:
            continue
        path_part = line[3:]
        if "->" in path_part:  # rename
            path_part = path_part.split("->", 1)[1].strip()
        if path_part.lower().endswith(".py"):
            changed.append(path_part)

    # de-dup preserve order
    seen = set()
    out = []
    for x in changed:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def stage_commit_push_one_by_one(changed_files: list[str]) -> None:
    if not changed_files:
        warn("No Python changes found to push today.")
        return

    info(f"Will commit/push these .py files one-by-one: {len(changed_files)}")

    for f in changed_files:
        # IMPORTANT: ensure each commit contains ONLY one file
        run(["git", "reset"], cwd=REPO_PATH)              # unstage everything
        run(["git", "add", "--", f], cwd=REPO_PATH)      # stage only this file

        # If nothing staged, skip
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(REPO_PATH))
        if diff.returncode == 0:
            warn(f"Skipping (no staged change): {f}")
            continue

        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        msg = f"Update {f} ({ts})"
        run(["git", "commit", "-m", msg], cwd=REPO_PATH)

        info(f"Pushing: {f}")
        run(["git", "push", "origin", "main"], cwd=REPO_PATH)


def main() -> None:
    ensure_repo()
    clean_worktree_keep_git()
    write_gitignore_py_only()

    copied, skipped_cloud, skipped_missing = collect_py_files()

    # DO NOT stage everything here (it would break one-by-one commits)
    changed = list_changed_py_files()
    stage_commit_push_one_by_one(changed)

    info(f"Copy summary: copied={copied}, skipped_cloud_only={skipped_cloud}, skipped_missing_sources={skipped_missing}")
    info("Done.")


if __name__ == "__main__":
    main()
