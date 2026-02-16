from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

REPO_URL = "https://github.com/aamarzan/projects.git"

AAM_DESKTOP = Path(r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop")
E_AHMED = Path(r"E:\Dr. Ahmed")

REPO_PATH = AAM_DESKTOP / "projects_py_only"

SOURCES = [
    ("AAM_Desktop", AAM_DESKTOP),
    ("Dr_Ahmed", E_AHMED),
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


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return CompletedProcess."""
    # Use shell=False for safety; Git and RoboCopy are executables on PATH.
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

    # Fetch and hard reset to origin/main if it exists
    # (prevents non-fast-forward problems)
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
    gi = REPO_PATH / ".gitignore"
    gi.write_text("*\n!*/\n!**/*.py\n!.gitignore\n", encoding="utf-8")


def should_skip_dir(path: Path) -> bool:
    return path.name in EXCLUDE_DIRS


def collect_py_files() -> None:
    for name, src_root in SOURCES:
        if not src_root.exists():
            warn(f"Source not found, skipping: {src_root}")
            continue

        dest_root = REPO_PATH / name
        dest_root.mkdir(parents=True, exist_ok=True)

        info(f"Collecting *.py from: {src_root} -> {dest_root}")

        # Walk source tree
        for root, dirs, files in os.walk(src_root):
            root_path = Path(root)

            # prune excluded dirs
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for fn in files:
                if not fn.lower().endswith(".py"):
                    continue
                src_file = root_path / fn

                # compute relative path under source root
                rel = src_file.relative_to(src_root)
                dst_file = dest_root / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)

                # copy only if changed (size+mtime quick check; then overwrite)
                if dst_file.exists():
                    try:
                        if src_file.stat().st_size == dst_file.stat().st_size and int(src_file.stat().st_mtime) == int(dst_file.stat().st_mtime):
                            continue
                    except FileNotFoundError:
                        pass

                shutil.copy2(src_file, dst_file)


def list_changed_py_files() -> list[str]:
    p = run(["git", "status", "--porcelain"], cwd=REPO_PATH, check=False)
    if p.returncode != 0:
        raise RuntimeError(p.stderr)

    changed = []
    for line in p.stdout.splitlines():
        if len(line) < 4:
            continue
        path_part = line[3:]  # keep spaces inside file path
        # Handle rename "old -> new"
        if "->" in path_part:
            path_part = path_part.split("->", 1)[1].strip()
        if path_part.lower().endswith(".py"):
            changed.append(path_part)
    # de-dup while preserving order
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
        # Stage ONLY this path
        run(["git", "add", "-A", "--", f], cwd=REPO_PATH)

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
    collect_py_files()

    # Stage everything (py-only enforced by .gitignore)
    run(["git", "add", "-A"], cwd=REPO_PATH)

    changed = list_changed_py_files()
    stage_commit_push_one_by_one(changed)

    info("Done.")


if __name__ == "__main__":
    main()
