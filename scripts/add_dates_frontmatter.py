#!/usr/bin/env python3
"""
Add date (created) and lastmod (modified) to frontmatter of all Obsidian vault .md files.

Usage:
  python3 scripts/add_dates_frontmatter.py              # process all vault files
  python3 scripts/add_dates_frontmatter.py file1.md ... # process specific files (git hook mode)
"""

import subprocess
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

VAULT_DIR = Path("/home/matrix/Documents/Obsidian Vault")
CONTENT_DIR = Path("/home/matrix/quartz/content")


def run_obsidian_file(relative_path: str) -> tuple[int | None, int | None]:
    """Run obsidian file command and return (created_ms, modified_ms)."""
    result = subprocess.run(
        ["obsidian", "file", f"path={relative_path}"],
        capture_output=True,
        text=True,
    )
    created = None
    modified = None
    for line in result.stdout.splitlines():
        if line.startswith("created\t"):
            created = int(line.split("\t", 1)[1])
        elif line.startswith("modified\t"):
            modified = int(line.split("\t", 1)[1])
    return created, modified


def ms_to_date_str(ms: int) -> str:
    """Convert millisecond timestamp to YYYY-MM-DD string."""
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def update_frontmatter(file_path: Path, date_str: str, lastmod_str: str) -> bool:
    """
    Add or update date and lastmod in file frontmatter.
    Returns True if file was modified.
    """
    content = file_path.read_text(encoding="utf-8")

    if content.startswith("---"):
        # Has frontmatter - find end of frontmatter block
        end = content.find("\n---", 3)
        if end == -1:
            print(f"  WARNING: malformed frontmatter in {file_path}, skipping")
            return False

        fm_block = content[3:end]  # content between the two ---
        rest = content[end + 4:]   # content after closing ---

        # Strip leading/trailing blank lines from frontmatter block before parsing
        lines = fm_block.strip().splitlines()
        has_date = False
        has_lastmod = False
        new_lines = []
        for line in lines:
            if line.startswith("date:"):
                new_lines.append(f"date: {date_str}")
                has_date = True
            elif line.startswith("lastmod:"):
                new_lines.append(f"lastmod: {lastmod_str}")
                has_lastmod = True
            else:
                new_lines.append(line)

        if not has_date:
            new_lines.insert(0, f"date: {date_str}")
        if not has_lastmod:
            # insert after date line
            date_idx = next((i for i, l in enumerate(new_lines) if l.startswith("date:")), 0)
            new_lines.insert(date_idx + 1, f"lastmod: {lastmod_str}")

        new_fm = "\n".join(new_lines)
        # Always add newline after opening --- and before closing ---
        new_content = f"---\n{new_fm}\n---{rest}"
    else:
        # No frontmatter - prepend it
        new_content = f"---\ndate: {date_str}\nlastmod: {lastmod_str}\n---\n{content}"

    if new_content != content:
        file_path.write_text(new_content, encoding="utf-8")
        return True
    return False


def process_file(file_path: Path) -> bool:
    """Process a single file. Returns True if updated."""
    # Try to find it in the vault to get obsidian timestamps
    vault_relative = None
    if file_path.is_relative_to(VAULT_DIR):
        vault_relative = str(file_path.relative_to(VAULT_DIR))
    elif file_path.is_relative_to(CONTENT_DIR):
        # quartz/content file - check if matching vault file exists
        content_relative = file_path.relative_to(CONTENT_DIR)
        vault_path = VAULT_DIR / content_relative
        if vault_path.exists():
            vault_relative = str(content_relative)

    if vault_relative:
        created_ms, modified_ms = run_obsidian_file(vault_relative)
    else:
        created_ms, modified_ms = None, None

    if created_ms is None or modified_ms is None:
        print(f"  WARNING: obsidian timestamps unavailable for {file_path}, skipping")
        return False

    date_str = ms_to_date_str(created_ms)
    lastmod_str = ms_to_date_str(modified_ms)

    changed = update_frontmatter(file_path, date_str, lastmod_str)
    if changed:
        print(f"  Updated: date={date_str}, lastmod={lastmod_str}")
    else:
        print(f"  Unchanged (date={date_str}, lastmod={lastmod_str})")
    return changed


def main():
    if len(sys.argv) > 1:
        # Called with specific file paths (e.g., from git hook)
        files = [Path(p) for p in sys.argv[1:] if p.endswith(".md")]
        if not files:
            print("No .md files to process")
            return
        print(f"Processing {len(files)} staged .md file(s)")
        updated = sum(1 for f in files if process_file(f))
        print(f"Done: {updated} file(s) updated")
    else:
        # Process all vault files
        md_files = sorted(VAULT_DIR.rglob("*.md"))
        print(f"Found {len(md_files)} markdown files in vault")
        updated = 0
        errors = 0
        for file_path in md_files:
            relative_path = str(file_path.relative_to(VAULT_DIR))
            print(f"Processing: {relative_path}")
            created_ms, modified_ms = run_obsidian_file(relative_path)
            if created_ms is None or modified_ms is None:
                print(f"  WARNING: could not get timestamps, skipping")
                errors += 1
                continue
            date_str = ms_to_date_str(created_ms)
            lastmod_str = ms_to_date_str(modified_ms)
            if update_frontmatter(file_path, date_str, lastmod_str):
                updated += 1
                print(f"  Updated: date={date_str}, lastmod={lastmod_str}")
            else:
                print(f"  Unchanged (date={date_str}, lastmod={lastmod_str})")
        print(f"\nDone: {updated} files updated, {errors} errors")


if __name__ == "__main__":
    main()
