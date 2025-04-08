import os
import shutil
import fnmatch


def parse_gitignore(gitignore_path):
    """Parse a .gitignore file and return a list of patterns."""
    patterns = []
    with open(gitignore_path, "r") as f:
        for line in f:
            # Ignore comments and blank lines
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Handle wildcards and directory separators
            patterns.append(line)
    return patterns


def file_matches_patterns(file_path, patterns):
    """Check if a file matches any of the patterns in .gitignore."""
    for pattern in patterns:
        if fnmatch.fnmatch(file_path, pattern):
            return True
    return False


def copy_codebase(src, dst, max_size_mb=5, gitignore_path=None):
    """Copy files from src to dst, skipping files larger than max_size_mb and matching .gitignore patterns."""
    if gitignore_path and os.path.exists(gitignore_path):
        patterns = parse_gitignore(gitignore_path)
    else:
        patterns = []
    print("patterns to ignore: ", patterns)
    os.makedirs(dst, exist_ok=True)
    for root, dirs, files in os.walk(src):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, src)
            dst_path = os.path.join(dst, relative_path)
            # ignore .git because of permission issues
            if "/.git/" in file_path:
                continue

            # Check .gitignore patterns
            if file_matches_patterns(file_path, patterns):
                # print(f"Skipping {file_path} because it matches a pattern in .gitignore")
                continue

            # Check file size
            if os.path.getsize(file_path) > max_size_mb * 1024 * 1024:
                print(f"Skipping {file_path} because it's larger than {max_size_mb}MB")
                continue

            # Make sure the destination directory exists
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(file_path, dst_path)
