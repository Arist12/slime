import json
import re
from typing import Any


def code_edit_grader(response: str, original_code: str, mode: str = "adaptive") -> tuple[str, bool, str]:
    """
    Determine the format used and whether it was successful.
    Follows the exact logic from code_rm.py

    Args:
        response: Model's response containing code edits
        original_code: Original code before edits
        mode: Editing mode, one of "adaptive", "find_replace", "full_rewrite"

    Returns: (edit_mode, format_success, extracted_code)
    """
    edit_mode, extracted_code = extract_model_code(response, original_code, mode)
    format_success = extracted_code is not None
    return edit_mode, format_success, extracted_code


def extract_model_code(response: str, original_code: str, mode: str = "adaptive") -> tuple[str | None, str | None]:
    """
    Extract model code from response based on the specified mode.
    Follows the exact logic from code_rm.py:
    - adaptive/find_replace: use SEARCH/REPLACE blocks (adaptive allows empty SEARCH)
    - full_rewrite: use markdown code blocks

    Args:
        response: Model's response containing code edits
        original_code: Original code before edits
        mode: Editing mode, one of "adaptive", "find_replace", "full_rewrite"

    Returns: (edit_mode, extracted_code) or (None, None) on failure.
    """
    try:
        if mode == "adaptive" or mode == "find_replace":
            # Use SEARCH/REPLACE blocks, passing the mode to handle empty SEARCH
            return extract_search_replace(response, original_code, mode)
        elif mode == "full_rewrite":
            # Use markdown code blocks
            return extract_full_rewrite(response)
        else:
            return None, None
    except Exception:
        return None, None


def extract_search_replace(response: str, original_code: str, mode: str = "adaptive") -> tuple[str | None, str | None]:
    """
    Extract code using SEARCH/REPLACE blocks.
    This follows the exact logic from code_rm.py's extract_search_replace function.

    Args:
        response: Model's response
        original_code: Original code before edits
        mode: "adaptive" allows empty SEARCH (full rewrite), "find_replace" requires non-empty SEARCH

    Returns: (edit_mode, extracted_code) or (edit_mode, None) on failure.
    """
    try:
        # Pattern to match SEARCH/REPLACE blocks (aligned with LLMFileEditor format)
        pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
        matches = list(re.finditer(pattern, response, re.DOTALL))

        if not matches:
            return None, None

        # Extract first search and replace blocks
        first_search = matches[0].group(1)
        first_replace = matches[0].group(2)

        # Check if this is a full rewrite (empty search block)
        if mode == "adaptive" and first_search.strip() == "":
            # Full rewrite mode: must have exactly one SEARCH/REPLACE block
            if len(matches) > 1:
                return "fully_rewrite", None
            return "fully_rewrite", first_replace.rstrip("\n")

        # Targeted edits: Apply each SEARCH/REPLACE block
        modified_code = original_code
        for match in matches:
            search_block = match.group(1).rstrip("\n")
            replace_block = match.group(2).rstrip("\n")

            # Check if search block exists in the current code
            if search_block not in modified_code:
                return "find_replace", None

            # Check for multiple occurrences (ambiguous, should error)
            occurrences = modified_code.count(search_block)
            if occurrences > 1:
                return "find_replace", None

            # Apply the replacement: replace only the first occurrence
            modified_code = modified_code.replace(search_block, replace_block, 1)

        return "find_replace", modified_code

    except Exception:
        return None, None


def extract_full_rewrite(response: str) -> tuple[str | None, str | None]:
    """
    Extract code from markdown code blocks for full rewrite mode.
    This follows the exact logic from code_rm.py's extract_full_rewrite function.
    Uses a generic pattern to match any programming language.

    Returns: ("fully_rewrite", extracted_code) or ("fully_rewrite", None) on failure.
    """
    try:
        # Pattern to match standard markdown code blocks with any language tag
        # \w+ matches one or more word characters (letters, digits, underscore)
        pattern = r"```\w+\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)

        if not match:
            return "fully_rewrite", None

        # Extract the code content
        code_content = match.group(1).rstrip("\n")
        return "fully_rewrite", code_content

    except Exception:
        return "fully_rewrite", None


def normalize_code(code: str) -> str:
    """
    Normalize code by removing comments and normalizing whitespace.
    This allows for comparison that tolerates comment and whitespace differences.

    Note: This uses regex-based heuristics and may incorrectly handle
    comment-like patterns inside string literals (e.g., "http://url" or "# not a comment").
    For most code comparison tasks, this is an acceptable trade-off.
    """
    # Remove multi-line comments first (before single-line to handle edge cases properly)
    # C-style /* */ comments
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    # Python docstrings / multi-line strings used as comments
    code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", "", code, flags=re.DOTALL)
    # HTML/XML comments
    code = re.sub(r"<!--.*?-->", "", code, flags=re.DOTALL)

    # Remove single-line comments (// for C-like languages, # for Python/shell/etc.)
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)

    # Normalize all whitespace (spaces, tabs, newlines) to single space
    # This collapses the code into a single line, ignoring all formatting differences
    code = re.sub(r"\s+", " ", code)

    # Strip leading/trailing whitespace
    code = code.strip()

    return code


def load_from_jsonl(file_path: str) -> list[dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_to_jsonl(data: list[dict[str, Any]], file_path: str) -> None:
    """Save data to JSONL file."""
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
