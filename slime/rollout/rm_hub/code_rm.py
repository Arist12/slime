import difflib
import re
from typing import Optional, Tuple

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")


def code_rm_simple(response: str, label: str, metadata: dict) -> float:
    """
    Reward model for adaptive code editing that supports both find_replace and fully_rewrite formats.

    Args:
        response: Model's response containing either find_replace blocks or fully rewritten code
        label: Ground truth code (the expected final result)
        metadata: Dictionary containing 'original_code' and 'language'

    Returns:
        Float score between 0.0 and 1.0
    """
    original_code = metadata["original_code"]

    if not response or not label:
        return 0.0

    # Determine which format the model used and extract the modified code
    modified_code = extract_model_code(response, original_code)

    if modified_code is None:
        return 0.0

    if modified_code == label or normalize_code(modified_code) == normalize_code(label):
        return 1.0

    return 0


def extract_model_code(response: str, original_code: str) -> str | None:
    """
    Extract model code from response.
    Handles both targeted edits (non-empty SEARCH) and full rewrites (empty SEARCH).
    """
    try:
        # Pattern to match SEARCH/REPLACE blocks (aligned with LLMFileEditor format)
        pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
        matches = list(re.finditer(pattern, response, re.DOTALL))

        if not matches:
            return None

        # Extract first search and replace blocks
        first_search = matches[0].group(1)
        first_replace = matches[0].group(2)

        # Check if this is a full rewrite (empty search block)
        if first_search.strip() == "":
            # Full rewrite mode: must have exactly one SEARCH/REPLACE block
            if len(matches) > 1:
                return None
            return first_replace.rstrip("\n")

        # Targeted edits: Apply each SEARCH/REPLACE block
        modified_code = original_code
        for match in matches:
            search_block = match.group(1).rstrip("\n")
            replace_block = match.group(2).rstrip("\n")

            # Check if search block exists in the current code
            if search_block not in modified_code:
                return None

            # Check for multiple occurrences (ambiguous, should error)
            occurrences = modified_code.count(search_block)
            if occurrences > 1:
                return None

            # Apply the replacement: replace only the first occurrence
            modified_code = modified_code.replace(search_block, replace_block, 1)

        return modified_code

    except Exception:
        return None


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
