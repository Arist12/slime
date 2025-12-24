import difflib
import json
import re
from typing import Any
from transformers import AutoTokenizer


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


def extract_model_code(response: str, original_code: str) -> tuple[str | None, str | None]:
    """
    Extract model code from response.
    Handles both targeted edits (non-empty SEARCH) and full rewrites (empty SEARCH).
    Returns: (edit_mode, extracted_code) or (None, None) on failure.
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
        if first_search.strip() == "":
            # Full rewrite mode: must have exactly one SEARCH/REPLACE block
            if len(matches) > 1:
                return None, None
            return "fully_rewrite", first_replace.rstrip("\n")

        # Targeted edits: Apply each SEARCH/REPLACE block
        modified_code = original_code
        for match in matches:
            search_block = match.group(1).rstrip("\n")
            replace_block = match.group(2).rstrip("\n")

            # Check if search block exists in the current code
            if search_block not in modified_code:
                return None, None

            # Check for multiple occurrences (ambiguous, should error)
            occurrences = modified_code.count(search_block)
            if occurrences > 1:
                return None, None

            # Apply the replacement: replace only the first occurrence
            modified_code = modified_code.replace(search_block, replace_block, 1)

        return "find_replace", modified_code

    except Exception:
        return None, None


def code_edit_grader(response: str, original_code: str) -> tuple[str, bool, str]:
    """
    Determine the format used and whether it was successful.
    Returns: (edit_mode, format_success, extracted_code)
    """
    edit_mode, extracted_code = extract_model_code(response, original_code)
    format_success = extracted_code is not None
    return edit_mode, format_success, extracted_code


def calculate_token_consumption(tokenizer: AutoTokenizer, response: str) -> int:
    """Calculate the number of tokens in the response."""
    return len(tokenizer.encode(response))

def get_qwen_tokenizer() -> AutoTokenizer:
    """Get the qwen3-8b tokenizer."""
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")


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


def display_side_by_side_diff(ground_truth: str, model_code: str, width: int = 250):
    """Display differences side by side with improved readability."""

    gt_lines = ground_truth.splitlines()
    model_lines = model_code.splitlines()

    # Get the diff operations
    matcher = difflib.SequenceMatcher(None, gt_lines, model_lines)

    print(f"\n{'=' * (width + 10)}")
    print("SIDE-BY-SIDE COMPARISON")
    print(f"{'=' * (width + 10)}")

    # Headers
    col_width = width // 2 - 2
    print(f"{'Ground Truth':<{col_width}} | {'Model Output':<{col_width}}")
    print(f"{'-' * col_width} | {'-' * col_width}")

    changes_found = False

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # Show a few context lines for equal sections
            for i in range(min(3, i2 - i1)):  # Show max 3 context lines
                if i1 + i < len(gt_lines) and j1 + i < len(model_lines):
                    gt_line = gt_lines[i1 + i][: col_width - 1]
                    model_line = model_lines[j1 + i][: col_width - 1]
                    print(f"{gt_line:<{col_width}} | {model_line:<{col_width}}")

            if i2 - i1 > 3:
                print(f"{'... (identical lines) ...':<{col_width}} | {'... (identical lines) ...':<{col_width}}")

        elif tag == "delete":
            changes_found = True
            print("\n--- DELETED from Ground Truth ---")
            for i in range(i1, i2):
                line = gt_lines[i][: col_width - 1]
                print(f"- {line:<{col_width - 2}} | {'':>{col_width}}")

        elif tag == "insert":
            changes_found = True
            print("\n+++ ADDED in Model Output +++")
            for j in range(j1, j2):
                line = model_lines[j][: col_width - 1]
                print(f"{'':>{col_width}} | + {line:<{col_width - 2}}")

        elif tag == "replace":
            changes_found = True
            print("\n~~~ CHANGED ~~~")
            max_lines = max(i2 - i1, j2 - j1)
            for k in range(max_lines):
                gt_line = gt_lines[i1 + k][: col_width - 1] if i1 + k < i2 else ""
                model_line = model_lines[j1 + k][: col_width - 1] if j1 + k < j2 else ""

                gt_prefix = "- " if i1 + k < i2 else "  "
                model_prefix = "+ " if j1 + k < j2 else "  "

                print(f"{gt_prefix}{gt_line:<{col_width - 2}} | {model_prefix}{model_line:<{col_width - 2}}")

    if not changes_found:
        print("No differences found!")
