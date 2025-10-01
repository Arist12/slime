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


def extract_model_code(response: str, original_code: str):
    """
    Extract model code from response.
    Handles both targeted edits (non-empty SEARCH) and full rewrites (empty SEARCH).
    """
    try:
        # Pattern to match SEARCH/REPLACE blocks
        pattern = r"<SEARCH>\n(.*?)</SEARCH>\n<REPLACE>\n(.*?)</REPLACE>"
        matches = re.findall(pattern, response, re.DOTALL)

        if not matches:
            return "failed", None

        first_search, first_replace = matches[0]
        if first_search == "":
            # Full rewrite mode: must have exactly one SEARCH/REPLACE block
            if len(matches) > 1:
                return "fully_rewrite", None
            return "fully_rewrite", first_replace

        # Targeted edits: Apply each SEARCH/REPLACE block
        modified_code = original_code
        for search_text, replace_text in matches:
            # Check if search text exists in the current code
            if search_text not in modified_code:
                return "find_replace", None

            # Apply the replacement: Use replace with count=1 to replace only the first occurrence
            modified_code = modified_code.replace(search_text, replace_text, 1)

        return "find_replace", modified_code

    except Exception:
        return "failed", None


def extract_code_from_response(response: str) -> str:
    """Extract code from markdown code blocks."""
    # Handle None or non-string responses
    if response is None:
        return ""

    pattern = r"```(?:\w+)?\s*\n(.*?)\n```"

    matches = re.findall(pattern, response, re.DOTALL)

    if matches:
        return matches[0]
    else:
        return response


def normalize_code(code: str) -> str:
    """
    Normalize code by removing comments and normalizing whitespace.
    This allows for comparison that tolerates comment and whitespace differences.
    """
    # Remove single-line comments (// and #)
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)

    # Remove multi-line comments (/* */ and """ """)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", "", code, flags=re.DOTALL)

    # Remove docstrings (strings at the beginning of functions/classes)
    # This is a simplified approach - for more robust handling, you might need AST parsing
    code = re.sub(r'^\s*""".*?"""\s*$', "", code, flags=re.MULTILINE | re.DOTALL)
    code = re.sub(r"^\s*'''.*?'''\s*$", "", code, flags=re.MULTILINE | re.DOTALL)

    # Normalize whitespace: replace multiple spaces/tabs with single space, remove trailing spaces
    code = re.sub(r"\s+", " ", code)
    code = re.sub(r" +$", "", code, flags=re.MULTILINE)

    # Remove empty lines
    code = re.sub(r"\n\s*\n", "\n", code)

    # Strip leading/trailing whitespace
    code = code.strip()

    return code


def determine_format_and_success_with_extract_model_code(response: str, original_code: str) -> tuple[str, bool, str]:
    """
    Determine the format used and whether it was successful.
    Returns: (format_used, format_success, extracted_code)
    """
    format_used, extracted_code = extract_model_code(response, original_code)
    return format_used, extracted_code is not None, extracted_code


def determine_format_and_success(response: str, original_code: str) -> tuple[str, bool, str]:
    """
    Determine the format used and whether it was successful.
    Returns: (format_used, format_success, extracted_code)
    """
    # Try find_replace format first
    find_replace_result = try_find_replace_format(response, original_code)
    if find_replace_result is not None:
        if find_replace_result == "failed":
            return "find_replace", False, ""
        else:
            return "find_replace", True, find_replace_result

    # Try fully_rewrite format
    fully_rewrite_result = try_fully_rewrite_format(response)
    if fully_rewrite_result is not None:
        return "fully_rewrite", True, fully_rewrite_result

    # Neither format worked
    return "unknown", False, ""



def try_find_replace_format(response: str, original_code: str) -> str:
    """Try to extract and apply find_replace blocks from the response."""
    try:
        # Pattern to match SEARCH/REPLACE blocks
        pattern = r"<SEARCH>\s*(.*?)\s*</SEARCH>\s*<REPLACE>\s*(.*?)\s*</REPLACE>"
        matches = re.findall(pattern, response, re.DOTALL)

        if not matches:
            return None

        # Apply each SEARCH/REPLACE block
        modified_code = original_code
        for search_text, replace_text in matches:
            if search_text not in modified_code:
                return "failed"

            # Apply the replacement: Use replace with count=1 to replace only the first occurrence
            modified_code = modified_code.replace(search_text, replace_text, 1)

        return modified_code

    except Exception:
        return None


def try_fully_rewrite_format(response: str) -> str:
    """Try to extract fully rewritten code from the response."""
    try:
        # Pattern to match code blocks (with or without language specification)
        pattern = r"```(?:\w+)?\s*\n(.*?)\n```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()
        else:
            return None

    except Exception:
        return None


def calculate_token_consumption(tokenizer: AutoTokenizer, response: str) -> int:
    """Calculate the number of tokens in the response."""
    return len(tokenizer.encode(response))


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
