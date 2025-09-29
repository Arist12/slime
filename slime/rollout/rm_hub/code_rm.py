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


def code_rm(response: str, label: str, metadata: dict) -> float:
    """
    Reward model for adaptive code editing that supports both find_replace and fully_rewrite formats.

    Args:
        response: Model's response containing either find_replace blocks or fully rewritten code
        label: Ground truth code (the expected final result)
        metadata: Dictionary containing 'original_code' and 'language'

    Returns:
        Float score between 0.0 and 1.0
    """
    # Assert metadata has required keys
    assert "original_code" in metadata, "original_code must be in metadata"

    original_code = metadata["original_code"]

    if not response or not label:
        return 0.0

    # Determine which format the model used and extract the modified code
    modified_code, format_used = extract_modified_code(response, original_code)

    if modified_code is None:
        # Both formats failed or invalid format
        return 0.0

    # Calculate accuracy score based on similarity to ground truth
    accuracy_score = calculate_accuracy_score(modified_code, label)

    # Calculate token efficiency score
    token_efficiency_score = calculate_token_efficiency(response, label)

    # Combine scores with weighted average
    final_score = 0.9 * accuracy_score + 0.1 * token_efficiency_score

    return max(0.0, min(1.0, final_score))


def extract_model_code(response: str, original_code: str) -> str | None:
    """
    Extract model code from response.
    Handles both targeted edits (non-empty SEARCH) and full rewrites (empty SEARCH).
    """
    try:
        # Pattern to match SEARCH/REPLACE blocks
        pattern = r"<SEARCH>\n(.*?)</SEARCH>\n<REPLACE>\n(.*?)</REPLACE>"
        matches = re.findall(pattern, response, re.DOTALL)

        if not matches:
            return None

        first_search, first_replace = matches[0]
        if first_search == "":
            # Full rewrite mode: must have exactly one SEARCH/REPLACE block
            if len(matches) > 1:
                return None
            return first_replace

        # Targeted edits: Apply each SEARCH/REPLACE block
        modified_code = original_code
        for search_text, replace_text in matches:
            # Check if search text exists in the current code
            if search_text not in modified_code:
                return None

            # Apply the replacement: Use replace with count=1 to replace only the first occurrence
            modified_code = modified_code.replace(search_text, replace_text, 1)

        return modified_code

    except Exception:
        return None


def extract_modified_code(response: str, original_code: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract modified code from response, trying both find_replace and fully_rewrite formats.

    Returns:
        Tuple of (modified_code, format_used) where format_used is 'find_replace', 'fully_rewrite', or None
    """
    # First, try to detect and apply find_replace format
    find_replace_result = try_find_replace_format(response, original_code)
    if find_replace_result is not None:
        return find_replace_result, "find_replace"

    # If find_replace failed, try fully_rewrite format
    fully_rewrite_result = try_fully_rewrite_format(response)
    if fully_rewrite_result is not None:
        return fully_rewrite_result, "fully_rewrite"

    # Both formats failed
    return None, None


def try_find_replace_format(response: str, original_code: str, strict: bool = True) -> Optional[str]:
    """
    Try to extract and apply find_replace blocks from the response.

    Returns:
        Modified code if successful, None if failed
    """
    try:
        # Pattern to match SEARCH/REPLACE blocks
        pattern = r"<SEARCH>\s*(.*?)\s*</SEARCH>\s*<REPLACE>\s*(.*?)\s*</REPLACE>"
        matches = re.findall(pattern, response, re.DOTALL)

        if not matches:
            return None

        # Apply each SEARCH/REPLACE block
        modified_code = original_code
        for search_text, replace_text in matches:
            # Check if search text exists in the current code
            if search_text not in modified_code:
                # If any search block fails, the entire find_replace fails
                if strict:
                    return None
                else:
                    continue

            # Apply the replacement: Use replace with count=1 to replace only the first occurrence
            modified_code = modified_code.replace(search_text, replace_text, 1)

        return modified_code

    except Exception:
        return None


def try_fully_rewrite_format(response: str) -> Optional[str]:
    """
    Try to extract fully rewritten code from the response.

    Returns:
        Extracted code if successful, None if failed
    """
    try:
        # Pattern to match code blocks (with or without language specification)
        pattern = r"```(?:\w+)?\s*\n(.*?)\n```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            # Return the first (and hopefully only) code block
            return matches[0].strip()
        else:
            # If no code blocks found, return empty string
            return None

    except Exception:
        return None


def calculate_accuracy_score(modified_code: str, ground_truth: str) -> float:
    """
    Calculate accuracy score based on similarity between modified code and ground truth.
    """

    # exact match
    if modified_code == ground_truth:
        return 1.0

    # Normalize both codes for comparison
    normalized_modified = normalize_code(modified_code)
    normalized_ground_truth = normalize_code(ground_truth)

    if normalized_modified == normalized_ground_truth:
        return 0.95  # High score for normalized match

    # Use a combination of different similarity metrics

    # 1. Line-based similarity (good for code structure)
    modified_lines = modified_code.splitlines()
    ground_truth_lines = ground_truth.splitlines()
    line_similarity = difflib.SequenceMatcher(None, modified_lines, ground_truth_lines).ratio()

    # 2. Character-based similarity (good for small changes)
    char_similarity = difflib.SequenceMatcher(None, modified_code, ground_truth).ratio()

    # 3. Word-based similarity (good for semantic similarity)
    modified_words = modified_code.split()
    ground_truth_words = ground_truth.split()
    word_similarity = difflib.SequenceMatcher(None, modified_words, ground_truth_words).ratio()

    # 4. Normalized similarity (ignoring formatting differences)
    normalized_similarity = difflib.SequenceMatcher(None, normalized_modified, normalized_ground_truth).ratio()

    # Weighted combination of similarities, should be smaller than 0.95
    # Give more weight to structural (line-based) and semantic (word-based) similarities
    combined_similarity = (
        0.3 * line_similarity + 0.2 * char_similarity + 0.3 * word_similarity + 0.2 * normalized_similarity
    ) * 0.95

    # Apply a non-linear transformation to make the score more discriminative
    # This helps spread out scores that would otherwise cluster near 1.0
    if combined_similarity > 0.9:
        # High similarity region: linear mapping from [0.9, 1.0] to [0.8, 1.0]
        score = 0.8 + 0.2 * (combined_similarity - 0.9) / 0.1
    elif combined_similarity > 0.7:
        # Medium-high similarity: linear mapping from [0.7, 0.9] to [0.5, 0.8]
        score = 0.5 + 0.3 * (combined_similarity - 0.7) / 0.2
    else:
        # Lower similarity: more aggressive penalty
        score = 0.5 * (combined_similarity / 0.7) ** 2

    return max(0.0, min(1.0, score))


def calculate_token_efficiency(response: str, ground_truth: str) -> float:
    """
    Calculate token efficiency score. Rewards shorter responses without penalizing correctness.

    Args:
        response: Full model response
        ground_truth: Ground truth code (not the full label, just the expected result)

    Returns:
        Score between 0.0 and 1.0, where 1.0 is most efficient
    """
    try:
        # Tokenize both response and ground truth
        response_tokens = len(tokenizer.encode(response))
        ground_truth_tokens = len(tokenizer.encode(ground_truth))

        if ground_truth_tokens == 0:
            return 0.5  # Neutral score for edge case

        # Calculate token ratio
        token_ratio = response_tokens / ground_truth_tokens

        # Penalize responses that are excessively long
        if token_ratio <= 2.0:
            # Good range - give higher scores for shorter responses
            score = 1.0 - 0.1 * max(0, token_ratio - 1.0)  # Slight bonus for being close to ground truth length
        elif token_ratio <= 3.0:
            # Getting too long - linear penalty
            score = 0.85 - 0.5 * (token_ratio - 1.5) / 1.5
        else:
            # Way too long - more aggressive penalty
            score = 0.35 * (1.0 / (token_ratio - 2.0))

        return max(0.0, min(1.0, score))

    except Exception:
        return 0.5  # Neutral score on error


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

    # Normalize whitespace: replace multiple spaces/tabs with single space
    code = re.sub(r"\s+", " ", code)
    code = re.sub(r" +$", "", code, flags=re.MULTILINE)

    # Remove empty lines
    code = re.sub(r"\n\s*\n", "\n", code)

    # Strip leading/trailing whitespace
    code = code.strip()

    return code
