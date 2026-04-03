"""
Cell-level DataFrame grading engine for DataGym.

Compares an agent's DataFrame against a clean ground truth and returns
a score in [0.0, 1.0] with granular partial credit.

Score breakdown:
  - 20% column structure (right columns exist, right count)
  - 10% row count (correct number of rows)
  - 70% cell accuracy (percentage of cells matching expected)
"""

import math
from typing import Optional

import numpy as np
import pandas as pd


def cells_match(actual, expected, tolerance: float = 1e-6) -> bool:
    """Compare two cell values with type-aware fuzzy matching."""
    if actual is None and expected is None:
        return True
    if pd.isna(actual) and pd.isna(expected):
        return True
    if pd.isna(actual) or pd.isna(expected):
        return False

    # Numeric comparison with tolerance
    if isinstance(expected, (int, float, np.integer, np.floating)):
        try:
            actual_num = float(actual)
            expected_num = float(expected)
            if math.isnan(actual_num) and math.isnan(expected_num):
                return True
            return abs(actual_num - expected_num) < max(tolerance, abs(expected_num) * tolerance)
        except (ValueError, TypeError):
            return False

    # String comparison — normalize whitespace and case-insensitive for close matches
    actual_str = str(actual).strip()
    expected_str = str(expected).strip()

    if actual_str == expected_str:
        return True

    # Case-insensitive match
    if actual_str.lower() == expected_str.lower():
        return True

    # Numeric strings: "42.0" == "42" == "42.00"
    try:
        if float(actual_str) == float(expected_str):
            return True
    except (ValueError, TypeError):
        pass

    # Date normalization: compare parsed dates
    for fmt_group in [
        ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y", "%m-%d-%Y"],
    ]:
        actual_date = _try_parse_date(actual_str, fmt_group)
        expected_date = _try_parse_date(expected_str, fmt_group)
        if actual_date is not None and expected_date is not None:
            return actual_date == expected_date

    return False


def _try_parse_date(s: str, formats: list) -> Optional[str]:
    """Try to parse a date string with multiple formats, return ISO date or None."""
    from datetime import datetime
    for fmt in formats:
        try:
            return datetime.strptime(s.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Try pandas for more flexible parsing
    try:
        return pd.to_datetime(s).strftime("%Y-%m-%d")
    except Exception:
        return None


def grade_dataframe(
    result_df: pd.DataFrame,
    expected_df: pd.DataFrame,
    column_weight: float = 0.20,
    row_weight: float = 0.10,
    cell_weight: float = 0.70,
) -> float:
    """
    Grade a DataFrame against expected output.

    Returns a score in [0.0, 1.0] with partial credit for:
    - Column structure (names match)
    - Row count (right number of rows)
    - Cell accuracy (individual cell values match)
    """
    score = 0.0

    # ── Column structure (20%) ──
    expected_cols = set(expected_df.columns)
    result_cols = set(result_df.columns)

    if not expected_cols:
        score += column_weight
    else:
        overlap = len(expected_cols & result_cols)
        col_score = overlap / len(expected_cols)
        # Penalty for extra columns (mild)
        extra = len(result_cols - expected_cols)
        col_penalty = min(0.1, extra * 0.02)
        score += column_weight * max(0, col_score - col_penalty)

    # ── Row count (10%) ──
    if len(expected_df) == 0:
        score += row_weight if len(result_df) == 0 else 0
    else:
        row_diff = abs(len(result_df) - len(expected_df)) / len(expected_df)
        score += row_weight * max(0, 1.0 - row_diff)

    # ── Cell accuracy (70%) ──
    shared_cols = sorted(expected_cols & result_cols)
    if not shared_cols:
        return min(score, 1.0)

    # Align rows: try index-based comparison first
    min_rows = min(len(result_df), len(expected_df))
    if min_rows == 0:
        return min(score, 1.0)

    total_cells = len(shared_cols) * len(expected_df)
    matching = 0

    for col in shared_cols:
        for i in range(min_rows):
            try:
                actual_val = result_df[col].iloc[i]
                expected_val = expected_df[col].iloc[i]
                if cells_match(actual_val, expected_val):
                    matching += 1
            except (IndexError, KeyError):
                pass

    cell_score = matching / max(total_cells, 1)
    score += cell_weight * cell_score

    return min(score, 1.0)


def strict_cells_match(actual, expected) -> bool:
    """Strict cell comparison — exact string match, no date parsing, no case folding."""
    if actual is None and expected is None:
        return True
    if pd.isna(actual) and pd.isna(expected):
        return True
    if pd.isna(actual) or pd.isna(expected):
        return False

    if isinstance(expected, (int, float, np.integer, np.floating)):
        try:
            actual_num = float(actual)
            expected_num = float(expected)
            if math.isnan(actual_num) and math.isnan(expected_num):
                return True
            return abs(actual_num - expected_num) < max(1e-6, abs(expected_num) * 1e-6)
        except (ValueError, TypeError):
            return False

    return str(actual).strip() == str(expected).strip()


def grade_dataframe_strict(
    result_df: pd.DataFrame,
    expected_df: pd.DataFrame,
    column_weight: float = 0.20,
    row_weight: float = 0.10,
    cell_weight: float = 0.70,
) -> float:
    """Grade using strict string matching — no case folding or date parsing."""
    score = 0.0

    expected_cols = set(expected_df.columns)
    result_cols = set(result_df.columns)

    if not expected_cols:
        score += column_weight
    else:
        overlap = len(expected_cols & result_cols)
        col_score = overlap / len(expected_cols)
        extra = len(result_cols - expected_cols)
        col_penalty = min(0.1, extra * 0.02)
        score += column_weight * max(0, col_score - col_penalty)

    if len(expected_df) == 0:
        score += row_weight if len(result_df) == 0 else 0
    else:
        row_diff = abs(len(result_df) - len(expected_df)) / len(expected_df)
        score += row_weight * max(0, 1.0 - row_diff)

    shared_cols = sorted(expected_cols & result_cols)
    if not shared_cols:
        return min(score, 1.0)

    min_rows = min(len(result_df), len(expected_df))
    if min_rows == 0:
        return min(score, 1.0)

    total_cells = len(shared_cols) * len(expected_df)
    matching = 0

    for col in shared_cols:
        for i in range(min_rows):
            try:
                actual_val = result_df[col].iloc[i]
                expected_val = expected_df[col].iloc[i]
                if strict_cells_match(actual_val, expected_val):
                    matching += 1
            except (IndexError, KeyError):
                pass

    cell_score = matching / max(total_cells, 1)
    score += cell_weight * cell_score

    return min(score, 1.0)


def score_breakdown(result_df: pd.DataFrame, expected_df: pd.DataFrame) -> str:
    """Human-readable breakdown of where the result differs from expected."""
    lines = []
    expected_cols = set(expected_df.columns)
    result_cols = set(result_df.columns)

    missing = expected_cols - result_cols
    extra = result_cols - expected_cols
    if missing:
        lines.append(f"Missing columns: {sorted(missing)}")
    if extra:
        lines.append(f"Extra columns (not expected): {sorted(extra)}")

    row_diff = len(result_df) - len(expected_df)
    if row_diff != 0:
        lines.append(f"Row count: {len(result_df)} (expected {len(expected_df)}, {'extra' if row_diff > 0 else 'missing'} {abs(row_diff)})")

    shared_cols = sorted(expected_cols & result_cols)
    min_rows = min(len(result_df), len(expected_df))
    mismatches_by_col = {}
    for col in shared_cols:
        bad = 0
        first_bad = None
        for i in range(min_rows):
            try:
                actual = result_df[col].iloc[i]
                expected = expected_df[col].iloc[i]
                if not cells_match(actual, expected):
                    bad += 1
                    if first_bad is None:
                        first_bad = (i, repr(actual), repr(expected))
            except (IndexError, KeyError):
                bad += 1
        if bad > 0:
            mismatches_by_col[col] = (bad, first_bad)

    if mismatches_by_col:
        lines.append(f"Cell mismatches in {len(mismatches_by_col)}/{len(shared_cols)} columns:")
        for col, (count, example) in sorted(mismatches_by_col.items()):
            row_i, got, want = example
            lines.append(f"  '{col}': {count} wrong (row {row_i}: got {got}, expected {want})")
    elif shared_cols and min_rows > 0:
        lines.append("All cells match correctly!")

    return "\n".join(lines) if lines else "Score breakdown unavailable"


def describe_issues(df: pd.DataFrame) -> str:
    """Generate a human-readable summary of data quality issues."""
    issues = []

    for col in df.columns:
        nulls = df[col].isna().sum()
        if nulls > 0:
            issues.append(f"Column '{col}': {nulls} null values ({nulls*100//len(df)}%)")

        if df[col].dtype == object:
            sample = df[col].dropna()
            if len(sample) > 0:
                # Check for mixed types
                types = set()
                for v in sample.head(50):
                    try:
                        float(str(v).replace(",", "").replace("$", "").replace("%", ""))
                        types.add("numeric_string")
                    except ValueError:
                        types.add("text")
                if len(types) > 1:
                    issues.append(f"Column '{col}': mixed content (numbers stored as text)")

                # Check for inconsistent casing
                lower_unique = len(sample.str.lower().unique())
                raw_unique = len(sample.unique())
                if lower_unique < raw_unique:
                    issues.append(f"Column '{col}': inconsistent casing ({raw_unique} unique vs {lower_unique} case-insensitive)")

    # Check for duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append(f"{dup_count} duplicate rows detected")

    if not issues:
        issues.append("No obvious issues detected — check data types and values carefully")

    return "\n".join(issues)


def column_info_str(df: pd.DataFrame) -> str:
    """Generate column info summary."""
    lines = []
    for col in df.columns:
        nulls = df[col].isna().sum()
        dtype = str(df[col].dtype)
        unique = df[col].nunique()
        lines.append(f"  {col}: {dtype}, {nulls} nulls, {unique} unique")
    return "\n".join(lines)


def target_schema_str(expected_df: pd.DataFrame) -> str:
    """Describe the expected schema."""
    lines = []
    for col in expected_df.columns:
        dtype = str(expected_df[col].dtype)
        lines.append(f"  {col}: {dtype}")
    return "\n".join(lines)
