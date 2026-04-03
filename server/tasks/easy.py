"""Easy tasks — single-issue datasets, clear cleaning objectives."""

import pandas as pd
import numpy as np
from .registry import Task, register_task
from ..grading import grade_dataframe_strict


# ── E1: Fix numeric types (dollar signs, commas, percentages) ────────────────

def _dirty_e1():
    return pd.DataFrame({
        "product": ["Widget A", "Widget B", "Gadget X", "Gadget Y", "Tool Z",
                     "Widget C", "Gadget W", "Tool Q"],
        "price": ["$29.99", "$45.50", "12.99", "$89.00", "$ 15.75",
                   "$120.00", "67.50", "$34.25"],
        "quantity": ["1,000", "500", "2,500", "100", "3,000",
                     "750", "1,200", "800"],
        "discount": ["10%", "5%", "15%", "0%", "20%",
                     "12.5%", "8%", "0%"],
    })

def _clean_e1():
    return pd.DataFrame({
        "product": ["Widget A", "Widget B", "Gadget X", "Gadget Y", "Tool Z",
                     "Widget C", "Gadget W", "Tool Q"],
        "price": [29.99, 45.50, 12.99, 89.00, 15.75, 120.00, 67.50, 34.25],
        "quantity": [1000, 500, 2500, 100, 3000, 750, 1200, 800],
        "discount": [0.10, 0.05, 0.15, 0.00, 0.20, 0.125, 0.08, 0.00],
    })

register_task(Task(
    task_id="e1_fix_numeric_types",
    difficulty="easy",
    description=(
        "Clean the numeric columns in this product dataset.\n"
        "- 'price': Remove dollar signs and spaces, convert to float\n"
        "- 'quantity': Remove commas, convert to integer\n"
        "- 'discount': Remove percent signs, convert to decimal (e.g., '10%' → 0.10)"
    ),
    hint="Use str.replace() to remove symbols, then astype() to convert types.",
    max_steps=5,
    setup_dirty=_dirty_e1,
    setup_clean=_clean_e1,
))


# ── E2: Handle null/missing values ──────────────────────────────────────────

def _dirty_e2():
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
                 "Grace", "Heidi", "Ivan", "Judy"],
        "age": [28.0, np.nan, 35.0, 42.0, np.nan, 31.0,
                np.nan, 29.0, 45.0, np.nan],
        "email": ["alice@co.com", "bob@co.com", "N/A", "diana@co.com",
                  "eve@co.com", "null", "grace@co.com", "--",
                  "ivan@co.com", "None"],
        "department": ["Engineering", "Marketing", "Engineering", "Sales",
                       "Engineering", "Marketing", "Sales", "Engineering",
                       "Sales", "Marketing"],
    })

def _clean_e2():
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
                 "Grace", "Heidi", "Ivan", "Judy"],
        "age": [28.0, 33.0, 35.0, 42.0, 33.0, 31.0,
                33.0, 29.0, 45.0, 33.0],
        "email": ["alice@co.com", "bob@co.com", None, "diana@co.com",
                  "eve@co.com", None, "grace@co.com", None,
                  "ivan@co.com", None],
        "department": ["Engineering", "Marketing", "Engineering", "Sales",
                       "Engineering", "Marketing", "Sales", "Engineering",
                       "Sales", "Marketing"],
    })

register_task(Task(
    task_id="e2_handle_nulls",
    difficulty="easy",
    description=(
        "Clean the missing values in this employee dataset.\n"
        "- 'age': Fill missing ages with the median of non-null ages\n"
        "- 'email': Values like 'N/A', 'null', 'None', '--' are actually missing — "
        "replace them with proper NaN/None values\n"
        "- Keep all rows, don't drop any"
    ),
    hint="Use df.replace() with a list of null-like strings, then fillna() for age.",
    max_steps=5,
    setup_dirty=_dirty_e2,
    setup_clean=_clean_e2,
))


# ── E3: Standardize date formats ────────────────────────────────────────────

def _dirty_e3():
    return pd.DataFrame({
        "event": ["Launch", "Review", "Deploy", "Standup", "Retro",
                  "Planning", "Demo", "Sync"],
        "date": ["01/15/2024", "2024-02-20", "15-03-2024", "2024.04.10",
                 "05/25/2024", "2024-06-01", "07-15-2024", "2024.08.30"],
        "attendees": [12, 5, 8, 15, 10, 7, 20, 6],
    })

def _clean_e3():
    return pd.DataFrame({
        "event": ["Launch", "Review", "Deploy", "Standup", "Retro",
                  "Planning", "Demo", "Sync"],
        "date": ["2024-01-15", "2024-02-20", "2024-03-15", "2024-04-10",
                 "2024-05-25", "2024-06-01", "2024-07-15", "2024-08-30"],
        "attendees": [12, 5, 8, 15, 10, 7, 20, 6],
    })

register_task(Task(
    task_id="e3_standardize_dates",
    difficulty="easy",
    description=(
        "Standardize all dates in the 'date' column to ISO 8601 format (YYYY-MM-DD).\n"
        "Current formats include: MM/DD/YYYY, YYYY-MM-DD, DD-MM-YYYY, YYYY.MM.DD.\n"
        "The result should be strings in 'YYYY-MM-DD' format."
    ),
    hint="pd.to_datetime() can parse mixed formats. Use .dt.strftime('%Y-%m-%d') to format.",
    max_steps=5,
    setup_dirty=_dirty_e3,
    setup_clean=_clean_e3,
    custom_grade=grade_dataframe_strict,
))


# ── E4: Fix inconsistent text casing and whitespace ─────────────────────────

def _dirty_e4():
    return pd.DataFrame({
        "name": ["  Alice Johnson ", "BOB SMITH", "charlie brown", "Diana Lee",
                 "EVE DAVIS", " frank Miller", "GRACE wilson ", "heidi CLARK"],
        "city": ["new york", "LOS ANGELES", "Chicago", "  san francisco  ",
                 "BOSTON", "seattle", "Denver", "MIAMI"],
        "category": ["electronics", "CLOTHING", "Electronics", "clothing",
                     "ELECTRONICS", "Clothing", "electronics", "CLOTHING"],
    })

def _clean_e4():
    return pd.DataFrame({
        "name": ["Alice Johnson", "Bob Smith", "Charlie Brown", "Diana Lee",
                 "Eve Davis", "Frank Miller", "Grace Wilson", "Heidi Clark"],
        "city": ["New York", "Los Angeles", "Chicago", "San Francisco",
                 "Boston", "Seattle", "Denver", "Miami"],
        "category": ["Electronics", "Clothing", "Electronics", "Clothing",
                     "Electronics", "Clothing", "Electronics", "Clothing"],
    })

register_task(Task(
    task_id="e4_fix_text",
    difficulty="easy",
    description=(
        "Clean the text columns in this dataset.\n"
        "- All columns: Strip leading/trailing whitespace\n"
        "- 'name': Convert to Title Case\n"
        "- 'city': Convert to Title Case\n"
        "- 'category': Standardize to Title Case (first letter uppercase, rest lowercase)"
    ),
    hint="Use .str.strip() for whitespace and .str.title() for casing.",
    max_steps=5,
    setup_dirty=_dirty_e4,
    setup_clean=_clean_e4,
    custom_grade=grade_dataframe_strict,
))


# ── E5: Remove exact duplicate rows ─────────────────────────────────────────

def _dirty_e5():
    return pd.DataFrame({
        "order_id": [101, 102, 103, 102, 104, 105, 103, 106, 107, 105, 108, 104],
        "customer": ["Alice", "Bob", "Charlie", "Bob", "Diana", "Eve",
                     "Charlie", "Frank", "Grace", "Eve", "Heidi", "Diana"],
        "amount": [250.0, 130.0, 89.50, 130.0, 420.0, 75.25,
                   89.50, 310.0, 195.0, 75.25, 560.0, 420.0],
        "status": ["shipped", "pending", "delivered", "pending", "shipped", "delivered",
                   "delivered", "pending", "shipped", "delivered", "pending", "shipped"],
    })

def _clean_e5():
    return pd.DataFrame({
        "order_id": [101, 102, 103, 104, 105, 106, 107, 108],
        "customer": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Heidi"],
        "amount": [250.0, 130.0, 89.50, 420.0, 75.25, 310.0, 195.0, 560.0],
        "status": ["shipped", "pending", "delivered", "shipped", "delivered", "pending", "shipped", "pending"],
    }).reset_index(drop=True)

register_task(Task(
    task_id="e5_remove_duplicates",
    difficulty="easy",
    description=(
        "Remove duplicate rows from this orders dataset.\n"
        "Duplicates have the same order_id, customer, amount, and status.\n"
        "Keep the first occurrence of each duplicate. Reset the index after removal."
    ),
    hint="Use df.drop_duplicates() and df.reset_index(drop=True).",
    max_steps=5,
    setup_dirty=_dirty_e5,
    setup_clean=_clean_e5,
))
