"""Medium tasks — multi-step cleaning, combining operations."""

import pandas as pd
import numpy as np
from .registry import Task, register_task


# ── M1: Split compound column + type fix ─────────────────────────────────────

def _dirty_m1():
    return pd.DataFrame({
        "full_name": ["Alice Johnson", "Bob Smith", "Charlie Brown",
                      "Diana Lee", "Eve Davis"],
        "location": ["New York, NY 10001", "Los Angeles, CA 90001",
                     "Chicago, IL 60601", "San Francisco, CA 94102",
                     "Boston, MA 02101"],
        "salary": ["$85,000", "$92,500", "$78,000", "$110,000", "$95,750"],
    })

def _clean_m1():
    return pd.DataFrame({
        "full_name": ["Alice Johnson", "Bob Smith", "Charlie Brown",
                      "Diana Lee", "Eve Davis"],
        "city": ["New York", "Los Angeles", "Chicago", "San Francisco", "Boston"],
        "state": ["NY", "CA", "IL", "CA", "MA"],
        "zip_code": ["10001", "90001", "60601", "94102", "02101"],
        "salary": [85000.0, 92500.0, 78000.0, 110000.0, 95750.0],
    })

register_task(Task(
    task_id="m1_split_and_clean",
    difficulty="medium",
    description=(
        "Transform this employee dataset:\n"
        "1. Split 'location' into three columns: 'city', 'state', 'zip_code'\n"
        "   Format is 'City, ST ZIPCODE'\n"
        "2. Remove the original 'location' column\n"
        "3. Clean 'salary': remove '$' and commas, convert to float\n"
        "4. Keep 'zip_code' as a string (leading zeros matter)"
    ),
    hint=None,
    max_steps=8,
    setup_dirty=_dirty_m1,
    setup_clean=_clean_m1,
))


# ── M2: Multi-source merge with conflict resolution ─────────────────────────

def _dirty_m2():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8],
        "name": ["Alice", "Bob", "CHARLIE", "diana", "Eve", "bob", "Alice", "frank"],
        "email": ["alice@co.com", "bob@co.com", "charlie@co.com", "diana@co.com",
                  "eve@co.com", "bob@co.com", "alice@co.com", "frank@co.com"],
        "score": [85, 72, 91, 68, 95, 78, 88, 62],
    })

def _clean_m2():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "name": ["Eve", "Charlie", "Alice", "Bob", "Diana", "Frank"],
        "email": ["eve@co.com", "charlie@co.com", "alice@co.com", "bob@co.com",
                  "diana@co.com", "frank@co.com"],
        "score": [95, 91, 88, 78, 68, 62],
    }).reset_index(drop=True)

register_task(Task(
    task_id="m2_dedup_merge",
    difficulty="medium",
    description=(
        "Clean this dataset with duplicate entries:\n"
        "1. Fix name casing: all names should be Title Case\n"
        "2. Find duplicates by email address (case-insensitive)\n"
        "3. For duplicates, keep the row with the HIGHER score\n"
        "4. Re-assign sequential IDs starting from 1\n"
        "5. Reset the index"
    ),
    hint=None,
    max_steps=8,
    setup_dirty=_dirty_m2,
    setup_clean=_clean_m2,
))


# ── M3: Fix data types + handle outliers ─────────────────────────────────────

def _dirty_m3():
    return pd.DataFrame({
        "product": ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard",
                    "Mouse", "Webcam", "Headset", "Speaker", "Cable"],
        "price": ["$999.99", "699", " 449.00", "329.50", "$79.99",
                  "29.99", "89.50", "$149.00", "59.99", "12.50"],
        "stock": [50, 200, -5, 100, -10, 999999, 75, -1, 150, 300],
        "rating": [4.5, "4.2", 3.8, "N/A", 4.7, 3.5, "N/A", "4.1", "N/A", "N/A"],
        "weight_kg": [2.1, -0.18, 0.5, 5.5, -0.8, 0.05, -0.15, -0.25, 1.2, 0.02],
    })

def _clean_m3():
    return pd.DataFrame({
        "product": ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard",
                    "Mouse", "Webcam", "Headset", "Speaker", "Cable"],
        "price": [999.99, 699.0, 449.0, 329.5, 79.99,
                  29.99, 89.5, 149.0, 59.99, 12.5],
        "stock": [50, 200, 0, 100, 0, 75, 75, 0, 150, 300],
        "rating": [4.5, 4.2, 3.8, np.nan, 4.7, 3.5, np.nan, 4.1, np.nan, np.nan],
        "weight_kg": [2.1, 0.18, 0.5, 5.5, 0.8, 0.05, 0.15, 0.25, 1.2, 0.02],
    })

register_task(Task(
    task_id="m3_types_and_outliers",
    difficulty="medium",
    description=(
        "Clean this product inventory dataset:\n"
        "1. 'price': Strip '$' signs, whitespace, and convert from string to float\n"
        "2. 'stock': Negative values should be 0; values > 10000 should be capped at the median of normal stock values\n"
        "3. 'rating': Convert 'N/A' to NaN, convert column to float\n"
        "4. 'weight_kg': Negative weights should be made positive (absolute value)"
    ),
    hint=None,
    max_steps=8,
    setup_dirty=_dirty_m3,
    setup_clean=_clean_m3,
))


# ── M4: Parse and restructure log data ───────────────────────────────────────

def _dirty_m4():
    return pd.DataFrame({
        "log_entry": [
            "2024-01-15 10:30:22 INFO User alice logged in from 192.168.1.10",
            "2024-01-15 10:31:05 ERROR Database connection failed: timeout",
            "2024-01-15 10:32:11 WARN Memory usage at 85%",
            "2024-01-15 10:33:00 INFO User bob logged in from 10.0.0.5",
            "2024-01-15 10:34:18 ERROR File not found: /data/config.yaml",
            "2024-01-15 10:35:42 INFO User alice logged out",
            "2024-01-15 10:36:30 WARN CPU usage at 92%",
            "2024-01-15 10:37:15 INFO Backup completed successfully",
        ],
    })

def _clean_m4():
    return pd.DataFrame({
        "timestamp": ["2024-01-15 10:30:22", "2024-01-15 10:31:05",
                      "2024-01-15 10:32:11", "2024-01-15 10:33:00",
                      "2024-01-15 10:34:18", "2024-01-15 10:35:42",
                      "2024-01-15 10:36:30", "2024-01-15 10:37:15"],
        "level": ["INFO", "ERROR", "WARN", "INFO", "ERROR", "INFO", "WARN", "INFO"],
        "message": [
            "User alice logged in from 192.168.1.10",
            "Database connection failed: timeout",
            "Memory usage at 85%",
            "User bob logged in from 10.0.0.5",
            "File not found: /data/config.yaml",
            "User alice logged out",
            "CPU usage at 92%",
            "Backup completed successfully",
        ],
    })

register_task(Task(
    task_id="m4_parse_logs",
    difficulty="medium",
    description=(
        "Parse the 'log_entry' column into structured columns.\n"
        "Each log entry has format: 'YYYY-MM-DD HH:MM:SS LEVEL Message text'\n"
        "Split into three columns:\n"
        "  - 'timestamp': the datetime string (keep as string)\n"
        "  - 'level': the log level (INFO, ERROR, WARN)\n"
        "  - 'message': the rest of the text\n"
        "Remove the original 'log_entry' column."
    ),
    hint=None,
    max_steps=8,
    setup_dirty=_dirty_m4,
    setup_clean=_clean_m4,
))


# ── M5: Wide to long format pivot ────────────────────────────────────────────

def _dirty_m5():
    return pd.DataFrame({
        "student": ["Alice", "Bob", "Charlie", "Diana"],
        "math_q1": [85, 72, 91, 68],
        "math_q2": [88, 75, 89, 72],
        "science_q1": [79, 85, 82, 90],
        "science_q2": [82, 88, 85, 93],
        "english_q1": [92, 68, 75, 88],
        "english_q2": [90, 71, 78, 91],
    })

def _clean_m5():
    rows = []
    data = {
        "Alice": {"english": [92, 90], "math": [85, 88], "science": [79, 82]},
        "Bob": {"english": [68, 71], "math": [72, 75], "science": [85, 88]},
        "Charlie": {"english": [75, 78], "math": [91, 89], "science": [82, 85]},
        "Diana": {"english": [88, 91], "math": [68, 72], "science": [90, 93]},
    }
    for student in sorted(data):
        for subject in sorted(data[student]):
            for q_idx, score in enumerate(data[student][subject], 1):
                rows.append({
                    "student": student,
                    "subject": subject,
                    "quarter": f"Q{q_idx}",
                    "score": score,
                })
    return pd.DataFrame(rows).reset_index(drop=True)

register_task(Task(
    task_id="m5_wide_to_long",
    difficulty="medium",
    description=(
        "Transform this wide-format grade dataset into long format.\n"
        "Current columns: student, math_q1, math_q2, science_q1, science_q2, english_q1, english_q2\n"
        "Target columns: student, subject, quarter, score\n"
        "Where subject is 'math'/'science'/'english' and quarter is 'Q1'/'Q2'.\n"
        "Sort by student (alphabetical), then subject, then quarter.\n"
        "Reset the index."
    ),
    hint=None,
    max_steps=8,
    setup_dirty=_dirty_m5,
    setup_clean=_clean_m5,
))
