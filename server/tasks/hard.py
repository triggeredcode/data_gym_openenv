"""Hard tasks — complex multi-step cleaning, cross-table operations."""

import pandas as pd
import numpy as np
from .registry import Task, register_task


# ── H1: Full pipeline — types + nulls + dedup + filter ───────────────────────

def _dirty_h1():
    return pd.DataFrame({
        "transaction_id": ["TXN001", "TXN002", "TXN003", "TXN004", "TXN005",
                          "TXN002", "TXN006", "TXN007", "TXN008", "TXN009",
                          "TXN010", "TXN005", "TXN011", "TXN012", "TXN013"],
        "date": ["01/15/2024", "2024-02-20", "15-03-2024", "2024.04.10",
                "05/25/2024", "2024-02-20", "2024-06-01", "07-15-2024",
                "2024.08.30", "09/10/2024", "2024-10-05", "05/25/2024",
                "11-20-2024", "2024.12.01", "2024-12-15"],
        "amount": ["$1,250.00", "850.50", "$2,100", "N/A", "$750.25",
                  "850.50", "$3,500.00", "1200", "null", "$425.75",
                  "$1,800.50", "$750.25", "$950.00", "None", "$2,250.00"],
        "category": ["electronics", "CLOTHING", "Electronics", "clothing",
                    "food", "CLOTHING", "electronics", "Food",
                    "ELECTRONICS", "clothing", "food", "food",
                    "CLOTHING", "Electronics", "Food"],
        "status": ["completed", "pending", "COMPLETED", "failed",
                  "Completed", "pending", "completed", "PENDING",
                  "completed", "Failed", "pending", "Completed",
                  "completed", "pending", "COMPLETED"],
    })

def _clean_h1():
    rows = [
        {"transaction_id": "TXN001", "date": "2024-01-15", "amount": 1250.00, "category": "Electronics", "status": "Completed"},
        {"transaction_id": "TXN002", "date": "2024-02-20", "amount": 850.50, "category": "Clothing", "status": "Pending"},
        {"transaction_id": "TXN003", "date": "2024-03-15", "amount": 2100.00, "category": "Electronics", "status": "Completed"},
        {"transaction_id": "TXN005", "date": "2024-05-25", "amount": 750.25, "category": "Food", "status": "Completed"},
        {"transaction_id": "TXN006", "date": "2024-06-01", "amount": 3500.00, "category": "Electronics", "status": "Completed"},
        {"transaction_id": "TXN007", "date": "2024-07-15", "amount": 1200.00, "category": "Food", "status": "Pending"},
        {"transaction_id": "TXN009", "date": "2024-09-10", "amount": 425.75, "category": "Clothing", "status": "Failed"},
        {"transaction_id": "TXN010", "date": "2024-10-05", "amount": 1800.50, "category": "Food", "status": "Pending"},
        {"transaction_id": "TXN011", "date": "2024-11-20", "amount": 950.00, "category": "Clothing", "status": "Completed"},
        {"transaction_id": "TXN013", "date": "2024-12-15", "amount": 2250.00, "category": "Food", "status": "Completed"},
    ]
    return pd.DataFrame(rows).reset_index(drop=True)

register_task(Task(
    task_id="h1_full_pipeline",
    difficulty="hard",
    description=(
        "Clean this transaction dataset end-to-end:\n"
        "1. Remove duplicate rows (by transaction_id, keep first occurrence)\n"
        "2. Fix 'date': standardize to YYYY-MM-DD format\n"
        "3. Fix 'amount': remove '$' and commas, convert to float.\n"
        "   Rows where amount is 'N/A', 'null', or 'None' should be DROPPED\n"
        "4. Fix 'category': standardize to Title Case\n"
        "5. Fix 'status': standardize to Title Case\n"
        "6. Reset the index"
    ),
    hint=None,
    max_steps=12,
    setup_dirty=_dirty_h1,
    setup_clean=_clean_h1,
))


# ── H2: Cross-table reconciliation ──────────────────────────────────────────

def _dirty_h2():
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "name": ["Alice Johnson", "bob smith", "Charlie Brown", "DIANA LEE", "Eve Davis"],
        "email": ["alice@mail.com", "bob@mail.com", "charlie@mail.com",
                 "diana@mail.com", "eve@mail.com"],
    })
    orders = pd.DataFrame({
        "order_id": [101, 102, 103, 104, 105, 106, 107],
        "customer_id": [1, 2, 1, 3, 5, 2, 4],
        "product": ["Laptop", "Phone", "Mouse", "Tablet", "Keyboard",
                   "Monitor", "Webcam"],
        "price": [999.99, 699.00, 29.99, 449.00, 79.99, 329.50, 89.50],
        "quantity": [1, 2, 3, 1, 1, 1, 2],
        "total": [999.99, 1498.00, 89.97, 449.00, 79.99, 329.50, 179.00],
    })
    # Introduce errors: wrong totals, wrong total for order 102 and 106
    orders.loc[orders.order_id == 102, "total"] = 1398.00  # should be 1398.00
    orders.loc[orders.order_id == 106, "total"] = 330.00   # should be 329.50
    return pd.concat([
        customers.assign(_table="customers"),
        orders.assign(_table="orders"),
    ], ignore_index=True)

def _clean_h2():
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "name": ["Alice Johnson", "Bob Smith", "Charlie Brown", "Diana Lee", "Eve Davis"],
        "email": ["alice@mail.com", "bob@mail.com", "charlie@mail.com",
                 "diana@mail.com", "eve@mail.com"],
    })
    orders = pd.DataFrame({
        "order_id": [101, 102, 103, 104, 105, 106, 107],
        "customer_id": [1, 2, 1, 3, 5, 2, 4],
        "product": ["Laptop", "Phone", "Mouse", "Tablet", "Keyboard",
                   "Monitor", "Webcam"],
        "price": [999.99, 699.00, 29.99, 449.00, 79.99, 329.50, 89.50],
        "quantity": [1, 2, 3, 1, 1, 1, 2],
        "total": [999.99, 1398.00, 89.97, 449.00, 79.99, 329.50, 179.00],
    })
    merged = orders.merge(customers[["customer_id", "name"]], on="customer_id", how="left")
    return merged[["order_id", "customer_id", "name", "product", "price",
                    "quantity", "total"]].reset_index(drop=True)

register_task(Task(
    task_id="h2_cross_table",
    difficulty="hard",
    description=(
        "This dataset has two embedded tables (marked by '_table' column):\n"
        "  - 'customers': customer_id, name, email\n"
        "  - 'orders': order_id, customer_id, product, price, quantity, total\n\n"
        "Tasks:\n"
        "1. Separate the two tables by '_table' value\n"
        "2. Fix customer names: standardize to Title Case\n"
        "3. Recalculate 'total' as price * quantity (some totals are wrong)\n"
        "4. Join orders with customer names\n"
        "5. Output a single DataFrame with columns: order_id, customer_id, name, "
        "product, price, quantity, total\n"
        "6. Reset the index"
    ),
    hint=None,
    max_steps=12,
    setup_dirty=_dirty_h2,
    setup_clean=_clean_h2,
))


# ── H3: Semi-structured JSON extraction ─────────────────────────────────────

def _dirty_h3():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "raw_data": [
            '{"name": "Alice", "age": 28, "skills": ["python", "sql"]}',
            '{"name": "Bob", "age": 35, "skills": ["java", "go", "rust"]}',
            '{"name": "Charlie", "age": null, "skills": ["python"]}',
            '{"name": "Diana", "age": 42, "skills": []}',
            '{"name": "Eve", "age": 30, "skills": ["javascript", "react", "node"]}',
            '{"name": "Frank", "age": 25, "skills": ["python", "django"]}',
        ],
    })

def _clean_h3():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
        "age": [28.0, 35.0, np.nan, 42.0, 30.0, 25.0],
        "num_skills": [2, 3, 1, 0, 3, 2],
        "primary_skill": ["python", "java", "python", None, "javascript", "python"],
    }).reset_index(drop=True)

register_task(Task(
    task_id="h3_json_extract",
    difficulty="hard",
    description=(
        "Parse the 'raw_data' column (JSON strings) into structured columns.\n"
        "Each JSON object has keys: name, age, skills (array)\n\n"
        "Output columns:\n"
        "  - 'id': keep as-is\n"
        "  - 'name': extracted from JSON\n"
        "  - 'age': extracted from JSON (keep as float, null stays NaN)\n"
        "  - 'num_skills': count of skills in the array\n"
        "  - 'primary_skill': first skill in the array (None if empty)\n"
        "Remove the 'raw_data' column. Reset the index."
    ),
    hint=None,
    max_steps=12,
    setup_dirty=_dirty_h3,
    setup_clean=_clean_h3,
))


# ── H4: Time series cleaning — gaps, duplicates, alignment ──────────────────

def _dirty_h4():
    return pd.DataFrame({
        "timestamp": [
            "2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00",
            "2024-01-01 02:00", "2024-01-01 05:00",
            "2024-01-01 06:00", "2024-01-01 06:00",
            "2024-01-01 09:00", "2024-01-01 10:00", "2024-01-01 11:00",
        ],
        "temperature": [
            20.5, 19.8, 19.2, 19.5, 22.3,
            23.1, 23.0,
            26.0, 26.8, 27.2,
        ],
        "humidity": [
            65, 67, 70, 69, 58,
            55, 56,
            45, 42, 40,
        ],
    })

def _clean_h4():
    df = _dirty_h4().drop_duplicates(subset="timestamp", keep="first")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    full_range = pd.date_range("2024-01-01 00:00", periods=12, freq="h")
    df = df.reindex(full_range).interpolate(method="linear").reset_index()
    df.columns = ["timestamp", "temperature", "humidity"]
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    return df.reset_index(drop=True)

register_task(Task(
    task_id="h4_timeseries_clean",
    difficulty="hard",
    description=(
        "Clean this hourly weather time series:\n"
        "1. Remove duplicate timestamps (keep the first occurrence)\n"
        "2. There are gaps (missing hours). Fill them using "
        "linear interpolation between adjacent values\n"
        "3. Ensure timestamps are hourly from 00:00 to 11:00 (12 rows)\n"
        "4. All numeric values should be float\n"
        "5. Reset the index"
    ),
    hint=None,
    max_steps=12,
    setup_dirty=_dirty_h4,
    setup_clean=_clean_h4,
))


# ── H5: Multi-issue real-world dataset ───────────────────────────────────────

def _dirty_h5():
    return pd.DataFrame({
        "emp_id": ["E001", "E002", "E003", "E004", "E005",
                  "E002", "E006", "E007", "E008", "E009",
                  "E010", "E003"],
        "name": ["  alice johnson ", "BOB SMITH", "charlie Brown",
                "Diana LEE", "eve DAVIS", "Bob Smith",
                " frank MILLER", "GRACE wilson", "heidi clark ",
                "Ivan JONES", "  judy CHEN", "Charlie Brown"],
        "department": ["engineering", "MARKETING", "Engineering",
                      "sales", "ENGINEERING", "marketing",
                      "Sales", "MARKETING", "engineering",
                      "SALES", "Engineering", "engineering"],
        "hire_date": ["01/15/2020", "2020-03-22", "15-06-2020",
                     "2020.09.01", "11/30/2020", "2020-03-22",
                     "2021-01-15", "02/28/2021", "2021.05.10",
                     "07-20-2021", "2021-09-01", "15-06-2020"],
        "salary": ["$85,000", "92500", "$78,000.00", "N/A", "$110,000",
                  "92500", "null", "$95,750", "88000", "$102,500",
                  "None", "$78,000.00"],
        "performance_rating": [4.5, 3.8, 4.2, 3.5, 4.8,
                              3.8, np.nan, 4.0, 3.9, "N/A",
                              4.1, 4.2],
    })

def _clean_h5():
    rows = [
        {"emp_id": "E001", "name": "Alice Johnson", "department": "Engineering",
         "hire_date": "2020-01-15", "salary": 85000.0, "performance_rating": 4.5},
        {"emp_id": "E002", "name": "Bob Smith", "department": "Marketing",
         "hire_date": "2020-03-22", "salary": 92500.0, "performance_rating": 3.8},
        {"emp_id": "E003", "name": "Charlie Brown", "department": "Engineering",
         "hire_date": "2020-06-15", "salary": 78000.0, "performance_rating": 4.2},
        {"emp_id": "E004", "name": "Diana Lee", "department": "Sales",
         "hire_date": "2020-09-01", "salary": np.nan, "performance_rating": 3.5},
        {"emp_id": "E005", "name": "Eve Davis", "department": "Engineering",
         "hire_date": "2020-11-30", "salary": 110000.0, "performance_rating": 4.8},
        {"emp_id": "E006", "name": "Frank Miller", "department": "Sales",
         "hire_date": "2021-01-15", "salary": np.nan, "performance_rating": np.nan},
        {"emp_id": "E007", "name": "Grace Wilson", "department": "Marketing",
         "hire_date": "2021-02-28", "salary": 95750.0, "performance_rating": 4.0},
        {"emp_id": "E008", "name": "Heidi Clark", "department": "Engineering",
         "hire_date": "2021-05-10", "salary": 88000.0, "performance_rating": 3.9},
        {"emp_id": "E009", "name": "Ivan Jones", "department": "Sales",
         "hire_date": "2021-07-20", "salary": 102500.0, "performance_rating": np.nan},
        {"emp_id": "E010", "name": "Judy Chen", "department": "Engineering",
         "hire_date": "2021-09-01", "salary": np.nan, "performance_rating": 4.1},
    ]
    return pd.DataFrame(rows).reset_index(drop=True)

register_task(Task(
    task_id="h5_real_world",
    difficulty="hard",
    description=(
        "Clean this messy employee HR dataset (multiple issues):\n"
        "1. Remove duplicate employees by emp_id (keep first occurrence)\n"
        "2. Fix 'name': strip whitespace, standardize to Title Case\n"
        "3. Fix 'department': standardize to Title Case\n"
        "4. Fix 'hire_date': standardize to YYYY-MM-DD format\n"
        "5. Fix 'salary': remove '$' and commas, convert to float.\n"
        "   'N/A', 'null', 'None' should become NaN (do NOT drop these rows)\n"
        "6. Fix 'performance_rating': convert 'N/A' to NaN, ensure float type\n"
        "7. Reset the index"
    ),
    hint=None,
    max_steps=12,
    setup_dirty=_dirty_h5,
    setup_clean=_clean_h5,
))
