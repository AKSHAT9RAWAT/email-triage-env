"""
tasks.py — Task definitions for the SQL Repair Environment.

Each task has:
  - schema_ddl:       CREATE TABLE statements
  - seed_data:        INSERT statements to populate the DB
  - broken_sql:       The broken query the agent must fix  (empty for task 3)
  - task_description: Natural language spec
  - expected_query:   Reference SQL (used to generate expected output)
  - grader:           Function(agent_sql, conn) -> float in [0.0, 1.0]
"""

import sqlite3
import json
from typing import Callable, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rows_to_set(rows):
    """Convert list of tuples to a frozenset of JSON-serialised rows for order-independent comparison."""
    return frozenset(json.dumps(list(r), default=str) for r in rows)


def _run_sql(conn: sqlite3.Connection, sql: str):
    """Run sql, return (rows, error_string). rows is None on error."""
    try:
        cur = conn.execute(sql)
        return cur.fetchall(), None
    except Exception as e:
        return None, str(e)


def _score_result(agent_rows, expected_rows) -> float:
    """
    Compare agent output to expected output.
    Returns 1.0 for exact match, partial credit for partial overlap,
    0.0 if the query errored.
    """
    if agent_rows is None:
        return 0.0
    agent_set = _rows_to_set(agent_rows)
    expected_set = _rows_to_set(expected_rows)
    if agent_set == expected_set:
        return 1.0
    if len(expected_set) == 0:
        return 1.0 if len(agent_set) == 0 else 0.1
    # Partial credit: Jaccard similarity
    intersection = len(agent_set & expected_set)
    union = len(agent_set | expected_set)
    jaccard = intersection / union if union > 0 else 0.0
    # Penalise if totally wrong row count
    count_ratio = min(len(agent_set), len(expected_set)) / max(len(agent_set), len(expected_set)) if max(len(agent_set), len(expected_set)) > 0 else 1.0
    return round((jaccard * 0.7 + count_ratio * 0.3), 4)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — EASY: Syntax Fix
# ─────────────────────────────────────────────────────────────────────────────

TASK1_SCHEMA = """
CREATE TABLE employees (
    id       INTEGER PRIMARY KEY,
    name     TEXT    NOT NULL,
    dept     TEXT    NOT NULL,
    salary   REAL    NOT NULL,
    manager_id INTEGER
);
""".strip()

TASK1_SEED = """
INSERT INTO employees VALUES (1, 'Alice',   'Engineering', 95000, NULL);
INSERT INTO employees VALUES (2, 'Bob',     'Engineering', 82000, 1);
INSERT INTO employees VALUES (3, 'Carol',   'Marketing',   74000, NULL);
INSERT INTO employees VALUES (4, 'Dave',    'Marketing',   68000, 3);
INSERT INTO employees VALUES (5, 'Eve',     'Engineering', 91000, 1);
INSERT INTO employees VALUES (6, 'Frank',   'HR',          61000, NULL);
INSERT INTO employees VALUES (7, 'Grace',   'HR',          59000, 6);
INSERT INTO employees VALUES (8, 'Heidi',   'Marketing',   77000, 3);
""".strip()

# Broken: missing comma between dept and salary in SELECT, wrong keyword FORM
TASK1_BROKEN = "SELEC name dept salary FORM employees WHERE dept = 'Engineering' ORDER BY salary DESC"

TASK1_EXPECTED = "SELECT name, dept, salary FROM employees WHERE dept = 'Engineering' ORDER BY salary DESC"

TASK1_DESCRIPTION = (
    "List the name, department, and salary of all Engineering employees, "
    "ordered by salary descending. "
    "The query has multiple syntax errors — fix them all."
)

TASK1_SAMPLE = json.dumps([
    {"id": 1, "name": "Alice", "dept": "Engineering", "salary": 95000, "manager_id": None},
    {"id": 2, "name": "Bob",   "dept": "Engineering", "salary": 82000, "manager_id": 1},
])


def grade_task1(agent_sql: str, conn: sqlite3.Connection) -> float:
    expected_rows, _ = _run_sql(conn, TASK1_EXPECTED)
    agent_rows, err = _run_sql(conn, agent_sql)
    return _score_result(agent_rows, expected_rows)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — MEDIUM: Logic Fix
# ─────────────────────────────────────────────────────────────────────────────

TASK2_SCHEMA = """
CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    amount      REAL    NOT NULL,
    status      TEXT    NOT NULL,   -- 'completed' | 'pending' | 'cancelled'
    order_date  TEXT    NOT NULL    -- ISO date string
);

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    country     TEXT NOT NULL
);
""".strip()

TASK2_SEED = """
INSERT INTO customers VALUES (1, 'Acme Corp',    'US');
INSERT INTO customers VALUES (2, 'Globex',       'UK');
INSERT INTO customers VALUES (3, 'Initech',      'US');
INSERT INTO customers VALUES (4, 'Umbrella',     'DE');
INSERT INTO customers VALUES (5, 'Hooli',        'US');

INSERT INTO orders VALUES (1,  1, 1500.00, 'completed', '2024-01-15');
INSERT INTO orders VALUES (2,  1,  300.00, 'cancelled', '2024-02-01');
INSERT INTO orders VALUES (3,  2, 2200.00, 'completed', '2024-01-20');
INSERT INTO orders VALUES (4,  3,  800.00, 'pending',   '2024-03-05');
INSERT INTO orders VALUES (5,  3, 1200.00, 'completed', '2024-02-14');
INSERT INTO orders VALUES (6,  4,  950.00, 'completed', '2024-01-30');
INSERT INTO orders VALUES (7,  5, 3100.00, 'completed', '2024-03-10');
INSERT INTO orders VALUES (8,  5,  450.00, 'cancelled', '2024-03-12');
INSERT INTO orders VALUES (9,  2,  670.00, 'pending',   '2024-04-01');
INSERT INTO orders VALUES (10, 1, 2000.00, 'completed', '2024-04-05');
""".strip()

# Bug: uses LEFT JOIN and includes cancelled orders, also aggregates wrong column
TASK2_BROKEN = """
SELECT c.name, SUM(o.order_id) AS total_revenue
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.name
ORDER BY total_revenue DESC
"""

TASK2_EXPECTED = """
SELECT c.name, SUM(o.amount) AS total_revenue
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
GROUP BY c.name
ORDER BY total_revenue DESC
"""

TASK2_DESCRIPTION = (
    "For each customer, calculate their total revenue from COMPLETED orders only. "
    "Show customer name and total_revenue, ordered by total_revenue descending. "
    "The query has logic errors: it uses the wrong JOIN type, sums the wrong column, "
    "and does not filter by status."
)

TASK2_SAMPLE = json.dumps([
    {"order_id": 1, "customer_id": 1, "amount": 1500.0, "status": "completed", "order_date": "2024-01-15"},
    {"customer_id": 1, "name": "Acme Corp", "country": "US"},
])


def grade_task2(agent_sql: str, conn: sqlite3.Connection) -> float:
    expected_rows, _ = _run_sql(conn, TASK2_EXPECTED)
    agent_rows, err = _run_sql(conn, agent_sql)
    return _score_result(agent_rows, expected_rows)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — HARD: Schema Rewrite from Natural Language
# ─────────────────────────────────────────────────────────────────────────────

TASK3_SCHEMA = """
CREATE TABLE products (
    product_id   INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL,
    category     TEXT    NOT NULL,
    price        REAL    NOT NULL,
    supplier_id  INTEGER NOT NULL
);

CREATE TABLE suppliers (
    supplier_id  INTEGER PRIMARY KEY,
    supplier_name TEXT   NOT NULL,
    country      TEXT    NOT NULL
);

CREATE TABLE inventory (
    product_id   INTEGER NOT NULL,
    warehouse    TEXT    NOT NULL,
    quantity     INTEGER NOT NULL,
    last_updated TEXT    NOT NULL,
    PRIMARY KEY (product_id, warehouse)
);

CREATE TABLE sales (
    sale_id      INTEGER PRIMARY KEY,
    product_id   INTEGER NOT NULL,
    quantity_sold INTEGER NOT NULL,
    sale_date    TEXT    NOT NULL,
    unit_price   REAL    NOT NULL
);
""".strip()

TASK3_SEED = """
INSERT INTO suppliers VALUES (1, 'TechSource',   'US');
INSERT INTO suppliers VALUES (2, 'GlobalParts',  'CN');
INSERT INTO suppliers VALUES (3, 'EuroSupply',   'DE');

INSERT INTO products VALUES (1,  'Laptop Pro',    'Electronics', 1200.00, 1);
INSERT INTO products VALUES (2,  'USB Hub',        'Electronics',   35.00, 2);
INSERT INTO products VALUES (3,  'Office Chair',   'Furniture',    350.00, 3);
INSERT INTO products VALUES (4,  'Standing Desk',  'Furniture',    650.00, 3);
INSERT INTO products VALUES (5,  'Webcam HD',      'Electronics',   89.00, 1);
INSERT INTO products VALUES (6,  'Keyboard Mech',  'Electronics',  145.00, 2);
INSERT INTO products VALUES (7,  'Monitor 27"',    'Electronics',  420.00, 1);
INSERT INTO products VALUES (8,  'Desk Lamp',      'Furniture',     55.00, 2);

INSERT INTO inventory VALUES (1, 'WH-East',  15, '2024-04-01');
INSERT INTO inventory VALUES (1, 'WH-West',   8, '2024-04-01');
INSERT INTO inventory VALUES (2, 'WH-East',  120,'2024-04-01');
INSERT INTO inventory VALUES (3, 'WH-East',  20, '2024-04-01');
INSERT INTO inventory VALUES (4, 'WH-West',   5, '2024-04-01');
INSERT INTO inventory VALUES (5, 'WH-East',  45, '2024-04-01');
INSERT INTO inventory VALUES (6, 'WH-West',  30, '2024-04-01');
INSERT INTO inventory VALUES (7, 'WH-East',  12, '2024-04-01');
INSERT INTO inventory VALUES (8, 'WH-East',  60, '2024-04-01');

INSERT INTO sales VALUES (1,  1, 3,  '2024-03-01', 1200.00);
INSERT INTO sales VALUES (2,  1, 2,  '2024-03-15', 1200.00);
INSERT INTO sales VALUES (3,  2, 10, '2024-03-05', 35.00);
INSERT INTO sales VALUES (4,  5, 6,  '2024-03-10', 89.00);
INSERT INTO sales VALUES (5,  7, 4,  '2024-03-20', 420.00);
INSERT INTO sales VALUES (6,  3, 2,  '2024-03-22', 350.00);
INSERT INTO sales VALUES (7,  6, 5,  '2024-03-25', 145.00);
INSERT INTO sales VALUES (8,  1, 1,  '2024-04-01', 1200.00);
INSERT INTO sales VALUES (9,  4, 3,  '2024-04-02', 650.00);
INSERT INTO sales VALUES (10, 2, 8,  '2024-04-03', 35.00);
""".strip()

TASK3_BROKEN = ""  # No broken query — agent must write from scratch

TASK3_EXPECTED = """
SELECT
    p.name                              AS product_name,
    p.category,
    s.supplier_name,
    COALESCE(SUM(inv.quantity), 0)      AS total_stock,
    COALESCE(SUM(sa.quantity_sold), 0)  AS total_sold,
    COALESCE(SUM(sa.quantity_sold * sa.unit_price), 0.0) AS total_revenue
FROM products p
JOIN suppliers s  ON p.supplier_id  = s.supplier_id
LEFT JOIN inventory inv ON p.product_id = inv.product_id
LEFT JOIN sales sa      ON p.product_id = sa.product_id
GROUP BY p.product_id, p.name, p.category, s.supplier_name
ORDER BY total_revenue DESC
"""

TASK3_DESCRIPTION = (
    "Write a SQL query from scratch that produces a product performance report. "
    "For each product show: product_name, category, supplier_name, "
    "total_stock (sum of inventory.quantity across all warehouses), "
    "total_sold (sum of sales.quantity_sold), "
    "and total_revenue (sum of quantity_sold * unit_price from sales). "
    "Use 0 for products with no inventory or sales. "
    "Order by total_revenue descending."
)

TASK3_SAMPLE = json.dumps([
    {"product_id": 1, "name": "Laptop Pro", "category": "Electronics", "price": 1200.0, "supplier_id": 1},
    {"supplier_id": 1, "supplier_name": "TechSource", "country": "US"},
    {"product_id": 1, "warehouse": "WH-East", "quantity": 15},
])


def grade_task3(agent_sql: str, conn: sqlite3.Connection) -> float:
    expected_rows, _ = _run_sql(conn, TASK3_EXPECTED)
    agent_rows, err = _run_sql(conn, agent_sql)
    return _score_result(agent_rows, expected_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Task registry
# ─────────────────────────────────────────────────────────────────────────────

TASKS = {
    "syntax_fix": {
        "schema_ddl":        TASK1_SCHEMA,
        "seed_sql":          TASK1_SEED,
        "broken_sql":        TASK1_BROKEN,
        "expected_sql":      TASK1_EXPECTED,
        "task_description":  TASK1_DESCRIPTION,
        "sample_data":       TASK1_SAMPLE,
        "grader":            grade_task1,
    },
    "logic_fix": {
        "schema_ddl":        TASK2_SCHEMA,
        "seed_sql":          TASK2_SEED,
        "broken_sql":        TASK2_BROKEN,
        "expected_sql":      TASK2_EXPECTED,
        "task_description":  TASK2_DESCRIPTION,
        "sample_data":       TASK2_SAMPLE,
        "grader":            grade_task2,
    },
    "schema_rewrite": {
        "schema_ddl":        TASK3_SCHEMA,
        "seed_sql":          TASK3_SEED,
        "broken_sql":        TASK3_BROKEN,
        "expected_sql":      TASK3_EXPECTED,
        "task_description":  TASK3_DESCRIPTION,
        "sample_data":       TASK3_SAMPLE,
        "grader":            grade_task3,
    },
}
