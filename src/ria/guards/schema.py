"""What the assistant knows about the dataset: table schemas, which columns are personal data, and the prompt-ready digest.

`STATIC_SCHEMA` is the built-in copy of the four thelook_ecommerce tables the app may use; `ria init` can replace it
with a live copy cached in schema.json. The SQL guard uses the schema to resolve columns and the PII list to block
personal fields; `graph.deps` uses the digest as the schema description in prompts. No dependencies on the rest of the package.
"""
from __future__ import annotations

import json
from pathlib import Path

STATIC_SCHEMA: dict[str, dict[str, str]] = {
    "users": {
        "id": "INT64", "first_name": "STRING", "last_name": "STRING", "email": "STRING", "age": "INT64",
        "gender": "STRING", "state": "STRING", "street_address": "STRING", "postal_code": "STRING",
        "city": "STRING", "country": "STRING", "latitude": "FLOAT64", "longitude": "FLOAT64",
        "traffic_source": "STRING", "created_at": "TIMESTAMP",
    },
    "orders": {
        "order_id": "INT64", "user_id": "INT64", "status": "STRING", "gender": "STRING",
        "created_at": "TIMESTAMP", "returned_at": "TIMESTAMP", "shipped_at": "TIMESTAMP",
        "delivered_at": "TIMESTAMP", "num_of_item": "INT64",
    },
    "order_items": {
        "id": "INT64", "order_id": "INT64", "user_id": "INT64", "product_id": "INT64",
        "inventory_item_id": "INT64", "status": "STRING", "created_at": "TIMESTAMP",
        "shipped_at": "TIMESTAMP", "delivered_at": "TIMESTAMP", "returned_at": "TIMESTAMP",
        "sale_price": "FLOAT64",
    },
    "products": {
        "id": "INT64", "cost": "FLOAT64", "category": "STRING", "name": "STRING", "brand": "STRING",
        "retail_price": "FLOAT64", "department": "STRING", "sku": "STRING",
        "distribution_center_id": "INT64",
    },
}

PII_COLUMNS: frozenset[str] = frozenset(
    {"email", "first_name", "last_name", "street_address", "postal_code", "latitude", "longitude"}
)

DEMOGRAPHIC_COLUMNS: frozenset[str] = frozenset({"age", "gender", "state", "city", "country", "traffic_source"})
# (table, column) pairs treated as demographics: the users columns plus orders.gender, which is copied from the user.
DEMOGRAPHIC_FIELDS: frozenset[tuple[str, str]] = frozenset(
    {("users", c) for c in DEMOGRAPHIC_COLUMNS} | {("orders", "gender")}
)

# Customer keys in order of preference for the minimum-group count (users.id, else order_items.user_id, else orders.user_id).
CUSTOMER_KEYS: tuple[tuple[str, str], ...] = (("users", "id"), ("order_items", "user_id"), ("orders", "user_id"))
# Everything that identifies one customer, one order or one item; none may sit next to a demographic outside an aggregate.
CUSTOMER_IDENTIFIERS: frozenset[tuple[str, str]] = frozenset(CUSTOMER_KEYS) | {
    ("orders", "order_id"), ("order_items", "order_id"), ("order_items", "id"), ("order_items", "inventory_item_id"),
}

TABLE_NOTES = {
    "users": f"one row per customer; demographics only may be used ({', '.join(sorted(DEMOGRAPHIC_COLUMNS))}), always grouped",
    "orders": "one row per order; status in Complete, Shipped, Processing, Cancelled, Returned",
    "order_items": "one row per item sold; sale_price is revenue per item; join products on product_id, users on user_id",
    "products": "catalog; {scope_column} drives scoping; cost vs retail_price gives margin",
}


def qualify_schema(dataset: str, schema: dict | None = None) -> dict:
    """Nest `schema` as `{project: {dataset: {table: columns}}}`, the shape sqlglot's `qualify` needs for fully qualified table names."""
    catalog, db = dataset.split(".")
    return {catalog: {db: dict(schema or STATIC_SCHEMA)}}


def schema_digest(
    schema: dict | None = None, dataset: str = "bigquery-public-data.thelook_ecommerce", scope_column: str = "brand"
) -> str:
    """Describe the tables, columns, join keys and table notes as text for prompts. Personal columns on `users`
    are left out, so the model is never shown them (the SQL guard blocks them as well, in case it guesses)."""
    schema = schema or STATIC_SCHEMA
    lines = [f"Dataset: `{dataset}` (always use fully qualified table names)."]
    for table, cols in schema.items():
        visible = [f"{c} {t}" for c, t in cols.items() if not (table == "users" and c in PII_COLUMNS)]
        lines.append(f"- `{dataset}.{table}`: {', '.join(visible)}. {TABLE_NOTES.get(table, '').format(scope_column=scope_column)}")
    lines.append("Joins: order_items.user_id = users.id; order_items.product_id = products.id; order_items.order_id = orders.order_id.")
    lines.append("Personal columns on users (email, names, address, coordinates) do not exist for you.")
    return "\n".join(lines)


def load_schema(path: Path) -> dict:
    """Load the cached live schema from `path`, or return `STATIC_SCHEMA` if the file is missing, not valid JSON,
    or lacks any of the required tables. A bad cache falls back silently rather than blocking startup."""
    p = Path(path)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            if isinstance(data, dict) and set(data) >= set(STATIC_SCHEMA):
                return data
        except json.JSONDecodeError:
            pass
    return STATIC_SCHEMA
