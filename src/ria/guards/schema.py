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

TABLE_NOTES = {
    "users": "one row per customer; demographics only may be used (state, age, gender, traffic_source, created_at)",
    "orders": "one row per order; status in Complete, Shipped, Processing, Cancelled, Returned",
    "order_items": "one row per item sold; sale_price is revenue per item; join products on product_id, users on user_id",
    "products": "catalog; category and department drive scoping; cost vs retail_price gives margin",
}


def qualify_schema(dataset: str, schema: dict | None = None) -> dict:
    catalog, db = dataset.split(".")
    return {catalog: {db: dict(schema or STATIC_SCHEMA)}}


def schema_digest(schema: dict | None = None, dataset: str = "bigquery-public-data.thelook_ecommerce") -> str:
    schema = schema or STATIC_SCHEMA
    lines = [f"Dataset: `{dataset}` (always use fully qualified table names)."]
    for table, cols in schema.items():
        visible = [f"{c} {t}" for c, t in cols.items() if not (table == "users" and c in PII_COLUMNS)]
        lines.append(f"- `{dataset}.{table}`: {', '.join(visible)}. {TABLE_NOTES.get(table, '')}")
    lines.append("Joins: order_items.user_id = users.id; order_items.product_id = products.id; order_items.order_id = orders.order_id.")
    lines.append("Personal columns on users (email, names, address, coordinates) do not exist for you.")
    return "\n".join(lines)


def load_schema(path: Path) -> dict:
    p = Path(path)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            if isinstance(data, dict) and set(data) >= set(STATIC_SCHEMA):
                return data
        except json.JSONDecodeError:
            pass
    return STATIC_SCHEMA
