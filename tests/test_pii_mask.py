from ria.guards.pii_mask import mask_pii


def test_masks_email():
    text, n = mask_pii("Contact jane.doe+x@example.co.uk for details")
    assert "[email]" in text and "jane" not in text and n == 1


def test_masks_phone_formats():
    text, n = mask_pii("Call (415) 555-2671 or +1 415.555.2671 or 415-555-2671")
    assert text.count("[phone]") == 3 and n == 3


def test_masks_street_address():
    text, n = mask_pii("Ships to 1234 Elm Street Apt 5 and 77 Ocean View Blvd.")
    assert text.count("[address]") == 2 and n == 2


def test_leaves_numbers_and_prose_alone():
    text, n = mask_pii("Revenue was $1,234,567 across 2024 and 2025; AOV 84.3")
    assert n == 0 and text.startswith("Revenue was $1,234,567")


def test_leaves_ids_and_counts_alone():
    for text in [
        "Orders: 1234567890 across 2025-10-01 to 2026-03-31",
        "| 12 | 2026-01-05 | 4155552671 |",
        "SKU 20260105123",
    ]:
        assert mask_pii(text) == (text, 0)


def test_leaves_titled_phrases_alone():
    for text in ["Sold 500 Best Way units", "Top 40 Home Place sets this month"]:
        assert mask_pii(text) == (text, 0)


def test_leaves_markdown_table_alone():
    table = "| state | orders |\n|---|---|\n| CA | 12034 |\n| TX | 9876 |"
    text, n = mask_pii(table)
    assert n == 0 and text == table
