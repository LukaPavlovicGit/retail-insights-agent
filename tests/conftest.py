from __future__ import annotations

import pytest

from ria import db
from ria.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        gemini_api_key="test-key",
        gcp_project="test-project",
        db_path=tmp_path / "ria.db",
        data_dir=tmp_path / "data",
        embedding_dim=32,
    )


@pytest.fixture
def conn(settings):
    c = db.connect(":memory:")
    db.init_schema(c, settings.embedding_dim)
    yield c
    c.close()
