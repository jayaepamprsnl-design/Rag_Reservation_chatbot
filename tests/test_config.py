from app.core.config import Settings, get_settings


def test_default_model_name():
    s = Settings()
    assert s.openai_model == "gpt-4o-mini"


def test_default_top_k():
    s = Settings()
    assert s.retrieval_top_k == 5


def test_default_db_url():
    s = Settings()
    assert "sqlite" in s.database_url


def test_settings_is_singleton():
    assert get_settings() is get_settings()


def test_log_level_default():
    s = Settings()
    assert s.log_level.upper() == "INFO"
