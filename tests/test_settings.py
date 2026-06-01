from agentic_researcher.utils.settings import Settings
import pytest
import os


class TestSettings:
    @pytest.fixture
    def setup_env(self):
        """
        Setup the environment variables for testing.
        """
        os.environ["GEMINI_API_KEY"] = "test_key"
        os.environ["RESEARCH_MODEL"] = "test_model"
        yield
        os.environ.pop("GEMINI_API_KEY")
        os.environ.pop("RESEARCH_MODEL")

    def test_load_from_env(self, setup_env):
        settings = Settings()
        assert settings.gemini_api_key == "test_key"
        assert settings.research_model == "test_model"
        assert settings.output_filepath == "research_report.md"
