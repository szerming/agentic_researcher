from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    gemini_api_key: str = Field(..., description="The google API key")
    research_model: str = Field(default="gemini-2.5-flash-lite", description="The google LLM model")
    output_filepath: str = Field(default="research_report.md", description="The output markdown file path")