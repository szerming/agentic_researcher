import os
from pydantic import BaseModel, Field
from pydantic_ai.models.google import GoogleModel
from dotenv import load_dotenv


class ResearchDeps(BaseModel):
    model_name: str = Field(
        ..., description="The name of the Google Gemini model to use."
    )
    api_key: str | None = Field(
        default=None, description="Optional Google Gemini API key."
    )
    output_filepath: str = Field(
        default="research_report.md",
        description="The markdown file path where the report is written.",
    )


def get_model(model_name: str | None = None, api_key: str | None = None) -> GoogleModel:
    load_dotenv()
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not model_name:
        model_name = os.getenv("RESEARCH_MODEL", "gemini-3-flash-preview")

    if key:
        os.environ["GEMINI_API_KEY"] = key

    return GoogleModel(model_name)
