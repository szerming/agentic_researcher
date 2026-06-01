from agentic_researcher.utils.settings import Settings
import argparse
import asyncio
import os
import sys
import typer
from loguru import logger

from agentic_researcher.graph import research_graph, SurveyNode
from agentic_researcher.state import ResearchState
from agentic_researcher.deps import ResearchDeps
from pathlib import Path

app = typer.Typer()


@app.command()
def run_research(
    model: str | None = typer.Option(default=None, help="Model to use"),
    output_filename: str | None = typer.Option(default=None, help="Output file path")
):
    """
    Run the technical researcher agentic workflow.
    """

    settings = Settings()
    if model is None:
        model = settings.research_model
    if output_filename is None:
        output_filename = settings.output_filepath

    logger.info(f"Running research with model: {model} and output file: {output_filename}")

    output_dir = Path(__file__).parent.parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename

    print("==========================================")
    print("🚀 Welcome to the Technical Researcher Agentic Workflow!")
    print(f"Using Model: {model}")
    print(f"Output File: {output_path}")
    print("==========================================")

    try:
        asyncio.run(async_main(model=model, output_path=output_path, gemini_api_key=settings.gemini_api_key))

        print("\n==========================================")
        print(f"Success! Technical research report released to: {output_path}")
        print("==========================================")

    except KeyboardInterrupt:
        print("\n⛔ Exited due to KeyboardInterrupt.")
        sys.exit(0)



async def async_main(model: str, output_path: Path, gemini_api_key: str):
    state = ResearchState()
    deps = ResearchDeps(
        model_name=model,
        api_key=gemini_api_key,
        output_filepath=str(output_path)
    )

    try:
        result_path = await research_graph.run(
            inputs=SurveyNode(),
            state=state,
            deps=deps
        )
    except Exception as e:
        logger.error(f"\nWorkflow failed with error: {e}")
        raise e

if __name__ == "__main__":
    app()
