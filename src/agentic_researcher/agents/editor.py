from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from agentic_researcher.state import ReportSkeleton

def get_editor_agent(model: GoogleModel) -> Agent[None, ReportSkeleton]:
    system_prompt = (
        "You are an expert technical report editor. Your task is to design a detailed outline/skeleton of a research report.\n\n"
        "You will be provided with:\n"
        "1. Survey Requirements (user's topic, scope, use cases, context).\n"
        "2. Research Plan (approved list of topics and subtopics).\n"
        "3. Research Findings (bulleted facts and statements collected for each subtopic).\n\n"
        "Your goal is to structure the report outline (ReportSkeleton) containing different levels of headers (H1, H2, H3), "
        "and map the detailed bullet-point findings to the correct sections of the outline. "
        "Each section in the skeleton should have a title, heading level, list of relevant bullet-point findings, "
        "and optional subsections. Do not write full paragraphs; only group the findings under headings to organize the report structure."
    )
    return Agent(
        model,
        result_type=ReportSkeleton,
        system_prompt=system_prompt,
    )
