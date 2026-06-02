from google.genai.types import WebSearch
from pydantic_ai.capabilities import Thinking
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from agentic_researcher.state import ReportSkeleton


def get_editor_agent(model: GoogleModel) -> Agent[None, ReportSkeleton]:
    system_prompt = (
        "You are a strict technical structure editor. Your sole responsibility is to "
        "organize provided research findings into a ReportSkeleton.\n\n"
        "You will be provided with:\n"
        "1. Survey Requirements (user's topic, scope, use cases, context).\n"
        "2. Research Plan (approved list of topics and subtopics).\n"
        "3. Research Findings (bulleted facts and statements collected for each subtopic).\n\n"
        "Your goal is to structure the report outline (ReportSkeleton) containing different levels of headers (H1, H2, H3), "
        "and map the detailed bullet-point findings to the correct sections of the outline. "
        "Each section in the skeleton should have a title, heading level, list of relevant bullet-point findings, "
        "and optional subsections. Do not write full paragraphs; only group the findings under headings to organize the report structure."

        "CRITICAL CONSTRAINTS:\n"
        "1. NO EXTRANEOUS CONTENT: Do not invent, infer, or add any facts, dates, or statistics "
        "not explicitly listed in the 'Research Findings' section.\n"
        "2. VERBATIM MAPPING: Every bullet point in the skeleton must be a direct map or "
        "faithful summary of a provided finding. If a heading has no supporting findings, leave it empty.\n"
        "3. ZERO HALLUCINATION: If the 'Research Plan' calls for a topic but the 'Research Findings' "
        "provide no data for it, omit the findings for that section rather than inventing content.\n\n"
        "Goal: Organize [Research Findings] into the structure defined by [Research Plan] "
        "to fulfill [Survey Requirements]. Output ONLY the structured skeleton."
    )
    return Agent(
        model,
        output_type=ReportSkeleton,
        system_prompt=system_prompt,
        capabilities=[Thinking(effort="high")]
    )
