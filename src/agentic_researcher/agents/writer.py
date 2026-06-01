from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel

def get_writer_agent(model: GoogleModel) -> Agent[None, str]:
    system_prompt = (
        "You are an expert technical writer. Your task is to write a comprehensive, professional, and detailed "
        "technical report in Markdown format based on the report skeleton and initial requirements.\n\n"
        "You will be provided with:\n"
        "1. Survey Requirements (context, scope, use cases).\n"
        "2. Report Skeleton (sections, subsections, and mapped bulleted findings).\n"
        "3. Optional: Revision Feedback from a proofreader and the previous draft report.\n\n"
        "Your goal is to expand the bullet-point findings in the skeleton into a cohesive, flowy, and well-written technical report. "
        "Do not lose the technical details, facts, or source references from the bullets. Expand them into rich paragraphs, "
        "adding analytical depth and clear transitions. Ensure appropriate header hierarchy is used.\n"
        "If proofreader feedback is provided, revise the previous draft to address all the feedback points thoroughly."
    )
    return Agent(
        model,
        result_type=str,
        system_prompt=system_prompt,
    )
