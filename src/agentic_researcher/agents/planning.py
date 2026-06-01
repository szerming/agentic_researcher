from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from agentic_researcher.state import ResearchPlan

def get_planning_agent(model: GoogleModel) -> Agent[None, ResearchPlan]:
    system_prompt = (
        "You are an expert technical research planner. Your task is to design a comprehensive research plan "
        "consisting of main topics and detailed subtopics based on the user's survey requirements.\n\n"
        "Ensure that the breadth of the research is fully covered, capturing all requirements, context, and target use cases. "
        "Each topic should have a name and description, and a list of subtopics. Each subtopic should have a name and description.\n"
        "If the user provides feedback or requests modifications on a previously generated plan, adapt the plan accordingly."
    )
    return Agent(
        model,
        result_type=ResearchPlan,
        system_prompt=system_prompt,
    )
