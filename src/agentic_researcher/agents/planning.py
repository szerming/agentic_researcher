from pydantic_ai import RunContext
from agentic_researcher.state import ResearchPlanningDependencies
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from agentic_researcher.state import ResearchPlan
from pydantic_ai.capabilities import Thinking


def get_planning_agent(model: GoogleModel) -> Agent[None, ResearchPlan]:
    
    system_prompt = (
        "You are an expert technical research planner. Your task is to design a comprehensive research plan "
        "consisting of main topics and detailed subtopics based on the user's survey requirements.\n\n"
        "Ensure that the breadth of the research is fully covered, capturing all requirements, context, and target use cases. "
        "Each topic should have a name and description, and a list of subtopics. Each subtopic should have a name and description.\n"
        "If the user provides feedback or requests modifications on a previously generated plan, adapt the plan accordingly\n\n."
    )
    agent = Agent(
        model,
        output_type=ResearchPlan,
        system_prompt=system_prompt,
        deps_type=ResearchPlanningDependencies,
        capabilities=[Thinking()],
    )

    @agent.system_prompt
    def inject_research_data(ctx: RunContext[ResearchPlanningDependencies]) -> str:
        # Start with the foundational survey data
        prompt = f"### FOUNDATIONAL REQUIREMENTS (SurveyData):\n{ctx.deps.survey_data}\n\n"
        
        # Conditionally add revision context if it exists
        if ctx.deps.previous_plan:
            prompt += f"### PREVIOUS PLAN TO REVISE:\n{ctx.deps.previous_plan}\n\n"
        
        if ctx.deps.user_feedback:
            prompt += f"### USER FEEDBACK FOR REVISION:\n{ctx.deps.user_feedback}\n\n"
            
        return prompt

    return agent
