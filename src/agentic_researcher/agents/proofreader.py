from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel

class ProofreadResult(BaseModel):
    satisfied: bool = Field(description="True if the report accurately and comprehensively answers the survey requirements.")
    feedback: list[str] = Field(default_factory=list, description="Detailed improvement feedback points if not satisfied. Empty if satisfied.")

def get_proofreader_agent(model: GoogleModel) -> Agent[None, ProofreadResult]:
    system_prompt = (
        "You are an expert technical editor and proofreader. Your task is to evaluate the draft research report "
        "against the initial survey requirements.\n\n"
        "You will be provided with:\n"
        "1. Survey Requirements (user's topic, scope, use cases, context).\n"
        "2. Draft Report.\n"
        "Analyze whether the report is comprehensive, technically accurate, covers all requested areas of interest, "
        "and is appropriate for the target context/use cases.\n\n"
        "If it is completely satisfactory, set `satisfied` to True and leave `feedback` empty. "
        "If there are gaps, inaccuracies, formatting issues, or areas requiring improvement, set `satisfied` to False "
        "and provide a list of concrete, actionable feedback points in the `feedback` list."
    )
    return Agent(
        model,
        result_type=ProofreadResult,
        system_prompt=system_prompt,
    )
