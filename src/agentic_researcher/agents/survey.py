from typing import Union
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from agentic_researcher.state import SurveyData

class AskQuestion(BaseModel):
    question: str = Field(description="The next question to ask the user to gather the topic details.")

class SurveyComplete(BaseModel):
    survey_data: SurveyData = Field(description="The finalized survey data once all context, scope, use cases, and areas of interest are understood.")

SurveyResult = Union[AskQuestion, SurveyComplete]

def get_survey_agent(model: GoogleModel) -> Agent[None, SurveyResult]:
    system_prompt = (
        "You are an expert research requirement gathering agent. Your goal is to gather detailed information about a research topic from the user. "
        "You need to understand:\n"
        "1. The core topic of interest.\n"
        "2. The scope of the research (what is in-scope vs out-of-scope).\n"
        "3. The background context or target audience.\n"
        "4. The primary use cases or applications for the report.\n"
        "5. Specific areas of interest or questions they want answered.\n\n"
        "Ask ONE question at a time to gather this information. Keep it brief and focused. "
        "Once you have gathered enough information to fully populate the SurveyData model (all fields topic, scope, context, use_case, areas_of_interest), "
        "return the `SurveyComplete` model. Do not continue asking questions once you have all the fields."
    )
    return Agent(
        model,
        output_type=SurveyResult,
        system_prompt=system_prompt,
    )
