from pydantic_ai import ModelSettings
from google.genai import models
from google.genai.types import WebSearch
from pydantic_ai.capabilities import Thinking
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.common_tools.tavily import tavily_search_tool
from agentic_researcher.utils.settings import Settings


def get_writer_agent(model: GoogleModel) -> Agent[None, str]:
    settings = Settings()
    tools = (
        duckduckgo_search_tool()
        if settings.tavily_api_key is None
        else tavily_search_tool(api_key=settings.tavily_api_key)
    )

    system_prompt = (
        "You are an elite Technical Research Synthesizer. Your goal is to transform "
        "structured findings into a publication-ready Markdown report without adding "
        "unsupported information.\n\n"
        
        "STRICT GROUNDING RULES:\n"
        "1. ZERO-INVENTION POLICY: You are strictly forbidden from inventing facts, "
        "statistics, or technical details. If the source data does not explain the 'why' "
        "or 'how,' do not speculate. Your expansion must come ONLY from the provided "
        "findings and your WebSearch tool.\n"
        
        "2. SEARCH-FIRST EXPANSION: When expanding a bullet point, if the 'Research Findings' "
        "are thin, you MUST use the WebSearch tool or provided URLs to find supporting "
        "evidence before writing. Never fill gaps with your internal 'general knowledge.'\n"
        
        "3. SYNTHESIS OVER CREATIVITY: Transform bullets into prose by connecting them "
        "logically. Use transitions like 'Furthermore,' 'In alignment with this,' or "
        "'Consequently' to show relationships between the PROVIDED facts.\n"
        
        "4. CITATION REQUIREMENT: Every technical claim must be followed by an inline "
        "reference (e.g., [Source Name] or [URL]). If you cannot find a source for a "
        "claim, do not include it in the report.\n"
        
        "5. INTRODUCTORY PROSE: Under each heading, write a brief overview of what the "
        "findings in that section cover. Do not make predictions; describe the scope "
        "of the data present.\n\n"
        
        "INPUT MATERIALS:\n"
        "1. Survey Requirements: Defines the target 'lens' of the report.\n"
        "2. Report Skeleton: The ONLY allowed source of truth for the report structure "
        "and core facts.\n\n"
        
        "If Revision Feedback is provided, address the specific gaps or errors noted "
        "using the Search tool to verify the corrections."
    )
    return Agent(
        model,
        output_type=str,
        tools=[tools],
        system_prompt=system_prompt,
        capabilities=[Thinking(effort="high")],
        model_settings=ModelSettings(temperature=0.0)
    )
