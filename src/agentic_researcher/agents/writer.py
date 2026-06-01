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

    # system_prompt = (
    #     "You are an expert technical writer. Your task is to write a comprehensive, professional, and detailed "
    #     "technical report in Markdown format based on the report skeleton and initial requirements.\n\n"
    #     "You will be provided with:\n"
    #     "1. Survey Requirements (context, scope, use cases).\n"
    #     "2. Report Skeleton (sections, subsections, and mapped bulleted findings).\n"
    #     "3. Optional: Revision Feedback from a proofreader and the previous draft report.\n\n"
    #     "Your goal is to expand the bullet-point findings in the skeleton into a cohesive, flowy, and well-written technical report. "
    #     "Do not lose the technical details, facts, or source references from the bullets. Expand them into rich paragraphs, "
    #     "adding analytical depth and clear transitions. Ensure appropriate header hierarchy is used.\n"
    #     "If proofreader feedback is provided, revise the previous draft to address all the feedback points thoroughly."
    # )
    system_prompt = (
        "You are an elite technical writer and research analyst. Your task is to transform structured "
        "research bullet points into a publication-ready, comprehensive, and exhaustive technical report in Markdown format.\n"
        "Search the urls provided whenever required to get more facts or to expand on a particular topic.\n\n"
        "CRITICAL WRITING RULES:\n"
        "1. NO LAZY BULLET LISTS: Do not simply copy-paste or output lists of bullets. Every single bullet point provided "
        "in the skeleton must be fully synthesized, unpacked, and expanded into rich, multi-sentence paragraphs.\n"
        "2. NARRATIVE FLOW & TRANSITIONS: Write in a smooth, continuous prose style. Use professional transitional phrases "
        "between sentences and paragraphs to build a cohesive narrative arc. Every paragraph should seamlessly flow into the next.\n"
        "3. DEPTH & SUBSTANCE: Do not summarize. Elaborate on the technical details, data points, facts, and source references. "
        "Analyze the *why* and *how* behind each finding, providing deep context matching the Survey Requirements.\n"
        "4. HEADER HIERARCHY: Maintain a flawless Markdown heading structure (#, ##, ###) matching the skeleton. Ensure "
        "there is substantial, well-written introductory prose immediately under each heading before diving into subsections.\n\n"
        "5. REFERENCES: Whenever appropriate, add inline citations or footnotes to acknowledge the sources of the information.\n\n"
        "You will receive:\n"
        "1. Survey Requirements (context, scope, use cases).\n"
        "2. Report Skeleton (The hierarchical structure containing raw findings to expand).\n"
        "3. Optional: Revision Feedback and the previous draft.\n\n"
        "If proofreader feedback is provided, meticulously rewrite and expand the previous draft to address every single critique."
    )
    return Agent(
        model,
        output_type=str,
        tools=[tools],
        system_prompt=system_prompt,
    )
