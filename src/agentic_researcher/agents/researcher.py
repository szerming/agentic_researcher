from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from agentic_researcher.state import SubtopicFindings
from agentic_researcher.utils.search import search_duckduckgo
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

def get_researcher_agent(model: GoogleModel) -> Agent[None, SubtopicFindings]:
    agent = Agent(
        model,
        output_type=SubtopicFindings,
        tools=[duckduckgo_search_tool()],
        system_prompt=(
            "You are an expert technical researcher. Your task is to investigate a given subtopic using the web search tool. "
            "You should formulate search queries, execute them, analyze the results, and extract concrete, detailed, and accurate findings.\n\n"
            "Format the output as bulleted findings (each finding should be a complete, informative technical statement or fact) "
            "and list the source URLs or reference names used. Try to be detailed and cover the depth of the subtopic. "
            "If the search returns no results or fails, use your internal parametric knowledge to answer the subtopic as best as you can."
        )
    )

    # @agent.tool
    # async def search_web(ctx: RunContext[None], query: str) -> str:
    #     """
    #     Search the web for information related to a query.
    #     Returns a list of search result titles, URLs, and snippets.
    #     """
    #     results = await search_duckduckgo(query, max_results=5)
    #     if not results:
    #         return "No search results found."
    #     formatted = []
    #     for i, r in enumerate(results):
    #         formatted.append(f"Result {i+1}:\nTitle: {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}\n")
    #     return "\n".join(formatted)

    return agent
