from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel

from agentic_researcher.state import ConversationTurn


def get_survey_agent(model: GoogleModel) -> Agent[None, ConversationTurn]:
    # Default result_type is str for normal conversational flows
    fireside_chat_agent = Agent[None, ConversationTurn](
        model,
        output_type=ConversationTurn,
        system_prompt=(
            "You are a friendly and engaging host for a fireside chat. "
            "Your objective is to gather information from the user to populate a structured summary model. "
            "Do not interrogate the user like a form. Instead, engage in a natural, flowing conversation. "
            "Gently steer the dialogue towards uncovering the details needed for the summary (Topic, Context, Purpose, Audience, Keywords). "
            "Update the `current_data` field in your response incrementally as you learn new information. "
            "If a specific piece of information is still unknown, keep its field as None or empty."
        ),
    )

    return fireside_chat_agent
