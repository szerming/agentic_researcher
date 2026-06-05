import asyncio
from typing import List

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel


# -----------------------------------------------------------------------------
# 1. Data Models
# -----------------------------------------------------------------------------


class FiresideChatOutput(BaseModel):
    """The structured data we aim to populate through the conversation."""

    topic: str = Field(
        default=None, description="The main theme or topic of the fireside chat."
    )
    context: str = Field(
        default=None,
        description="The background, setting, or landscape surrounding the topic.",
    )
    purpose: str = Field(
        default=None,
        description="The core objective or why this conversation is happening.",
    )
    audience: str = Field(
        default=None, description="The target demographic or audience for this summary."
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="A list of 3-7 core keywords or tags from the chat.",
    )


class ConversationTurn(BaseModel):
    """
    The return type for the agent.
    It allows the agent to output both the natural language response
    and the current state of the structured data simultaneously.
    """

    message: str = Field(
        description="The natural language follow-up message to the user. "
        "Primary intention: steer conversation to populate data. "
        "Secondary intention: keep the conversation flow."
    )
    current_data: FiresideChatOutput = Field(
        description="The updated state of the FiresideChatOutput model based on the conversation so far."
    )


# -----------------------------------------------------------------------------
# 2. Agent Configuration
# -----------------------------------------------------------------------------

# Initialize the model (requires OPENAI_API_KEY in environment variables)
model = "gemini-2.5-flash-lite"

fireside_chat_agent = Agent(
    model=GoogleModel(model),
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


# -----------------------------------------------------------------------------
# 3. Workflow Logic
# -----------------------------------------------------------------------------


def check_completeness(data: FiresideChatOutput) -> List[str]:
    """Checks if the FiresideChatOutput is fully populated."""
    missing_fields = []

    if not data.topic:
        missing_fields.append("topic")
    if not data.context:
        missing_fields.append("context")
    if not data.purpose:
        missing_fields.append("purpose")
    if not data.audience:
        missing_fields.append("audience")
    # Keywords should ideally have at least 3 items based on description,
    # but strictly we check if it's empty for basic completeness.
    if not data.keywords:
        missing_fields.append("keywords")

    return missing_fields


async def main():
    # Track message history for context continuity
    message_history = []

    # Track the accumulated structured data
    final_output_state = FiresideChatOutput()

    print("🔥 Fireside Chat Agent initialized.")
    print("Agent: Hello! Let's have a chat. Tell me, what is on your mind today?")
    print("(Type 'stop' when you are ready to finish.)\n")

    try:
        while True:
            # Get user input
            user_input = input("You: ")

            # Termination condition
            if user_input.lower().strip() == "stop":
                missing_fields = check_completeness(final_output_state)

                if not missing_fields:
                    print("\n--- ✅ Final Output Generated ---")
                    print(final_output_state.model_dump_json(indent=2))
                    print("\nAgent: Thanks for the great chat! Here is your summary.")
                    break
                else:
                    # If incomplete, use the agent to generate a polite explanation
                    # instead of just printing a system error message.
                    explanation_prompt = (
                        f"The user wants to stop, but we are missing the following details: {', '.join(missing_fields)}. "
                        "Please explain to the user that we need a little more information on these specific points "
                        "to generate a complete summary. Be polite and conversational."
                    )

                    # Run agent specifically to generate the "missing info" explanation
                    explanation_result = await fireside_chat_agent.run(
                        explanation_prompt, message_history=message_history
                    )

                    print(f"Agent: {explanation_result.output.message}")

                    # We add this interaction to history so the agent remembers why it stopped
                    message_history.extend(explanation_result.new_messages())

                    # Continue the loop so the user can provide the missing info
                    continue

            # Normal conversation flow
            result = await fireside_chat_agent.run(
                user_input, message_history=message_history
            )

            # Extract data from the result
            response_turn = result.output
            final_output_state = response_turn.current_data

            # Update history
            message_history.extend(result.new_messages())

            # Print agent's natural language response
            print(f"Agent: {response_turn.message}")

    except KeyboardInterrupt:
        print("\n\nAgent: Conversation ended by user.")


if __name__ == "__main__":
    asyncio.run(main())
