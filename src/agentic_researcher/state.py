from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional


class QuestionAndAnswer(BaseModel):
    question: str
    answer: str


class Subtopic(BaseModel):
    name: str = Field(description="Name of the subtopic.")
    description: str = Field(
        description="Brief explanation of what this subtopic should cover."
    )


class TopicPlan(BaseModel):
    name: str = Field(description="Name of the main research topic.")
    description: str = Field(
        description="Description of what should be investigated under this main topic."
    )
    subtopics: List[Subtopic] = Field(
        default_factory=list,
        description="List of subtopics to investigate under this main topic.",
    )


class ResearchPlan(BaseModel):
    topics: List[TopicPlan] = Field(
        default_factory=list, description="List of planned topics to be researched."
    )


class SubtopicFindings(BaseModel):
    subtopic_name: str = Field(
        description="Name of the subtopic these findings relate to."
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Synthesized bullet-point facts and statements.",
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Source URLs or reference names used to gather these findings.",
    )

    def to_markdown(self) -> str:
        markdown = f"### {self.subtopic_name}\n"
        for finding in self.findings:
            markdown += f"- {finding}\n"
        if self.sources:
            markdown += "\n**Sources:**\n"
            for source in self.sources:
                markdown += f"- {source}\n"
        return markdown


class TopicFindings(BaseModel):
    topic_name: str = Field(description="Name of the main topic.")
    subtopics: List[SubtopicFindings] = Field(
        default_factory=list,
        description="Findings gathered for each subtopic under this main topic.",
    )

    def to_markdown(self) -> str:
        markdown = f"## {self.topic_name}\n"
        for subtopic in self.subtopics:
            markdown += subtopic.to_markdown()
        return markdown


class ReportSkeletonSection(BaseModel):
    title: str = Field(description="Heading title.")
    level: int = Field(
        description="Markdown heading level (e.g. 1 for #, 2 for ##, 3 for ###)."
    )
    bullets: List[str] = Field(
        default_factory=list,
        description="List of bulleted research findings to include in this section.",
    )
    subsections: List[ReportSkeletonSection] = Field(
        default_factory=list, description="Subsections nested under this section."
    )

    def to_outline(self) -> str:
        """Recursively converts the skeleton Pydantic model into a highly readable text string."""
        lines = []
        indent = "  " * (self.level - 1)
        lines.append(f"{indent}Heading Level {self.level}: {self.title}")
        lines.append(f"{indent}  Raw Findings to expand into prose:")

        for bullet in self.bullets:
            lines.append(f"{indent}  - {bullet}")

        for subsection in self.subsections:
            lines.append(subsection.to_outline())

        return "\n".join(lines)


class ReportSkeleton(BaseModel):
    title: str = Field(description="Title of the research report.")
    sections: List[ReportSkeletonSection] = Field(
        default_factory=list, description="List of top-level sections in the skeleton."
    )

    def to_outline(self) -> str:
        """Recursively converts the skeleton Pydantic model into a highly readable text string."""
        lines = []
        lines.append(f"Report Title: {self.title}")
        for sec in self.sections:
            lines.append(sec.to_outline())
        return "\n".join(lines)


class ResearchState(BaseModel):
    survey_data: Optional[ChatDetails] = Field(
        default=None,
        description="Detailed user requirements gathered during the survey.",
    )
    research_plan: Optional[ResearchPlan] = Field(
        default=None, description="Approved topics and subtopics for investigation."
    )
    findings: List[TopicFindings] = Field(
        default_factory=list,
        description="Bulleted facts and statements gathered by the Researcher agent.",
    )
    skeleton: Optional[ReportSkeleton] = Field(
        default=None,
        description="The outline skeleton of the report structured by the Editor agent.",
    )
    draft_report: Optional[str] = Field(
        default=None,
        description="The complete text report compiled by the Writer agent.",
    )
    proofreader_feedback: List[str] = Field(
        default_factory=list,
        description="Improvement feedback points provided by the Proofreader agent.",
    )
    iteration_count: int = Field(
        default=0,
        description="The number of draft-proofread revision cycles completed.",
    )


class ResearchPlanningDependencies(BaseModel):
    """Input to the research planning stage.
    This contains the survey data, forming the base requirements, and optional previous
    research plan and user feedback for revision.
    """

    survey_data: ChatDetails = Field(
        description="Details of the discussion on the topic, providing the foundational requirements for the research plan."
    )
    previous_plan: Optional[ResearchPlan] = Field(
        default=None,
        description="Optional previous research plan. If provided, the agent should review and potentially revise it based on user feedback.",
    )
    user_feedback: Optional[str] = Field(
        default=None,
        description="Optional user feedback on the previous plan. If provided, the agent should incorporate this feedback while revising the research plan.",
    )


class FiresideChatOutput(BaseModel):
    """The structured data we aim to populate through the conversation."""

    topic: str = Field(
        default=None, description="The main theme or topic of the fireside chat."
    )
    scope: str = Field(
        default=None,
        description="The scope of the discussion, defining limits or boundaries.",
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
    summary: str = Field(default=None, description="An summary of the conversation.")
    keypoints: list[str] = Field(
        default_factory=list,
        description="A list of at least 5 statements, information, details from the chat.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="A list of 3-7 core keywords or tags from the chat.",
    )

    def check_completeness(self) -> list[str]:
        """Checks if the FiresideChatOutput is fully populated."""

        missing_fields = []

        if not self.topic:
            missing_fields.append("topic")
        if not self.scope:
            missing_fields.append("scope")
        if not self.context:
            missing_fields.append("context")
        if not self.purpose:
            missing_fields.append("purpose")
        if not self.audience:
            missing_fields.append("audience")
        if not self.summary:
            missing_fields.append("summary")

        # Keywords should ideally have at least 3 items based on description,
        # but strictly we check if it's empty for basic completeness.
        if not self.keypoints or len(self.keypoints) < 5:
            missing_fields.append("keypoints")
        if not self.keywords:
            missing_fields.append("keywords")

        return missing_fields


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


class ChatDetails(FiresideChatOutput):
    questions_and_answers: list[QuestionAndAnswer] = Field(
        default_factory=list, description="The complete transcript of the conversation."
    )

    @staticmethod
    def build(
        fireside_chat_output: FiresideChatOutput,
        questions_and_answers: list[QuestionAndAnswer],
    ) -> ChatDetails:
        return ChatDetails(
            **fireside_chat_output.model_dump(),
            questions_and_answers=questions_and_answers,
        )
