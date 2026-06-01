from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional


class SurveyData(BaseModel):
    topic: str = Field(description="The core topic of interest for the research.")
    scope: str = Field(
        description="The scope of the research, defining limits or boundaries."
    )
    context: str = Field(
        description="The background context or target audience for the research."
    )
    use_case: str = Field(
        description="The primary use cases or applications for the report."
    )
    areas_of_interest: List[str] = Field(
        default_factory=list,
        description="Specific subtopics or questions the user is highly interested in.",
    )
    additional_notes: str = Field(
        default="",
        description="Any other constraints or instructions given by the user.",
    )


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
    survey_data: Optional[SurveyData] = Field(
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
