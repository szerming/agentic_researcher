from anthropic.types import effort_capability
from typing import Union
from pydantic_graph import BaseNode, End, GraphBuilder, GraphRunContext
from agentic_researcher.state import (
    ResearchState, TopicFindings, SubtopicFindings
)
from agentic_researcher.deps import ResearchDeps, get_model
from agentic_researcher.agents.survey import get_survey_agent, SurveyComplete
from agentic_researcher.agents.planning import get_planning_agent
from agentic_researcher.agents.researcher import get_researcher_agent
from agentic_researcher.agents.editor import get_editor_agent
from agentic_researcher.agents.writer import get_writer_agent
from agentic_researcher.agents.proofreader import get_proofreader_agent
from loguru import logger

class SurveyNode(BaseNode[ResearchState, ResearchDeps]):
    async def run(self, ctx: GraphRunContext[ResearchState, ResearchDeps]) -> 'ResearchPlanningNode':
        logger.info("🤖🤔 Survey Phase...")
        model = get_model(ctx.deps.model_name, ctx.deps.api_key)
        agent = get_survey_agent(model)

        message_history = None
        prompt_msg = "Hello! I am your research Survey Agent. What technical topic would you like to research today?"

        while True:
            print(f"\n[Survey Agent] 👋👋: {prompt_msg}")
            try:
                user_input = input("🤔You: 👉 ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nSurvey cancelled.")
                raise

            if not user_input:
                continue

            result = await agent.run(user_input, message_history=message_history)
            data = result.output

            message_history = result.all_messages()
            # logger.debug(f"🎨 [Survey Agent] {message_history=}")

            if isinstance(data, SurveyComplete):
                ctx.state.survey_data = data.survey_data
                logger.info("🤖🤔 Great! I have gathered all requirements.")
                break
            else:
                prompt_msg = data.question

        logger.debug(f"🤖🤔 [Survey data] {ctx.state.survey_data}")
        return ResearchPlanningNode()

class ResearchPlanningNode(BaseNode[ResearchState, ResearchDeps]):
    async def run(self, ctx: GraphRunContext[ResearchState, ResearchDeps]) -> 'ResearcherNode':
        logger.info("🤖📐 Research Planning Phase...")
        model = get_model(ctx.deps.model_name, ctx.deps.api_key)
        agent = get_planning_agent(model)

        survey_data_str = ctx.state.survey_data.model_dump_json(indent=2)
        prompt = f"Please generate a research plan for these requirements:\n{survey_data_str}"

        count = 1
        while True:
            print(f"\nGenerating research plan ({count})...")
            count += 1
            result = await agent.run(prompt)
            plan = result.output

            # Format and print the plan
            print("\n--- Proposed Research Plan ---")
            for i, topic in enumerate(plan.topics, 1):
                print(f"\nTopic {i}: {topic.name}")
                print(f"  Description: {topic.description}")
                print("  Subtopics:")
                for sub in topic.subtopics:
                    print(f"    - {sub.name}: {sub.description}")
            print("------------------------------")

            logger.info("🤖📐 Research Planning Phase completed. ")
            print("\nOptions: 👉 ")
            print("1. Enter 'approve' to approve this research plan and proceed to web research.")
            print("2. Enter any comments, additions, or changes you want to make to the plan.")

            try:
                user_input = input("\nYour choice/feedback: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nPlanning cancelled.")
                raise

            if user_input.lower() == 'approve':
                ctx.state.research_plan = plan
                print("\nResearch plan approved!")
                logger.debug(f"🤖📐 [Research plan] {plan}")
                break
            elif not user_input:
                continue
            else:
                prompt = (
                    f"Initial Requirements:\n{survey_data_str}\n\n"
                    f"Previous Plan:\n{plan.model_dump_json(indent=2)}\n\n"
                    f"User Feedback: {user_input}\n\n"
                    "Please update the research plan based on this feedback."
                )

        return ResearcherNode()

class ResearcherNode(BaseNode[ResearchState, ResearchDeps]):
    async def run(self, ctx: GraphRunContext[ResearchState, ResearchDeps]) -> 'EditorNode':
        logger.info("🤖🤓 Research Phase...")
        model = get_model(ctx.deps.model_name, ctx.deps.api_key)
        agent = get_researcher_agent(model)

        plan = ctx.state.research_plan
        all_topic_findings = []

        for topic in plan.topics:
            logger.info(f"🤖🤓 Researching Topic: {topic.name}...")
            sub_findings_list = []

            for sub in topic.subtopics:
                logger.info(f"🤖🤓 Researching Subtopic: {sub.name}...")
                researcher_prompt = (
                    f"Topic: {topic.name}\n"
                    f"Subtopic Name: {sub.name}\n"
                    f"Subtopic Description: {sub.description}\n\n"
                    "Please search the web and extract key factual findings for this subtopic."
                )
                result = await agent.run(researcher_prompt)
                findings: SubtopicFindings = result.output
                sub_findings_list.append(findings)
                logger.info(f"🤖🤓 \t\t-> Extracted {len(findings.findings)} findings.")

            all_topic_findings.append(TopicFindings(
                topic_name=topic.name,
                subtopics=sub_findings_list
            ))

        ctx.state.findings = all_topic_findings
        logger.info("🤖🤓 Research completed.")
        return EditorNode()

class EditorNode(BaseNode[ResearchState, ResearchDeps]):
    async def run(self, ctx: GraphRunContext[ResearchState, ResearchDeps]) -> 'WriterNode':
        logger.info("🤖🪜 Editing Phase...")
        model = get_model(ctx.deps.model_name, ctx.deps.api_key)
        agent = get_editor_agent(model)

        survey_str = ctx.state.survey_data.model_dump_json(indent=2)
        plan_str = ctx.state.research_plan.model_dump_json(indent=2)
        findings_str = "\n\n".join([f.model_dump_json(indent=2) for f in ctx.state.findings])

        prompt = (
            f"Survey Requirements:\n{survey_str}\n\n"
            f"Research Plan:\n{plan_str}\n\n"
            f"Research Findings:\n{findings_str}\n\n"
            "Please design the report skeleton matching these inputs."
        )

        result = await agent.run(prompt)
        ctx.state.skeleton = result.output
        logger.info("🤖🪜 Report skeleton successfully generated.")
        return WriterNode()

class WriterNode(BaseNode[ResearchState, ResearchDeps]):
    async def run(self, ctx: GraphRunContext[ResearchState, ResearchDeps]) -> 'ProofReadNode':
        logger.info("🤖📝 Writing Phase...")
        model = get_model(ctx.deps.model_name, ctx.deps.api_key)
        agent = get_writer_agent(model)

        survey_str = ctx.state.survey_data.model_dump_json(indent=2)
        skeleton_str = ctx.state.skeleton.model_dump_json(indent=2)

        if ctx.state.proofreader_feedback and ctx.state.draft_report:
            logger.info(f"🤖📝 Revising draft (Revision Cycle {ctx.state.iteration_count})...")
            feedback_str = "\n".join([f"- {f}" for f in ctx.state.proofreader_feedback])
            prompt = (
                f"Survey Requirements:\n{survey_str}\n\n"
                f"Report Skeleton:\n{skeleton_str}\n\n"
                f"Previous Draft Report:\n{ctx.state.draft_report}\n\n"
                f"Proofreader Feedback:\n{feedback_str}\n\n"
                "Please rewrite and revise the draft report to fully address the proofreader feedback."
            )
        else:
            logger.info("🤖📝 Drafting initial report...")
            prompt = (
                f"Survey Requirements:\n{survey_str}\n\n"
                f"Report Skeleton:\n{skeleton_str}\n\n"
                "Please write the first full draft of the technical research report."
            )

        result = await agent.run(prompt)
        ctx.state.draft_report = result.output
        logger.info("🤖📝 Report draft generated successfully.")
        return ProofReadNode()

class ProofReadNode(BaseNode[ResearchState, ResearchDeps]):
    MAX_ITERATIONS = 3
    
    async def run(self, ctx: GraphRunContext[ResearchState, ResearchDeps]) -> Union['WriterNode', End[str]]:
        logger.info("🤖👩🏻‍💻 Proof-Read Phase...")
        model = get_model(ctx.deps.model_name, ctx.deps.api_key)
        agent = get_proofreader_agent(model)

        survey_str = ctx.state.survey_data.model_dump_json(indent=2)
        draft_str = ctx.state.draft_report

        prompt = (
            f"Survey Requirements:\n{survey_str}\n\n"
            f"Draft Report:\n{draft_str}\n\n"
            "Please proofread this report draft against the survey requirements."
        )

        result = await agent.run(prompt)
        res_data = result.output

        if res_data.satisfied:
            logger.info("✅️ [Proofreader] The report is satisfactory! Releasing report...")
            with open(ctx.deps.output_filepath, "w", encoding="utf-8") as f:
                f.write(ctx.state.draft_report)


            return End(ctx.deps.output_filepath)
        else:
            ctx.state.iteration_count += 1
            logger.info(f"🤖👩🏻‍💻 [Proofreader] Feedback received (Cycle {ctx.state.iteration_count}/{self.MAX_ITERATIONS}):")
            feedbacks = "\n".join([f"{fb}" for fb in res_data.feedback])
            logger.info(f"🤖👩🏻‍💻 feedback: \n {feedbacks}")

            if ctx.state.iteration_count >= self.MAX_ITERATIONS:
                logger.warning("🤖👩🏻‍💻 Max revision iterations reached. Releasing final draft as-is.")
                warning_header = (
                    "<!-- WARNING: Max revision iterations (3) reached. "
                    "The proofreader was not fully satisfied with the following points:\n"
                    + "\n".join([f"- {f}" for f in res_data.feedback])
                    + "\n-->\n\n"
                )
                with open(ctx.deps.output_filepath, "w", encoding="utf-8") as f:
                    f.write(warning_header + ctx.state.draft_report)
                return End(ctx.deps.output_filepath)

            ctx.state.proofreader_feedback = res_data.feedback
            return WriterNode()

# Graph Builder Setup
builder = GraphBuilder(
    state_type=ResearchState,
    deps_type=ResearchDeps,
    output_type=str
)

# Register nodes
builder.add(builder.node(SurveyNode))
builder.add(builder.node(ResearchPlanningNode))
builder.add(builder.node(ResearcherNode))
builder.add(builder.node(EditorNode))
builder.add(builder.node(WriterNode))
builder.add(builder.node(ProofReadNode))

# Connect start node to Entry Node
builder.add(builder.edge_from(builder.start_node).to(SurveyNode))

# Build graph
research_graph = builder.build()
