from typing import Union

from loguru import logger
from pydantic_graph import BaseNode, End, GraphBuilder, GraphRunContext

from agentic_researcher.agents.editor import get_editor_agent
from agentic_researcher.agents.planning import get_planning_agent
from agentic_researcher.agents.proofreader import get_proofreader_agent, ProofreadResult
from agentic_researcher.agents.researcher import get_researcher_agent
from agentic_researcher.agents.survey import get_survey_agent
from agentic_researcher.agents.writer import get_writer_agent
from agentic_researcher.deps import ResearchDeps, get_model
from agentic_researcher.state import (
    ResearchPlanningDependencies,
    FiresideChatOutput,
    ConversationTurn,
    QuestionAndAnswer,
    ChatDetails,
)
from agentic_researcher.state import ResearchState, TopicFindings, SubtopicFindings
from agentic_researcher.utils.file_utils import FileUtils
from agentic_researcher.utils.multiline_input import MultilineInput


class SurveyNode(BaseNode[ResearchState, ResearchDeps]):
    """Node for gathering survey data from the user through a series of question and answer."""

    async def run(
        self, ctx: GraphRunContext[ResearchState, ResearchDeps]
    ) -> "ResearchPlanningNode":
        # Track message history for context continuity
        message_history = []

        # Track the accumulated structured data
        final_output_state = FiresideChatOutput()

        logger.info("🤖📐 Research Planning Phase...")
        model = get_model(ctx.deps.model_name, ctx.deps.api_key)
        fireside_chat_agent = get_survey_agent(model=model)

        print("🔥 Fireside Chat Agent initialized.")
        # print("Agent: Hello! Welcome back to the fire chat. What should we talk about today?")
        # print("(Type 'stop' when you are ready to finish.)\n")

        conversation: list[QuestionAndAnswer] = []
        interviewer_question = (
            "Hello! Welcome back to the fire chat. What should we talk about today?"
        )
        is_finishing_conversation = False

        while True:
            # Get user input
            user_input = MultilineInput.get_multiline_input(interviewer_question)
            conversation.append(
                QuestionAndAnswer(question=interviewer_question, answer=user_input)
            )

            # Termination condition
            if user_input.lower().strip() in ["stop", "end", "exit"]:
                is_finishing_conversation = True

            if is_finishing_conversation:
                missing_fields = final_output_state.check_completeness()

                if not missing_fields:
                    logger.info("Finishing this chat. Thank you for your input. ")
                    logger.debug("\n--- ✅ Final Output Generated ---")
                    logger.debug(final_output_state.model_dump_json(indent=2))
                    logger.debug(
                        "\nAgent: Thanks for the great chat! Here is your summary."
                    )

                    chat_details = ChatDetails.build(
                        fireside_chat_output=final_output_state,
                        questions_and_answers=conversation,
                    )
                    ctx.state.survey_data = chat_details

                    # dump intermediate file
                    filename = FileUtils.write_temporary_markdown_file(
                        content=chat_details,
                        filename=f"{FileUtils.get_timestamp()}_chat.json",
                    )
                    logger.info(f"🤖🤓 Intermediate chat dumped to {filename}")

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

                    interviewer_question = explanation_result.output.message

                    # We add this interaction to history so the agent remembers why it stopped
                    message_history.extend(explanation_result.new_messages())

                    # Continue the loop so the user can provide the missing info
                    continue

            # Normal conversation flow
            result = await fireside_chat_agent.run(
                user_input, message_history=message_history
            )

            # Extract data from the result
            response_turn: ConversationTurn = result.output
            final_output_state = response_turn.current_data

            # Update history
            message_history.extend(result.new_messages())

            # Print agent's natural language response
            interviewer_question = response_turn.message

        return ResearchPlanningNode()


class ResearchPlanningNode(BaseNode[ResearchState, ResearchDeps]):
    """Node for planning the research based on the survey data."""

    async def run(
        self, ctx: GraphRunContext[ResearchState, ResearchDeps]
    ) -> "ResearcherNode":
        logger.info("🤖📐 Research Planning Phase...")
        model = get_model(ctx.deps.model_name, ctx.deps.api_key)
        agent = get_planning_agent(model)

        user_prompt = "Please generate a research plan based on the requirements provided in the SurveyData. "

        count = 1
        if ctx.state.survey_data is None:
            raise RuntimeError("Survey data not provided.")

        deps = ResearchPlanningDependencies(survey_data=ctx.state.survey_data)
        while True:
            print(f"\nGenerating research plan ({count})...")
            count += 1

            logger.debug(f"🤖📐 [Research Planning Dependencies] {deps=}")

            result = await agent.run(user_prompt=user_prompt, deps=deps)
            plan = result.output
            deps.previous_plan = plan

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
            print(
                "1. Enter 'approve' to approve this research plan and proceed to web research."
            )
            print(
                "2. Enter any comments, additions, or changes you want to make to the plan."
            )

            try:
                user_input = input("\nYour choice/feedback: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nPlanning cancelled.")
                logger.error("Planning cancelled by user.")
                raise

            if user_input.lower() == "approve":
                ctx.state.research_plan = plan
                print("\nResearch plan approved!")
                logger.debug(f"🤖📐 [Research plan] {plan}")
                break
            elif not user_input:
                continue
            else:
                deps.user_feedback = user_input

        return ResearcherNode()


class ResearcherNode(BaseNode[ResearchState, ResearchDeps]):
    async def run(
        self, ctx: GraphRunContext[ResearchState, ResearchDeps]
    ) -> "EditorNode":
        logger.info("🤖🤓 Research Phase...")
        model = get_model(ctx.deps.model_name, ctx.deps.api_key)
        agent = get_researcher_agent(model)

        plan = ctx.state.research_plan
        all_topic_findings = []

        for topic in plan.topics:
            logger.info(f"🤖🤓 Researching Topic: {topic.name}...")
            sub_findings_list = []

            for sub in topic.subtopics:
                logger.info(f"🤖🤓 \tResearching Subtopic: {sub.name}...")
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

            all_topic_findings.append(
                TopicFindings(topic_name=topic.name, subtopics=sub_findings_list)
            )

        ctx.state.findings = all_topic_findings
        logger.info("🤖🤓 Research completed.")

        # dump intermediate file
        findings: list[str] = [
            topic_findings.to_markdown() for topic_findings in all_topic_findings
        ]
        filename = FileUtils.write_temporary_markdown_file(
            content=findings, filename="research_output.md"
        )
        logger.info(f"🤖🤓 Intermediate research output dumped to {filename}")

        return EditorNode()


class EditorNode(BaseNode[ResearchState, ResearchDeps]):
    async def run(
        self, ctx: GraphRunContext[ResearchState, ResearchDeps]
    ) -> "WriterNode":
        logger.info("🤖🪜 Editing Phase...")
        model = get_model(ctx.deps.model_name, ctx.deps.api_key)
        agent = get_editor_agent(model)

        survey_str = ctx.state.survey_data.model_dump_json(indent=2)
        plan_str = ctx.state.research_plan.model_dump_json(indent=2)
        findings_str = "\n\n".join(
            [f.model_dump_json(indent=2) for f in ctx.state.findings]
        )

        prompt = (
            f"Survey Requirements:\n{survey_str}\n\n"
            f"Research Plan:\n{plan_str}\n\n"
            f"Research Findings:\n{findings_str}\n\n"
            "Please design the report skeleton matching these inputs."
        )

        result = await agent.run(prompt)
        ctx.state.skeleton = result.output
        logger.info("🤖🪜 Report skeleton successfully generated.")

        # dump intermediate file
        filename = FileUtils.write_temporary_markdown_file(
            content=result.output,
            filename=f"{FileUtils.get_timestamp()}_editor_skeleton.json",
        )
        logger.info(f"🤖🪜 Intermediate editor output dumped to {filename}")

        return WriterNode()


class WriterNode(BaseNode[ResearchState, ResearchDeps]):
    async def run(
        self, ctx: GraphRunContext[ResearchState, ResearchDeps]
    ) -> "ProofReadNode":
        logger.info("🤖📝 Writing Phase...")
        model = get_model(ctx.deps.model_name, ctx.deps.api_key)
        agent = get_writer_agent(model)

        survey_str = ctx.state.survey_data.model_dump_json(indent=2)
        skeleton_str = ctx.state.skeleton.to_outline()

        if ctx.state.proofreader_feedback and ctx.state.draft_report:
            logger.info(
                f"🤖📝 Revising draft (Revision Cycle {ctx.state.iteration_count})..."
            )
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

        filename = FileUtils.write_temporary_markdown_file(
            content=result.output,
            filename=f"{FileUtils.get_timestamp()}_writer_draft.json",
        )
        logger.info(f"🤖🪜 Draft writer output dumped to {filename}")

        return ProofReadNode()


class ProofReadNode(BaseNode[ResearchState, ResearchDeps]):
    MAX_ITERATIONS = 10

    async def run(
        self, ctx: GraphRunContext[ResearchState, ResearchDeps]
    ) -> Union["WriterNode", End[str]]:
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
        res_data: ProofreadResult = result.output

        filename = FileUtils.write_temporary_markdown_file(
            content=res_data,
            filename=f"{FileUtils.get_timestamp()}_reviewer_draft.json",
        )
        logger.info(f"🤖🪜 Draft writer output dumped to {filename}")

        if res_data.satisfied:
            logger.info(
                "✅️ [Proofreader] The report is satisfactory! Releasing report..."
            )
            with open(ctx.deps.output_filepath, "w", encoding="utf-8") as f:
                f.write(ctx.state.draft_report)

            return End(ctx.deps.output_filepath)
        else:
            ctx.state.iteration_count += 1
            logger.info(
                f"🤖👩🏻‍💻 [Proofreader] Feedback received (Cycle {ctx.state.iteration_count}/{self.MAX_ITERATIONS}):"
            )
            feedbacks = "\n".join([f"{fb}" for fb in res_data.feedback])
            logger.info(f"🤖👩🏻‍💻 feedback: \n {feedbacks}")

            if ctx.state.iteration_count >= self.MAX_ITERATIONS:
                logger.warning(
                    "🤖👩🏻‍💻 Max revision iterations reached. Releasing final draft as-is."
                )
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
logger.info("Building the graph for Agentic Researcher...")
builder = GraphBuilder(
    state_type=ResearchState, deps_type=ResearchDeps, output_type=str
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

logger.info("Agentic Researcher graph successfully built. Ready to run!")
mermaid_output = research_graph.render(title="Agentic Researcher Graph")
logger.debug(f"Mermaid diagram generated. {{mermaid_output={mermaid_output}}}")
