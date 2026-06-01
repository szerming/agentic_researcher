import argparse
import asyncio
import os
import sys
from dotenv import load_dotenv

from agentic_researcher.graph import research_graph, SurveyNode
from agentic_researcher.state import ResearchState
from agentic_researcher.deps import ResearchDeps

async def async_main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Multi-Agent Technical Researcher Workflow")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("RESEARCH_MODEL", "gemini-2.5-flash-lite"),
        help="Google Gemini model name (default: gemini-2.5-flash-lite)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="research_report.md",
        help="Output markdown file path (default: research_report.md)"
    )
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY or GOOGLE_API_KEY environment variable must be set.", file=sys.stderr)
        print("Please check your .env file or environment setup.", file=sys.stderr)
        sys.exit(1)

    state = ResearchState()
    deps = ResearchDeps(
        model_name=args.model,
        api_key=api_key,
        output_filepath=args.output
    )

    print("==========================================")
    print("Welcome to the Technical Researcher Agentic Workflow!")
    print(f"Using Model: {args.model}")
    print(f"Output File: {args.output}")
    print("==========================================")

    try:
        result_path = await research_graph.run(
            inputs=SurveyNode(),
            state=state,
            deps=deps
        )
        print("\n==========================================")
        print(f"Success! Technical research report released to: {result_path}")
        print("==========================================")
    except Exception as e:
        print(f"\nWorkflow failed with error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nExited due to KeyboardInterrupt.")
        sys.exit(0)

if __name__ == "__main__":
    main()
