import pytest
from agentic_researcher.graph import research_graph
from agentic_researcher.utils.search import search_duckduckgo

def test_graph_structure():
    # Verify that the graph builds correctly and has all required nodes registered
    assert research_graph is not None
    assert "SurveyNode" in research_graph.nodes
    assert "ResearchPlanningNode" in research_graph.nodes
    assert "ResearcherNode" in research_graph.nodes
    assert "EditorNode" in research_graph.nodes
    assert "WriterNode" in research_graph.nodes
    assert "ProofReadNode" in research_graph.nodes

@pytest.mark.anyio
async def test_search_duckduckgo(monkeypatch):
    # Mock response HTML structured like lite.duckduckgo.com/lite/
    fake_html = """
    <html>
    <body>
    <a rel="nofollow" href="https://example.com/result1" class='result-link'>Example Result 1</a>
    <td class='result-snippet'>This is the first snippet for testing.</td>
    <a rel="nofollow" href="https://example.com/result2" class='result-link'>Example Result 2</a>
    <td class='result-snippet'>This is the second snippet for testing.</td>
    </body>
    </html>
    """

    class MockResponse:
        status_code = 200
        text = fake_html

    async def mock_post(*args, **kwargs):
        return MockResponse()

    # Monkeypatch the post method of httpx.AsyncClient
    monkeypatch.setattr("httpx.AsyncClient.post", mock_post)

    results = await search_duckduckgo("test query")
    assert len(results) == 2
    assert results[0]["title"] == "Example Result 1"
    assert results[0]["url"] == "https://example.com/result1"
    assert results[0]["snippet"] == "This is the first snippet for testing."
    assert results[1]["title"] == "Example Result 2"
    assert results[1]["url"] == "https://example.com/result2"
    assert results[1]["snippet"] == "This is the second snippet for testing."
