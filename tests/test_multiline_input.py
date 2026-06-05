from agentic_researcher.utils.multiline_input import MultilineInput
from unittest.mock import patch
import pytest


class TestMultilineInput:
    @pytest.mark.skip("For manual testing only")
    def test_user_manual_input(self):
        multiline_input = MultilineInput()
        text = multiline_input.get_multiline_input("Please enter your name:")
        print(f"User input:\n>>{text}<<")

    def test_get_multiline_input(self, capsys):
        # 1. Define the simulated user inputs.
        # The last empty string simulates the user pressing ENTER twice.
        multiline_inputs = [
            "Hello World",
            "This is line two",
            "third line with trailing spaces   ",
            "",
        ]

        # 2. Patch the built-in 'input' function
        with patch("builtins.input", side_effect=multiline_inputs) as mock_input:
            # 3. Call the method
            result = MultilineInput.get_multiline_input("Enter your story")

            # 4. Assertions
            # Check if the returned text correctly joins the inputs with newlines
            assert (
                result
                == "Hello World\nThis is line two\nthird line with trailing spaces"
            )

            # Verify that input() was called exactly 3 times
            assert mock_input.call_count == 4
