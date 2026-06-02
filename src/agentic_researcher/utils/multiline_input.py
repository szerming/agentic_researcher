class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[91m'
    GRAY = '\033[30m'
    PURPLE = '\033[35m'

class MultilineInput:
    
    @staticmethod
    def get_multiline_input(prompt: str) -> str:
        """
        Get multiline input from the user.
        Press ENTER twice to finish.
        """
        def get_input() -> str:
            return input().strip()

        print(f"\n👋👋: {BColors.PURPLE} {prompt} {BColors.ENDC}\n{BColors.GRAY}(Press ENTER twice to finish.){BColors.ENDC} ➡️")
        text = "\n".join(iter(get_input, ""))
        return text
