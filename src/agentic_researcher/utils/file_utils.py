import os
from pathlib import Path
from datetime import datetime

class FileUtils:
    @staticmethod
    def get_outputs_path() -> Path:
        """
        Returns the path to the outputs directory.
        """
        project_root = Path(__file__).parent.parent.parent
        outputs_dir = project_root / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        return outputs_dir

    def write_temporary_markdown_file(content: str | list[str], filename:str="temp.txt") -> Path:
        if isinstance(content, list):
            content = "\n".join(content)

        timestamp = FileUtils.get_timestamp()
        filename = f"{timestamp}_{filename}"
        filepath = FileUtils.get_outputs_path() / filename
        with open(filepath, "w") as f:
            f.write(content)
        return filepath 


    def get_timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
        