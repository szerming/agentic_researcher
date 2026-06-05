from pathlib import Path
from datetime import datetime
from pydantic import BaseModel


class FileUtils:
    @staticmethod
    def get_project_root() -> Path:
        return Path(__file__).parent.parent.parent.parent

    @staticmethod
    def get_outputs_path() -> Path:
        """
        Returns the path to the outputs directory.
        """
        project_root = FileUtils.get_project_root()
        outputs_dir = project_root / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        return outputs_dir

    @staticmethod
    def get_resource_path() -> Path:
        """
        Returns the path to the outputs directory.
        """
        project_root = FileUtils.get_project_root()
        resources_dir = project_root / "resources"
        resources_dir.mkdir(exist_ok=True)
        return resources_dir

    @staticmethod
    def write_temporary_markdown_file(
        content: str | list[str] | BaseModel, filename: str = "temp.txt"
    ) -> Path:
        if isinstance(content, list):
            content = "\n".join(content)
        elif isinstance(content, BaseModel):
            content = content.model_dump_json(indent=2)

        timestamp = FileUtils.get_timestamp()
        filename = f"{timestamp}_{filename}"
        filepath = FileUtils.get_outputs_path() / filename
        with open(filepath, "w") as f:
            f.write(content)
        return filepath

    @staticmethod
    def get_timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
