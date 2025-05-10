from pathlib import Path

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent

def get_relative_path(*path_parts) -> Path:
    """Get a path relative to the project root."""
    return get_project_root().joinpath(*path_parts)

# Default output directory
DEFAULT_OUTPUT_DIR = get_relative_path("output") 