import ast
from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECK_MODULE = (
    PROJECT_ROOT
    / "src/openpi/models_pytorch/transformers_replace/models/siglip/check.py"
)


def test_transformers_dependency_matches_replacement_contract() -> None:
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(CHECK_MODULE.read_text()).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
    }
    expected_version = assignments["EXPECTED_TRANSFORMERS_VERSION"]

    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    pins = [
        dependency.split("==", maxsplit=1)[1]
        for dependency in project["project"]["dependencies"]
        if dependency.startswith("transformers==")
    ]
    assert pins == [expected_version]
