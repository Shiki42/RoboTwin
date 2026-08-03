import ast
from pathlib import Path
import tomllib

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECK_MODULE = PROJECT_ROOT / "src/openpi/models_pytorch/transformers_replace/models/siglip/check.py"
SIGLIP_MODULE = PROJECT_ROOT / "src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py"


def test_transformers_dependency_matches_replacement_contract() -> None:
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(CHECK_MODULE.read_text()).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    expected_version = assignments["EXPECTED_TRANSFORMERS_VERSION"]

    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    pins = [
        dependency.split("==", maxsplit=1)[1]
        for dependency in project["project"]["dependencies"]
        if dependency.startswith("transformers==")
    ]
    assert pins == [expected_version]


def test_siglip_patch_stem_stays_float32_under_bfloat16_autocast(monkeypatch) -> None:
    source = SIGLIP_MODULE.read_text()
    module = ast.parse(source)
    embedding = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "SiglipVisionEmbeddings"
    )
    forward = next(node for node in embedding.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    forward.decorator_list = []
    function = ast.FunctionDef(
        name="stem_forward",
        args=forward.args,
        body=forward.body,
        decorator_list=[],
    )
    namespace = {"torch": torch, "nn": torch.nn}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "<siglip-test>", "exec"),
        namespace,
    )

    patch_embedding = torch.nn.Conv2d(3, 8, kernel_size=2, stride=2)
    position_embedding = torch.nn.Embedding(4, 8)
    stub = type(
        "Stem",
        (),
        {
            "patch_embedding": patch_embedding,
            "position_embedding": position_embedding,
            "position_ids": torch.arange(4)[None],
        },
    )()
    seen = {}
    original = torch.nn.functional.conv2d
    monkeypatch.setattr(
        torch.nn.functional,
        "conv2d",
        lambda image, weight, *args, **kwargs: seen.update(image=image, weight=weight)
        or original(image, weight, *args, **kwargs),
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = namespace["stem_forward"](stub, torch.randn(2, 3, 4, 4))

    assert seen["image"].dtype == seen["weight"].dtype == output.dtype == torch.float32
