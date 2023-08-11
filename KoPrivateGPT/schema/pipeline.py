from typing import NamedTuple, Dict, Any


class PipelineConfigAlias(NamedTuple):
    module_name: str
    init_params: Dict[str, Any]
