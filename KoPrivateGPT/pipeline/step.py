import types
from typing import Any, get_type_hints, Type, List


class Step:
    def __init__(self, instance: Any, function_name: str):
        self.instance = instance
        if not hasattr(self.instance, function_name):
            raise ValueError(f"Step does not have function {function_name}")
        self.function = getattr(self.instance, function_name)
        self.output_types: types.GenericAlias = get_type_hints(self.function)["return"]
        self.next_steps = []

    def execute(self, *args, **kwargs) -> Any:
        if len(self.next_steps) <= 0:
            return self.function(*args, **kwargs)

        output = self.function(*args, **kwargs)
        output = list(output)
        for next_step in self.next_steps:
            next_step.execute(*output)

    def connect(self, next_step: Type["Step"]):
        self._validate_next_step(next_step)
        self.next_steps.append(next_step)

    def _validate_next_step(self, next_step: Type["Step"]):
        type_hints = get_type_hints(next_step.function).items()
        input_type_list: List[type] = [value for name, value in type_hints]
        output_types: List[type] = list(self.output_types.__args__)
        assert (len(input_type_list) == len(output_types))
        for input_type, output_type in zip(input_type_list, output_types):
            assert (input_type == output_type)
