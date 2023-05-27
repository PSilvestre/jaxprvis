from dataclasses import dataclass
from typing import Tuple

Color = Tuple[int, int, int]


@dataclass
class Colorscheme:
    background: Color
    arrow: Color
    op: Color
    inner_computation: Color
    tensor_intermediate: Color
    tensor_input: Color
    tensor_output: Color


nord = Colorscheme(
    background=(46, 52, 64),
    arrow=(216, 222, 233),
    op=(76, 86, 106),
    inner_computation=(180, 142, 173),
    tensor_intermediate=(129, 161, 193),
    tensor_input=(163, 190, 140),
    tensor_output=(208, 135, 112),
)
