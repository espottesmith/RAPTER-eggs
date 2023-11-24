from typing import Dict, List

from rapter_eggs.schema.calc_types import TaskType

JAGUAR_FUNCTIONAL_QUALITY_SCORES: Dict[str, int] = {
    "wb97x-v": 4,
    "wb97x-d": 3,
    "CAM-B3LYP-D3": 3,
    "M11": 2,
    "PBE0": 2,
    "mn12-l": 1
}

JAGUAR_BASIS_QUALITY_SCORES: Dict[str, int] = {
    "def2-svp": 1,
    "def2-svpd(-f)": 2,
    "def2-svpd": 3,
    "def2-tzvpd": 4,
    "def2-tzvppd(-g)": 5,
}

JAGUAR_SOLVENT_MODEL_QUALITY_SCORES: Dict[str, int] = {
    "PCM": 3,
    "VACUUM": 1
}

JAGUAR_ALLOWED_TASK_TYPES: List[TaskType] = [t.value for t in TaskType]