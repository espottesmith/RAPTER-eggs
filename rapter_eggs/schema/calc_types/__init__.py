from pathlib import Path

try:
    import rapter_eggs.schema.calc_types.enums
except ImportError:
    import rapter_eggs.schema.calc_types.generate

from rapter_eggs.schema.calc_types.enums import CalcType, LevelOfTheory, TaskType
from rapter_eggs.schema.calc_types.utils import calc_type, level_of_theory, task_type
