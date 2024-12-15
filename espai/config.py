import enum
from dataclasses import dataclass

from espai import Framing
from espai.fit import DistFamily, LossFunction
from espai.aggregate import AggMethod


@dataclass(frozen=True)
class Config:
    framing: Framing | None  # `None` means all framings
    family: DistFamily
    aggregation: AggMethod
    loss_function: LossFunction | None
