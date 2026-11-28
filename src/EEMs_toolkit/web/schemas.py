from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class HistoryJumpRequest(BaseModel):
    index: int = Field(ge=0)


class SubDatasetRequest(BaseModel):
    sample_orders: Optional[list[int]] = None
    ex_lt: Optional[float] = None
    ex_gt: Optional[float] = None
    em_lt: Optional[float] = None
    em_gt: Optional[float] = None


class DiluteRequest(BaseModel):
    factors: Optional[list[float]] = None


class SmoothRequest(BaseModel):
    sigma: float = 0.5
    filter_size: Optional[int] = None


class RamanRequest(BaseModel):
    ex: float = 350
    em_range: tuple[float, float] = (378, 424)


class CutScatterRequest(BaseModel):
    ray: tuple[float, float] = (15, 15)
    second_ray: tuple[float, float] = (20, 15)
    ram: tuple[float, float] = (15, 15)
    second_ram: tuple[float, float] = (10, 10)


class ParafacJobRequest(BaseModel):
    f: list[int] = Field(default_factory=lambda: [2, 3, 4, 5])
    starts: int = Field(default=10, ge=1, le=100)
    tol: float = 1e-6
    max_iter: int = Field(default=2500, ge=1)
    n_processes: Optional[int] = None


class SplitJobRequest(ParafacJobRequest):
    random_state: Optional[int] = None


class ExcludeSamplesRequest(BaseModel):
    orders: list[int]


class ProjectSaveRequest(BaseModel):
    include_history: bool = True
    f: Optional[int] = None


class ExportAllRequest(BaseModel):
    include: Optional[list[str]] = None
    f: Optional[int] = None
