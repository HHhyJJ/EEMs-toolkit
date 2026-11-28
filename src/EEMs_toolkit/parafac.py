from __future__ import annotations

import time
from dataclasses import dataclass, field
from multiprocessing.pool import Pool
from typing import Callable, Iterable, Optional

import numpy as np
import tensorly as tl
import tlviz
from tensorly.cp_tensor import CPTensor
from tensorly.decomposition import non_negative_parafac_hals

from .constant import CROSS, N_SPLIT

ProgressCallback = Callable[[int, int, int], None]


class JobCancelled(Exception):
    """Raised when a long-running fit is stopped by the caller."""


@dataclass
class FitResult:
    """Result of fitting models with several component numbers."""

    factors: dict[int, CPTensor] = field(default_factory=dict)
    leverage: dict[int, np.ndarray] = field(default_factory=dict)
    sse: dict[int, np.ndarray] = field(default_factory=dict)
    core_consistency: dict[int, float] = field(default_factory=dict)
    explanation_rate: dict[int, float] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)


@dataclass
class SplitResult:
    """Result of split-half analysis."""

    split_result: dict[int, list[CPTensor]] = field(default_factory=dict)
    splits_index: list[np.ndarray] = field(default_factory=list)
    log: list[str] = field(default_factory=list)


def _fit_once(x, f: int, tol: float = 1e-6, max_iter: int = 2500, init: str = 'random', *,
              j: Optional[int] = None) -> tuple[CPTensor | tuple, int, int] | tuple[CPTensor | tuple, int]:
    """Fit one non-negative PARAFAC model, optionally tagged with a split index."""
    r = non_negative_parafac_hals(
        tensor=tl.tensor(x),
        rank=f,
        n_iter_max=max_iter,
        init=init, tol=tol,
        return_errors=False,
    )
    if j is not None:
        return r, f, j
    return r, f


def component_numbers(f: Optional[int | Iterable[int]], default: Iterable[int] = range(2, 6)) -> tuple[int, ...]:
    """Normalize the requested component numbers into a tuple."""
    if f is None:
        return tuple(default)
    if isinstance(f, int):
        return (f,)
    return tuple(int(i) for i in f)


def process_count(starts: int, n_processes: Optional[int] = None) -> int:
    """Number of worker processes, capped at 8 when starts is larger than 8."""
    if n_processes is not None:
        return max(1, int(n_processes))
    return max(1, min(int(starts), 8))


class _PoolRunner:
    """Submit fits to a process pool and poll them so the job can be cancelled."""

    def __init__(
            self,
            size: int,
            total: int,
            stop_event,
            progress: Optional[ProgressCallback],
            log: list[str],
            on_result: Callable[[tuple, tuple], None],
    ):
        self.pool = Pool(size)
        self.total = total
        self.stop_event = stop_event
        self.progress = progress
        self.log = log
        self.on_result = on_result
        self.done = 0
        self.terminated = False
        self.pending: list[tuple] = []

    def submit(self, args: tuple, key: tuple, kwargs: Optional[dict] = None) -> None:
        self.pending.append((self.pool.apply_async(_fit_once, args=args, kwds=kwargs or {}), key))

    def wait(self) -> None:
        while self.pending:
            if self.stop_event is not None and self.stop_event.is_set():
                self.pool.terminate()
                self.terminated = True
                self.pool.join()
                raise JobCancelled('Job stopped by user.')
            for item in list(self.pending):
                ar, key = item
                if not ar.ready():
                    continue
                self.pending.remove(item)
                try:
                    result = ar.get()
                except Exception as exc:  # noqa: BLE001 - keep fitting other starts
                    self.log.append(f'Fit failed: {exc}')
                else:
                    self.on_result(result, key)
                self.done += 1
                if self.progress is not None:
                    self.progress(self.done, self.total, key[0])
            if self.pending:
                time.sleep(0.02)

    def close(self) -> None:
        if not self.terminated:
            self.pool.close()
            self.pool.join()


def fit_models(
        x: np.ndarray,
        fs: Iterable[int],
        *,
        starts: int = 10,
        tol: float = 1e-6,
        max_iter: int = 2500,
        n_processes: Optional[int] = None,
        progress: Optional[ProgressCallback] = None,
        stop_event=None,
) -> FitResult:
    """
    Fit several starts for every requested number of components and keep the best model.

    :param x: EEM data with shape (nSample, nem, nex).
    :param fs: Numbers of components to fit.
    :param starts: Number of random initializations per number of components.
    :param tol: Convergence tolerance of the HALS algorithm.
    :param max_iter: Maximum number of iterations of one fit.
    :param n_processes: Worker processes, defaults to min(starts, 8).
    :param progress: Optional callback called as progress(done, total, f).
    :param stop_event: Optional threading.Event used to cancel the job.
    """
    x = np.asarray(x)
    if np.any(np.isnan(x)):
        raise ValueError("x contains nan.")
    fs = tuple(int(i) for i in fs)
    if not fs:
        raise ValueError("No component number is given.")
    starts = int(starts)
    if starts < 1:
        raise ValueError("starts should be a positive integer.")
    log: list[str] = []
    raw: dict[int, list] = {fi: [] for fi in fs}

    def on_result(result: tuple, key: tuple) -> None:
        _, fi = result
        raw[fi].append(result[0])
        log.append(f'A {fi}-component model has been fitted.')

    runner = _PoolRunner(
        process_count(starts, n_processes),
        total=len(fs) * starts,
        stop_event=stop_event,
        progress=progress,
        log=log,
        on_result=on_result,
    )
    try:
        for fi in fs:
            for _ in range(starts):
                runner.submit((x, fi, tol, max_iter), (fi,))
        runner.wait()
    finally:
        runner.close()

    out = FitResult(log=log)
    for fi in fs:
        if not raw[fi]:
            raise RuntimeError(f'All fits failed for {fi} components.')
        best_model = tlviz.multimodel_evaluation.get_model_with_lowest_error(raw[fi], x)
        best_model = tlviz.factor_tools.distribute_weights_in_one_mode(best_model, mode=0)
        best_model = tlviz.factor_tools.permute_cp_tensor(best_model)
        out.factors[fi] = best_model
        model = tl.cp_tensor.cp_to_tensor(best_model)
        out.leverage[fi] = tlviz.outliers.compute_leverage(best_model[1][0])
        out.sse[fi] = tlviz.outliers.compute_slabwise_sse(model, x, normalise=True)
        out.core_consistency[fi] = tlviz.model_evaluation.core_consistency(best_model, x, normalised=True)
        out.explanation_rate[fi] = 1 - tlviz.model_evaluation.relative_sse(best_model, x)
    return out


def split_fit(
        x: np.ndarray,
        factors: dict[int, CPTensor],
        fs: Iterable[int],
        *,
        starts: int = 10,
        tol: float = 1e-6,
        max_iter: int = 2500,
        random_state: Optional[int] = None,
        n_processes: Optional[int] = None,
        progress: Optional[ProgressCallback] = None,
        stop_event=None,
) -> SplitResult:
    """
    Generate PARAFAC models nested in dataset splits (S4C6T3).

    :param x: EEM data with shape (nSample, nem, nex).
    :param factors: Overall models, used to align the order of components.
    :param fs: Numbers of components to fit.
    :param random_state: Optional seed of the sample shuffling.
    """
    x = np.asarray(x)
    if np.any(np.isnan(x)):
        raise ValueError("x contains nan.")
    fs = tuple(int(i) for i in fs)
    if not fs:
        raise ValueError("No component number is given.")
    starts = int(starts)
    log: list[str] = []

    index = np.arange(x.shape[0])
    rng = np.random.RandomState(random_state)
    rng.shuffle(index)
    index_split = [index[i * x.shape[0] // N_SPLIT:(i + 1) * x.shape[0] // N_SPLIT] for i in range(N_SPLIT)]
    splits = [x[idx] for idx in index_split]
    n_slots = len(CROSS) * 2
    raw: dict[int, list[list]] = {fi: [[] for _ in range(n_slots)] for fi in fs}

    def on_result(result: tuple, key: tuple) -> None:
        _, fi, slot = result
        raw[fi][slot].append(result[0])
        log.append(f'A {fi}-component model has been fitted.')

    runner = _PoolRunner(
        process_count(starts, n_processes),
        total=len(fs) * n_slots * starts,
        stop_event=stop_event,
        progress=progress,
        log=log,
        on_result=on_result,
    )
    try:
        for j, (n1, n2) in enumerate(CROSS):
            c_x1 = np.concatenate((splits[n1[0]], splits[n1[1]]), axis=0)
            c_x2 = np.concatenate((splits[n2[0]], splits[n2[1]]), axis=0)
            for fi in fs:
                for _ in range(starts):
                    runner.submit((c_x1, fi, tol, max_iter), (fi, 2 * j), {'j': 2 * j})
                    runner.submit((c_x2, fi, tol, max_iter), (fi, 2 * j + 1), {'j': 2 * j + 1})
        runner.wait()
    finally:
        runner.close()

    out = SplitResult(splits_index=index_split, log=log)
    for fi in fs:
        models: list[CPTensor] = []
        for ji, candidates in enumerate(raw[fi]):
            c1, c2 = divmod(ji, 2)
            half = CROSS[c1][c2]
            c_x = np.concatenate((splits[half[0]], splits[half[1]]), axis=0)
            if not candidates:
                raise RuntimeError(f'All split fits failed for {fi} components (split {ji}).')
            best_model = tlviz.multimodel_evaluation.get_model_with_lowest_error(candidates, c_x)
            best_model = tlviz.factor_tools.distribute_weights_in_one_mode(best_model, mode=0)
            order = tlviz.factor_tools.get_factor_matrix_permutation(factors[fi][1][1], best_model[1][1])
            best_model = tlviz.factor_tools.permute_cp_tensor(best_model, permutation=tuple(order))
            models.append(best_model)
        out.split_result[fi] = models
    return out


def split_validation(
        factors: dict[int, CPTensor],
        split_result: dict[int, list[CPTensor]],
        fs: Iterable[int],
) -> dict[int, dict]:
    """
    Compare split models with the overall model using Tucker congruence coefficients.

    :return: Mapping of component number to {min_tcc, validated, em_tcc, ex_tcc}.
    """
    out: dict[int, dict] = {}
    for fi in fs:
        fi = int(fi)
        ex_l = [
            tlviz.factor_tools.cosine_similarity(factors[fi][1][2], ti[1][2])
            for ti in split_result[fi]
        ]
        em_l = [
            tlviz.factor_tools.cosine_similarity(factors[fi][1][1], ti[1][1])
            for ti in split_result[fi]
        ]
        for i in range(len(CROSS)):
            ex_l.append(
                tlviz.factor_tools.cosine_similarity(
                    split_result[fi][2 * i][1][2],
                    split_result[fi][2 * i + 1][1][2],
                )
            )
            em_l.append(
                tlviz.factor_tools.cosine_similarity(
                    split_result[fi][2 * i][1][1],
                    split_result[fi][2 * i + 1][1][1],
                )
            )
        m1 = float(np.min(ex_l))
        m2 = float(np.min(em_l))
        mm = min(m1, m2)
        out[fi] = {
            'min_tcc': mm,
            'validated': bool(mm >= 0.95),
            'em_tcc': [float(i) for i in em_l],
            'ex_tcc': [float(i) for i in ex_l],
        }
    return out
