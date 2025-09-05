import numpy as np

def cmeans(
    data: np.ndarray,
    c: int,
    m: float,
    error: float,
    maxiter: int,
    init: np.ndarray | None = ...,
    seed: int | None = ...,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float, int, int]: ...
def cmeans_predict(
    test_data: np.ndarray,
    cntr_trained: np.ndarray,
    m: float,
    error: float,
    maxiter: int,
    init: np.ndarray | None = ...,
    seed: int | None = ...,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]: ...
