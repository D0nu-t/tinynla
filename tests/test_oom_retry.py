import pytest
import torch

from nla import utils
from nla.utils import is_cuda_oom, retry_on_cuda_oom


def test_retries_oom_then_succeeds(monkeypatch):
    monkeypatch.setattr(utils, "wait_for_gpu", lambda *a, **k: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("CUDA error: out of memory")
        return "ok"

    assert retry_on_cuda_oom(flaky, tries=5) == "ok" and len(calls) == 3


def test_other_errors_and_exhaustion_raise(monkeypatch):
    monkeypatch.setattr(utils, "wait_for_gpu", lambda *a, **k: None)
    with pytest.raises(ValueError):
        retry_on_cuda_oom(lambda: (_ for _ in ()).throw(ValueError("bad")), tries=5)

    def always():
        raise torch.OutOfMemoryError("CUDA out of memory")
    with pytest.raises(torch.OutOfMemoryError):
        retry_on_cuda_oom(always, tries=3)


def test_is_cuda_oom():
    assert is_cuda_oom(RuntimeError("CUDA error: out of memory"))
    assert not is_cuda_oom(RuntimeError("shape mismatch"))


def test_wait_for_gpu_survives_failing_empty_cache(monkeypatch):
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda s: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def boom():
        raise RuntimeError("CUDA error: out of memory")
    monkeypatch.setattr(torch.cuda, "empty_cache", boom)
    utils.wait_for_gpu(1, wait=0)            # must not raise
