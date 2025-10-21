try:
    from .lib import (
        hello,
        matmul,
        initBuffers,
        toGPU,
        toCPU,
        updateGpuMemory,
        linear,
        linearBack,
        initBuff,
        relu,
        reluBack,
        matMatSub,
        maxpool,
    )
except Exception:
    from .native_fallback import native as _fb

    hello = lambda: print("hello (fallback)")
    matmul = _fb.matmul
    relu = _fb.relu
    reluBack = _fb.reluBack
    toGPU = _fb.toGPU
    toCPU = _fb.toCPU
    updateGpuMemory = _fb.updateGpuMemory
    linear = _fb.linear
    linearBack = _fb.linearBack
    initBuffers = _fb.initBuffers
    initBuff = _fb.initBuff
    matMatSub = _fb.matMatSub
    maxpool = _fb.maxpool

__all__ = [
    "hello",
    "matmul",
    "relu",
    "reluBack",
    "toGPU",
    "initBuffers",
    "updateGpuMemory",
    "linear",
    "linearBack",
    "toCPU",
    "initBuff",
    "matMatSub",
    "maxpool",
]
