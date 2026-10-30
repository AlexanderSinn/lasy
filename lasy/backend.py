import os

lasy_backend = "AUTO"
if "LASY_BACKEND" in os.environ:
    assert os.environ["LASY_BACKEND"] in ["NP", "CP", "AUTO"], (
        "The enviroment variable 'LASY_BACKEND' must be one of "
        + "'NP' (NumPy), 'CP' (CuPy) or 'AUTO'!"
    )
    lasy_backend = os.environ["LASY_BACKEND"]

if lasy_backend == "AUTO":
    try:
        import cupy as xp
        from cupyx.scipy.interpolate import RegularGridInterpolator
        from cupyx.scipy.signal import hilbert, zoom_fft
        from cupyx.scipy.special import j0

        # xp.is_available() might cause a CUDARuntimeError
        lasy_backend = "CP" if xp.is_available() else "NP"

        xp, RegularGridInterpolator, hilbert, zoom_fft, j0  # workaround for pyflakes
    except Exception:
        lasy_backend = "NP"

    print("LASY: using backend", lasy_backend)

if lasy_backend == "CP":
    import cupy as xp
    from cupyx.scipy.interpolate import RegularGridInterpolator
    from cupyx.scipy.signal import hilbert, zoom_fft
    from cupyx.scipy.special import j0

    use_cupy = True

    def to_cpu(arr):
        """Convert array from cupy to numpy."""
        if isinstance(arr, xp.ndarray):
            return xp.asnumpy(arr)
        elif isinstance(arr, list):
            return [to_cpu(a) for a in arr]
        elif isinstance(arr, tuple):
            return (to_cpu(a) for a in arr)
        else:
            return arr

    def to_gpu(arr):
        """Convert array from numpy to cupy."""
        if not isinstance(arr, xp.ndarray):
            return xp.asarray(arr)
        else:
            return arr

else:
    import numpy as xp
    from scipy.interpolate import RegularGridInterpolator
    from scipy.signal import hilbert, zoom_fft
    from scipy.special import j0

    use_cupy = False

    def to_cpu(arr):
        """Convert array from cupy to numpy."""
        return arr

    def to_gpu(arr):
        """Convert array from numpy to cupy."""
        return arr

import numpy as np
def chunk(*arrays, max_size=100, dims=None):
    def get_slice(a, b, e):
        return a[
            tuple( slice(int(b[i]), int(e[i])) if a.shape[i] > 1 else slice(None, None) for i in range(len(e)))
        ]

    s = np.broadcast_shapes(*[a.shape for a in arrays])

    total_size = np.array(s)
    tile_size = total_size.copy()

    if dims is None:
        dims = tuple(range(len(tile_size)))

    prod = 1
    for i in range(len(s) - 1, -1, -1):
        if i not in dims:
            prod *= total_size[i]

    for i in range(len(s) - 1, -1, -1):
        if i in dims:
            tile_size[i] = min(
                total_size[i],
                    max(1, max_size // prod)
                )
            prod *= tile_size[i]

    num_tiles = total_size // tile_size

    begin = np.zeros(len(num_tiles))

    while True:
        end = begin + tile_size
        yield tuple(get_slice(a, begin, end) for a in arrays)

        for i in range(len(s) -1, -1, -1):
            if end[i] < total_size[i]:
                begin[i] += tile_size[i]
                break
            else:
                begin[i] = 0
        else:
            break

__all__ = [
    "lasy_backend",
    "use_cupy",
    "xp",
    "to_cpu",
    "to_gpu",
    "RegularGridInterpolator",
    "hilbert",
    "zoom_fft",
    "j0",
    "chunk",
]
