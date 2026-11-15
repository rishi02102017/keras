import os, traceback, numpy as np
import keras
from keras import ops
print("keras", keras.__version__, keras.backend.backend())
rng = np.random.default_rng(0)
a = rng.standard_normal((3, 4, 5)).astype("float32")
an = a.copy(); an[0, 1, 2] = np.nan; an[2, 3, 1] = np.nan
v = rng.standard_normal((7,)).astype("float32")
m = rng.standard_normal((4, 4)).astype("float32")
i = rng.integers(0, 5, (3, 4)).astype("int32")
C = []
def c(name, kf, nf, *args):
    C.append((name, kf, nf, args))
for ax in [None, 0, 1, -1, (0, 2), (1, 2)]:
    for kd in [False, True]:
        for nm in ["median", "nanmedian", "nanmean", "nanstd", "nanvar", "nansum", "nanprod", "nanmax", "nanmin", "ptp", "std", "var", "mean", "sum", "prod", "max", "min", "any", "all", "count_nonzero"]:
            if nm == "count_nonzero":
                if kd: continue
                c(f"{nm} ax={ax}", lambda x, nm=nm, ax=ax: getattr(ops, nm)(x, axis=ax), lambda x, nm=nm, ax=ax: getattr(np, nm)(x, axis=ax), an if nm.startswith("nan") else a)
                continue
            c(f"{nm} ax={ax} kd={kd}", lambda x, nm=nm, ax=ax, kd=kd: getattr(ops, nm)(x, axis=ax, keepdims=kd), lambda x, nm=nm, ax=ax, kd=kd: getattr(np, nm)(x, axis=ax, keepdims=kd), an if nm.startswith("nan") else a)
        for nm in ["quantile", "nanquantile", "percentile", "nanpercentile"]:
            q = [0.2, 0.7] if "quant" in nm else [20., 70.]
            for meth in ["linear", "lower", "higher", "midpoint", "nearest"]:
                c(f"{nm} ax={ax} kd={kd} {meth}", lambda x, nm=nm, ax=ax, kd=kd, q=q, meth=meth: getattr(ops, nm)(x, np.array(q, "float32"), axis=ax, keepdims=kd, method=meth), lambda x, nm=nm, ax=ax, kd=kd, q=q, meth=meth: getattr(np, nm)(x, q, axis=ax, keepdims=kd, method=meth), an if nm.startswith("nan") else a)
    for nm in ["nanargmax", "nanargmin", "argmax", "argmin"]:
        if ax is None or isinstance(ax, int):
            for kd in [False, True]:
                c(f"{nm} ax={ax} kd={kd}", lambda x, nm=nm, ax=ax, kd=kd: getattr(ops, nm)(x, axis=ax, keepdims=kd), lambda x, nm=nm, ax=ax, kd=kd: getattr(np, nm)(x, axis=ax, keepdims=kd), an if nm.startswith("nan") else a)
    if ax is None or isinstance(ax, int):
        for nm in ["cumsum", "cumprod", "nancumsum", "nancumprod"]:
            c(f"{nm} ax={ax}", lambda x, nm=nm, ax=ax: getattr(ops, nm)(x, axis=ax), lambda x, nm=nm, ax=ax: getattr(np, nm)(x, axis=ax), an if nm.startswith("nan") else a)
        c(f"average ax={ax}", lambda x, ax=ax: ops.average(x, axis=ax, weights=None), lambda x, ax=ax: np.average(x, axis=ax), a)
        if ax is not None:
            c(f"diff n=2 ax={ax}", lambda x, ax=ax: ops.diff(x, n=2, axis=ax), lambda x, ax=ax: np.diff(x, n=2, axis=ax), a)
            c(f"sort ax={ax}", lambda x, ax=ax: ops.sort(x, axis=ax), lambda x, ax=ax: np.sort(x, axis=ax), a)
            c(f"argsort ax={ax}", lambda x, ax=ax: ops.argsort(x, axis=ax), lambda x, ax=ax: np.argsort(x, axis=ax, kind="stable"), a)
            c(f"trapezoid ax={ax}", lambda x, ax=ax: ops.trapezoid(x, axis=ax), lambda x, ax=ax: np.trapezoid(x, axis=ax), a)
            c(f"argpartition ax={ax}", lambda x, ax=ax: ops.take_along_axis(x, ops.argpartition(x, 2, axis=ax), axis=ax)[...] , lambda x, ax=ax: np.take_along_axis(x, np.argpartition(x, 2, axis=ax), axis=ax), a)
            c(f"array_split 3 ax={ax}", lambda x, ax=ax: ops.concatenate([ops.sum(t) [None] for t in ops.array_split(x, 3, axis=ax)]), lambda x, ax=ax: np.stack([np.sum(t) for t in np.array_split(x, 3, axis=ax)]), a)
            c(f"roll ax={ax}", lambda x, ax=ax: ops.roll(x, 2, axis=ax), lambda x, ax=ax: np.roll(x, 2, axis=ax), a)
            c(f"repeat ax={ax}", lambda x, ax=ax: ops.repeat(x, 2, axis=ax), lambda x, ax=ax: np.repeat(x, 2, axis=ax), a)
c("average weights ax1", lambda x: ops.average(x, axis=1, weights=np.arange(1, 5, dtype="float32")), lambda x: np.average(x, axis=1, weights=np.arange(1, 5)), a)
c("average weights full", lambda x: ops.average(x, axis=None, weights=np.abs(a) + 1), lambda x: np.average(x, axis=None, weights=np.abs(a) + 1), a)
c("diff n=0", lambda x: ops.diff(x, n=0), lambda x: np.diff(x, n=0), a)
c("trapezoid x", lambda y: ops.trapezoid(y, x=np.cumsum(np.abs(v))), lambda y: np.trapezoid(y, x=np.cumsum(np.abs(v))), v)
c("trapezoid dx", lambda y: ops.trapezoid(y, dx=0.5, axis=1), lambda y: np.trapezoid(y, dx=0.5, axis=1), a)
c("histogram", lambda x: ops.histogram(x, bins=5)[0], lambda x: np.histogram(x, bins=5)[0], v)
c("histogram edges", lambda x: ops.histogram(x, bins=5)[1], lambda x: np.histogram(x, bins=5)[1], v)
c("histogram range", lambda x: ops.histogram(x, bins=4, range=(-1, 1))[0], lambda x: np.histogram(x, bins=4, range=(-1, 1))[0], v)
c("cov", lambda x: ops.cov(x), lambda x: np.cov(x), m)
c("cov 1d", lambda x: ops.cov(x), lambda x: np.cov(x), v)
c("corrcoef", lambda x: ops.corrcoef(x), lambda x: np.corrcoef(x), m)
c("corrcoef 1d", lambda x: ops.corrcoef(x[None]), lambda x: np.corrcoef(x[None]), v)
for mode in ["valid", "same", "full"]:
    c(f"correlate {mode}", lambda x, mode=mode: ops.correlate(x, x[:3], mode=mode), lambda x, mode=mode: np.correlate(x, x[:3], mode=mode), v)
    c(f"correlate swapped {mode}", lambda x, mode=mode: ops.correlate(x[:3], x, mode=mode), lambda x, mode=mode: np.correlate(x[:3], x, mode=mode), v)
c("digitize", lambda x: ops.digitize(x, np.array([-1., 0., 1.], "float32")), lambda x: np.digitize(x, [-1., 0., 1.]), v)
c("digitize decreasing", lambda x: ops.digitize(x, np.array([1., 0., -1.], "float32")), lambda x: np.digitize(x, [1., 0., -1.]), v)
for side in ["left", "right"]:
    c(f"searchsorted {side}", lambda x, side=side: ops.searchsorted(np.sort(v), x, side=side), lambda x, side=side: np.searchsorted(np.sort(v), x, side=side), np.concatenate([v[:3], [0.]]).astype("float32"))
c("bincount", lambda x: ops.bincount(x, minlength=7), lambda x: np.bincount(x, minlength=7), np.array([0, 1, 1, 3], "int32"))
c("bincount w", lambda x: ops.bincount(x, weights=np.array([.5, 1, 2, 3], "float32")), lambda x: np.bincount(x, weights=np.array([.5, 1, 2, 3], "float32")), np.array([0, 1, 1, 3], "int32"))
c("round -1", lambda x: ops.round(x, -1), lambda x: np.round(x, -1), np.array([14, 15, 25, -15, 123], "int32"))
c("round 2", lambda x: ops.round(x, 2), lambda x: np.round(x, 2), np.array([1.234, 2.345, -0.555], "float32"))
c("round half even", lambda x: ops.round(x), lambda x: np.round(x), np.array([0.5, 1.5, 2.5, -0.5], "float32"))
c("vander", lambda x: ops.vander(x, 3), lambda x: np.vander(x, 3), v)
c("vander inc", lambda x: ops.vander(x, increasing=True), lambda x: np.vander(x, increasing=True), v[:4])
c("trace off", lambda x: ops.trace(x, offset=1, axis1=1, axis2=2), lambda x: np.trace(x, offset=1, axis1=1, axis2=2), a)
c("trace negoff", lambda x: ops.trace(x, offset=-2), lambda x: np.trace(x, offset=-2), m)
c("diagonal off", lambda x: ops.diagonal(x, offset=-1, axis1=2, axis2=0), lambda x: np.diagonal(x, offset=-1, axis1=2, axis2=0), a)
c("tri", lambda x: ops.tri(3, 5, k=-1), lambda x: np.tri(3, 5, k=-1), a)
c("tril k", lambda x: ops.tril(x, k=-1), lambda x: np.tril(x, k=-1), a)
c("eye k", lambda x: ops.eye(3, 4, k=2), lambda x: np.eye(3, 4, k=2), a)
c("kaiser", lambda x: ops.kaiser(6, 2.0), lambda x: np.kaiser(6, 2.0), a)
c("nan_to_num", lambda x: ops.nan_to_num(x, nan=1.0, posinf=2.0, neginf=-2.0), lambda x: np.nan_to_num(x, nan=1.0, posinf=2.0, neginf=-2.0), np.array([np.nan, np.inf, -np.inf, 1.], "float32"))
c("isclose", lambda x: ops.isclose(x, x + 1e-3, rtol=0, atol=1e-2), lambda x: np.isclose(x, x + 1e-3, rtol=0, atol=1e-2), v)
c("isclose nan", lambda x: ops.isclose(x, x, equal_nan=True), lambda x: np.isclose(x, x, equal_nan=True), np.array([np.nan, 1.], "float32"))
c("isin invert", lambda x: ops.isin(x, np.array([1, 2], "int32"), invert=True), lambda x: np.isin(x, [1, 2], invert=True), i)
c("unravel", lambda x: ops.stack(ops.unravel_index(x, (2, 5))), lambda x: np.stack(np.unravel_index(x, (2, 5))), np.array([0, 3, 7, 9], "int32"))
c("logspace base", lambda x: ops.logspace(0, 2, 5, base=2.0), lambda x: np.logspace(0, 2, 5, base=2.0), a)
c("logspace noend", lambda x: ops.logspace(0, 2, 5, endpoint=False), lambda x: np.logspace(0, 2, 5, endpoint=False), a)
c("geomspace", lambda x: ops.geomspace(1, 1000, 4), lambda x: np.geomspace(1, 1000, 4), a)
c("geomspace neg", lambda x: ops.geomspace(-1, -1000, 4), lambda x: np.geomspace(-1, -1000, 4), a)
c("geomspace noend", lambda x: ops.geomspace(1, 1000, 3, endpoint=False), lambda x: np.geomspace(1, 1000, 3, endpoint=False), a)
c("linspace arr axis", lambda x: ops.linspace(np.array([0., 1.]), np.array([1., 3.]), 4, axis=1), lambda x: np.linspace(np.array([0., 1.]), np.array([1., 3.]), 4, axis=1), a)
c("linspace noend", lambda x: ops.linspace(0, 1, 4, endpoint=False), lambda x: np.linspace(0, 1, 4, endpoint=False), a)
c("cross", lambda x: ops.cross(x[:, :3], x[::-1, :3]), lambda x: np.cross(x[:, :3], x[::-1, :3]), m)
c("cross 2d", lambda x: ops.cross(x[:, :2], x[::-1, :2]), lambda x: np.cross(x[:, :2], x[::-1, :2]), m)
c("kron", lambda x: ops.kron(x, x[:2, :3]), lambda x: np.kron(x, x[:2, :3]), m)
c("ldexp", lambda x: ops.ldexp(x, np.array([1, 2, 3, 4], "int32")), lambda x: np.ldexp(x, np.array([1, 2, 3, 4], "int32")), m)
c("heaviside", lambda x: ops.heaviside(x, 0.5), lambda x: np.heaviside(x, 0.5), np.array([-1., 0., 2.], "float32"))
c("sinc", lambda x: ops.sinc(x), lambda x: np.sinc(x), np.array([0., 0.5, 1.], "float32"))
c("i0", lambda x: ops.i0(x), lambda x: np.i0(x), v)
c("signbit", lambda x: ops.signbit(x), lambda x: np.signbit(x), np.array([-0., 0., -1.], "float32"))
c("nextafter", lambda x: ops.nextafter(x, np.float32(2)), lambda x: np.nextafter(x, np.float32(2)), np.array([1.], "float32"))
c("select", lambda x: ops.select([x < 0, x > 1], [x, x * 2], default=7), lambda x: np.select([x < 0, x > 1], [x, x * 2], default=7), v)
c("pad reflect", lambda x: ops.pad(x, ((0, 0), (2, 1)), mode="reflect"), lambda x: np.pad(x, ((0, 0), (2, 1)), mode="reflect"), m)
c("pad symmetric", lambda x: ops.pad(x, ((1, 0), (2, 1)), mode="symmetric"), lambda x: np.pad(x, ((1, 0), (2, 1)), mode="symmetric"), m)
c("pad const", lambda x: ops.pad(x, 1, constant_values=3), lambda x: np.pad(x, 1, constant_values=3), m)
c("tile", lambda x: ops.tile(x, (2, 1, 1, 2)), lambda x: np.tile(x, (2, 1, 1, 2)), a)
c("tile short", lambda x: ops.tile(x, 2), lambda x: np.tile(x, 2), a)
c("moveaxis", lambda x: ops.moveaxis(x, [0, 1], [-1, 0]), lambda x: np.moveaxis(x, [0, 1], [-1, 0]), a)
c("squeeze tuple", lambda x: ops.squeeze(x[:, :1, :1], axis=(1, 2)), lambda x: np.squeeze(x[:, :1, :1], axis=(1, 2)), a)
c("take neg idx", lambda x: ops.take(x, np.array([-1, 0], "int32"), axis=1), lambda x: np.take(x, [-1, 0], axis=1), a)
c("take none", lambda x: ops.take(x, np.array([[1, 2]], "int32")), lambda x: np.take(x, [[1, 2]]), a)
c("take_along None", lambda x: ops.take_along_axis(x, np.array([3, 0], "int32"), axis=None), lambda x: np.take_along_axis(x, np.array([3, 0]), axis=None), a)
c("tensordot", lambda x: ops.tensordot(x, x, axes=([1, 2], [1, 2])), lambda x: np.tensordot(x, x, axes=([1, 2], [1, 2])), a)
c("einsum", lambda x: ops.einsum("ijk,ilk->jl", x, x), lambda x: np.einsum("ijk,ilk->jl", x, x), a)
c("meshgrid ij", lambda x: ops.stack(ops.meshgrid(x[:3], x[:2], indexing="ij")), lambda x: np.stack(np.meshgrid(x[:3], x[:2], indexing="ij")), v)
c("split idx", lambda x: ops.concatenate([ops.shape(t)[1] * ops.ones((1,)) for t in ops.split(x, [1, 3], axis=1)]), lambda x: np.array([t.shape[1] for t in np.split(x, [1, 3], axis=1)], "float32"), a)
c("unique", lambda x: ops.unique(x), lambda x: np.unique(x), i.ravel())
c("slogdet", lambda x: ops.stack(ops.slogdet(x)), lambda x: np.stack(np.linalg.slogdet(x)), m)
c("clip int", lambda x: ops.clip(x, 1, 3), lambda x: np.clip(x, 1, 3), i)
c("mod neg", lambda x: ops.mod(x, -3), lambda x: np.mod(x, -3), np.array([-5, 5, 7], "int32"))
c("fmod", lambda x: ops.fmod(x, 3), lambda x: np.fmod(x, 3), np.array([-5, 5, 7], "int32"))
c("floor_divide neg", lambda x: ops.floor_divide(x, -3), lambda x: np.floor_divide(x, -3), np.array([-5, 5, 7], "int32"))
c("gcd", lambda x: ops.gcd(x, 6), lambda x: np.gcd(x, 6), np.array([-4, 9, 0], "int32"))
c("lcm", lambda x: ops.lcm(x, 6), lambda x: np.lcm(x, 6), np.array([-4, 9, 0], "int32"))
c("hypot", lambda x: ops.hypot(x, 2.), lambda x: np.hypot(x, 2.), v)
c("logaddexp2", lambda x: ops.logaddexp2(x, 1.), lambda x: np.logaddexp2(x, 1.), v)
c("copysign", lambda x: ops.copysign(x, -1.), lambda x: np.copysign(x, -1.), v)
c("float_power", lambda x: ops.float_power(x, 2), lambda x: np.float_power(x, 2), np.abs(v))
c("cbrt", lambda x: ops.cbrt(x), lambda x: np.cbrt(x), v)
c("rot90 k2 axes", lambda x: ops.rot90(x, k=3, axes=(2, 0)), lambda x: np.rot90(x, k=3, axes=(2, 0)), a)
c("diagflat k", lambda x: ops.diagflat(x, k=-1), lambda x: np.diagflat(x, k=-1), m[:2])
c("hamming", lambda x: ops.hamming(5), lambda x: np.hamming(5), a)
c("hanning", lambda x: ops.hanning(1), lambda x: np.hanning(1), a)
c("blackman", lambda x: ops.blackman(1), lambda x: np.blackman(1), a)
c("bartlett", lambda x: ops.bartlett(5), lambda x: np.bartlett(5), a)
c("hamming1", lambda x: ops.hamming(1), lambda x: np.hamming(1), a)
c("kaiser1", lambda x: ops.kaiser(1, 2.), lambda x: np.kaiser(1, 2.), a)
c("bartlett1", lambda x: ops.bartlett(1), lambda x: np.bartlett(1), a)
c("empty mean", lambda x: ops.sum(x, axis=0), lambda x: np.sum(x, axis=0), np.zeros((0, 3), "float32"))
c("empty max kd", lambda x: ops.max(x, axis=1), lambda x: np.max(x, axis=1), np.zeros((2, 0, 3), "float32").reshape(2, 0, 3).transpose(0, 2, 1)[:, :, :0] if False else np.zeros((0, 3), "float32"))
c("argmax empty-ish", lambda x: ops.argmax(x, axis=-1), lambda x: np.argmax(x, axis=-1), np.array([[1., np.nan, 3.]], "float32"))
c("inner", lambda x: ops.inner(x, x), lambda x: np.inner(x, x), a)
c("vdot", lambda x: ops.vdot(x, x), lambda x: np.vdot(x, x), m)
c("outer", lambda x: ops.outer(x, x[:2]), lambda x: np.outer(x, x[:2]), m)
c("dot nd", lambda x: ops.dot(x, m[:5, :2] if False else np.ones((5, 2), "float32")), lambda x: np.dot(x, np.ones((5, 2), "float32")), a)

from keras import KerasTensor
nfail = 0
for name, kf, nf, args in C:
    try:
        exp = np.asarray(nf(*[np.asarray(t) for t in args]))
    except Exception as e:
        print(f"SKIP {name}: numpy err {e}"); continue
    try:
        out = kf(*args)
        got = np.asarray(ops.convert_to_numpy(out))
    except Exception as e:
        nfail += 1
        print(f"ERR  {name}: {type(e).__name__}: {str(e)[:200]}"); continue
    msg = []
    if got.shape != exp.shape:
        msg.append(f"shape {got.shape} vs np {exp.shape}")
    elif not np.allclose(got.astype("float64"), exp.astype("float64"), rtol=1e-3, atol=1e-4, equal_nan=True):
        msg.append(f"value {got.ravel()[:6]} vs np {exp.ravel()[:6]}")
    try:
        sargs = [KerasTensor(np.shape(t), dtype=str(np.asarray(t).dtype)) for t in args]
        sout = kf(*sargs)
        if hasattr(sout, "shape") and tuple(sout.shape) != tuple(got.shape) and not any(d is None for d in sout.shape):
            msg.append(f"symbolic shape {sout.shape} vs eager {got.shape}")
        if hasattr(sout, "dtype") and hasattr(out, "dtype") and keras.backend.standardize_dtype(sout.dtype) != keras.backend.standardize_dtype(out.dtype):
            msg.append(f"symbolic dtype {sout.dtype} vs eager {keras.backend.standardize_dtype(out.dtype)}")
    except Exception as e:
        msg.append(f"symbolic err {type(e).__name__}: {str(e)[:150]}")
    if msg:
        nfail += 1
        print(f"FAIL {name}: " + "; ".join(msg))
print("cases", len(C), "issues", nfail)
