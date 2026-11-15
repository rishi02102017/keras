import numpy as np, keras
from keras import ops
print("keras", keras.__version__, keras.backend.backend())
def valid(x, idx, kth, axis=-1):
    idx = np.asarray(idx)
    s = np.sort(x, axis=axis)
    ok_perm = np.array_equal(np.sort(idx, axis=axis), np.broadcast_to(np.arange(x.shape[axis]).reshape([-1 if i == (axis % x.ndim) else 1 for i in range(x.ndim)]), x.shape))
    return ok_perm and np.array_equal(np.take_along_axis(x, idx, axis=axis).take(kth, axis=axis), s.take(kth, axis=axis)) and np.all(np.take_along_axis(x, idx, axis=axis).take(range(kth), axis=axis) <= np.expand_dims(s.take(kth, axis=axis), axis))
cases = [
    ("float 1d", np.array([0.7, 0.2, 0.5, 0.9], "float32"), 1, -1),
    ("neg int 1d", np.array([-3, -1, -2], "int32"), 0, -1),
    ("int 2d", np.array([[3, 4, 2, 1], [9, 8, 7, 6], [5, 0, 1, 2]], "int32"), 1, -1),
    ("int 3d", np.arange(60).reshape(3, 4, 5)[..., ::-1].astype("int32"), 2, -1),
    ("float 2d axis0", np.array([[0.3, 0.1], [0.2, 0.4], [0.1, 0.3]], "float32"), 0, 0),
]
for name, x, kth, axis in cases:
    try:
        r = ops.convert_to_numpy(ops.argpartition(x, kth, axis=axis))
        print(f"{name}: got {r.tolist()} valid={valid(x, r, kth, axis)}  np={np.argpartition(x, kth, axis=axis).tolist()}")
    except Exception as e:
        print(f"{name}: ERR {type(e).__name__}: {str(e)[:150]}")
