import numpy as np, keras, traceback
from keras import ops, layers
print("keras", keras.__version__, keras.backend.backend())
def t(name, f):
    try:
        print(f"[{name}]", f())
    except Exception as e:
        print(f"[{name}] ERR {type(e).__name__}: {str(e)[:300]}")
# utils
t("to_categorical tensor None", lambda: ops.convert_to_numpy(keras.utils.to_categorical(ops.convert_to_tensor([0, 2, 1]))).tolist())
t("to_categorical neg", lambda: keras.utils.to_categorical([0, 1, -1], 3).tolist())
t("pad_sequences maxlen0", lambda: keras.utils.pad_sequences([[1, 2], [3]], maxlen=0).shape)
t("pad_sequences empty", lambda: keras.utils.pad_sequences([]).shape)
t("pad_sequences value", lambda: keras.utils.pad_sequences([[1, 2], [3]], value=-1, padding="post").tolist())
def ts():
    d = keras.utils.timeseries_dataset_from_array(np.arange(20), None, sequence_length=4, sequence_stride=3, sampling_rate=2, start_index=2, end_index=17, batch_size=100)
    return [x.numpy().tolist() for x in d]
t("timeseries", ts)
def ts2():
    d = keras.utils.timeseries_dataset_from_array(np.arange(10), np.arange(10), sequence_length=3, batch_size=2, shuffle=True, seed=1)
    return [(x.numpy().tolist(), y.numpy().tolist()) for x, y in d]
t("timeseries shuffle", ts2)
t("normalize np axis None", lambda: keras.utils.normalize(np.array([[3., 4.], [0., 0.]]), axis=None).tolist())
t("normalize tensor axis None", lambda: ops.convert_to_numpy(keras.utils.normalize(ops.convert_to_tensor(np.array([[3., 4.], [0., 0.]])), axis=None)).tolist())
t("normalize np order1", lambda: keras.utils.normalize(np.array([[3., 4.], [0., 0.]]), order=1).tolist())
t("normalize tensor order1", lambda: ops.convert_to_numpy(keras.utils.normalize(ops.convert_to_tensor(np.array([[3., 4.], [0., 0.]])), order=1)).tolist())
# layers
def rt(layer, x):
    y = ops.convert_to_numpy(layer(x))
    l2 = layer.__class__.from_config(layer.get_config())
    try:
        y2 = ops.convert_to_numpy(l2(x))
        same = y2.shape == y.shape and np.allclose(y2, y)
    except Exception as e:
        same = f"ERR {e}"
    cos = layer.compute_output_shape(tuple(np.shape(x)))
    return f"out {y.shape} cos {cos} roundtrip_same {same} vals {y.ravel()[:8].tolist()}"
x = np.array([[0.1, 1.5, 2.7, -3.0]], "float32")
t("Discretization bins", lambda: rt(layers.Discretization(bin_boundaries=[0., 1., 2.]), x))
t("Discretization one_hot", lambda: rt(layers.Discretization(bin_boundaries=[0., 1., 2.], output_mode="one_hot"), x[0]))
t("Discretization multi_hot", lambda: rt(layers.Discretization(bin_boundaries=[0., 1., 2.], output_mode="multi_hot"), x))
t("Discretization count", lambda: rt(layers.Discretization(bin_boundaries=[0., 1., 2.], output_mode="count"), x))
def disc_adapt():
    l = layers.Discretization(num_bins=4, epsilon=0.01)
    l.adapt(np.arange(100, dtype="float32"))
    return rt(l, np.array([[5., 30., 60., 99.]], "float32")) + f" bb={l.bin_boundaries}"
t("Discretization adapt", disc_adapt)
def norm():
    l = layers.Normalization(mean=[1., 2.], variance=[4., 9.], invert=True)
    return rt(l, np.array([[1., 1.]], "float32"))
t("Normalization invert", norm)
def norm2():
    l = layers.Normalization(axis=None)
    l.adapt(np.array([[1., 2.], [3., 4.]], "float32"))
    return rt(l, np.array([[1., 4.]], "float32"))
t("Normalization axis None", norm2)
def norm3():
    l = layers.Normalization(axis=(1, 2))
    l.adapt(np.arange(24, dtype="float32").reshape(2, 3, 4))
    return rt(l, np.arange(12, dtype="float32").reshape(1, 3, 4))
t("Normalization axis tuple", norm3)
t("IntegerLookup", lambda: rt(layers.IntegerLookup(vocabulary=[12, 36, 1138], mask_token=0, num_oov_indices=2), np.array([[12, 0, 5, 1138]])))
t("IntegerLookup invert", lambda: rt(layers.IntegerLookup(vocabulary=[12, 36, 1138], invert=True), np.array([[1, 3, 0]])))
t("IntegerLookup multi_hot", lambda: rt(layers.IntegerLookup(vocabulary=[12, 36, 1138], output_mode="multi_hot"), np.array([[12, 36, 7]])))
t("IntegerLookup tf_idf", lambda: rt(layers.IntegerLookup(vocabulary=[12, 36, 1138], idf_weights=[.1, .2, .3], output_mode="tf_idf"), np.array([[12, 36, 36]])))
t("IntegerLookup count pad_to", lambda: rt(layers.IntegerLookup(vocabulary=[12, 36, 1138], output_mode="count", pad_to_max_tokens=True, max_tokens=8), np.array([[12, 36, 36]])))
t("StringLookup", lambda: rt(layers.StringLookup(vocabulary=["a", "b", "c"], mask_token="", oov_token="[X]"), np.array([["a", "", "z"]])))
t("StringLookup one_hot", lambda: rt(layers.StringLookup(vocabulary=["a", "b", "c"], output_mode="one_hot"), np.array(["a", "z"])))
t("CategoryEncoding count w", lambda: ops.convert_to_numpy(layers.CategoryEncoding(num_tokens=4, output_mode="count")(np.array([[0, 1, 1]]), count_weights=np.array([[.5, 1., 2.]], "float32"))).tolist())
t("CategoryEncoding cos", lambda: rt(layers.CategoryEncoding(num_tokens=4, output_mode="one_hot"), np.array([[0], [2]])))
t("Hashing salt", lambda: rt(layers.Hashing(num_bins=5, salt=[1, 2]), np.array([["a"], ["b"]])))
t("Hashing mask", lambda: rt(layers.Hashing(num_bins=5, mask_value="", output_mode="one_hot"), np.array(["a", ""])))
t("HashedCrossing", lambda: ops.convert_to_numpy(layers.HashedCrossing(num_bins=5)((np.array(["a", "b"]), np.array([1, 2])))).tolist())
t("TextVectorization ngrams", lambda: rt(layers.TextVectorization(vocabulary=["a", "b", "a b"], ngrams=2, output_sequence_length=6), np.array(["a b c"])))
t("TextVectorization split char", lambda: rt(layers.TextVectorization(vocabulary=["a", "b"], split="character", output_sequence_length=4), np.array(["ab ba"])))
t("Rescaling", lambda: rt(layers.Rescaling(scale=[1., 2.], offset=[0., 1.]), np.ones((1, 2), "float32")))
img = np.random.default_rng(0).uniform(0, 255, (2, 8, 8, 3)).astype("float32")
for nm, kw in [("RandomBrightness", dict(factor=0.5, value_range=(0, 255), seed=1)), ("RandomContrast", dict(factor=0.5, value_range=(0, 255), seed=1)), ("RandomSaturation", dict(factor=0.5, value_range=(0, 255), seed=1)), ("RandomHue", dict(factor=0.5, value_range=(0, 255), seed=1)), ("RandomSharpness", dict(factor=0.5, value_range=(0, 255), seed=1)), ("RandomColorJitter", dict(value_range=(0, 255), brightness_factor=0.3, contrast_factor=0.3, seed=1)), ("Solarization", dict(addition_factor=0.5, threshold_factor=0.5, value_range=(0, 255), seed=1)), ("RandomPosterization", dict(factor=(1, 4), value_range=(0, 255), seed=1)), ("RandomGrayscale", dict(factor=1.0, seed=1)), ("RandomInvert", dict(factor=1.0, value_range=(0, 255), seed=1)), ("AutoContrast", dict(value_range=(0, 255))), ("Equalization", dict(value_range=(0, 255))), ("RandomGaussianBlur", dict(factor=1.0, value_range=(0, 255), seed=1)), ("RandomColorDegeneration", dict(factor=0.5, value_range=(0, 255), seed=1))]:
    def f(nm=nm, kw=kw):
        L = getattr(layers, nm)
        l = L(**kw)
        y = ops.convert_to_numpy(l(img, training=True))
        l2 = L.from_config(l.get_config())
        y2 = ops.convert_to_numpy(l2(img, training=True))
        diff = l.get_config().keys() ^ l2.get_config().keys()
        cfgdiff = {k: (l.get_config()[k], l2.get_config()[k]) for k in l.get_config() if l.get_config()[k] != l2.get_config()[k]}
        return f"min {y.min():.2f} max {y.max():.2f} shape {y.shape} cos {l.compute_output_shape(img.shape)} cfgdiff {cfgdiff} seeded_same {np.allclose(y, y2)}"
    t(nm, f)
# losses
yt = np.array([0, 1, 2, 1]); yp = np.random.default_rng(1).standard_normal((4, 3)).astype("float32")
t("SCCE ignore_class", lambda: float(keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=1)(yt, yp)))
t("SCCE ignore_class sw", lambda: float(keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=1)(yt, yp, sample_weight=np.array([1., 2., 3., 4.]))))
t("SCCE ignore_class ref", lambda: float(keras.losses.SparseCategoricalCrossentropy(from_logits=True)(yt[[0, 2]], yp[[0, 2]], sample_weight=np.array([1., 3.]))))
t("SCCE ignore_class none red", lambda: ops.convert_to_numpy(keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=1, reduction=None)(yt, yp)).tolist())
t("CFocal label_smoothing", lambda: float(keras.losses.CategoricalFocalCrossentropy(label_smoothing=0.1, from_logits=True)(np.eye(3)[yt], yp)))
t("Dice axis", lambda: ops.convert_to_numpy(keras.losses.Dice(axis=(1,), reduction=None)(np.eye(3)[yt].astype("float32"), np.abs(yp))).tolist())
t("Tversky", lambda: float(keras.losses.Tversky(alpha=0.3, beta=0.7)(np.eye(3)[yt].astype("float32"), np.abs(yp))))
t("Huber delta", lambda: float(keras.losses.Huber(delta=0.5)(yp, yp * 2)))
t("CosineSimilarity axis0", lambda: ops.convert_to_numpy(keras.losses.CosineSimilarity(axis=0, reduction=None)(yp, yp * 2)).tolist())
# metrics
def miou():
    m = keras.metrics.MeanIoU(num_classes=3, ignore_class=1, sparse_y_pred=False)
    m.update_state(yt, yp); return float(m.result())
t("MeanIoU ignore", miou)
def f1():
    m = keras.metrics.F1Score(average="macro", threshold=0.5)
    m.update_state(np.eye(3)[yt].astype("float32"), np.abs(yp) / 3); return float(m.result())
t("F1", f1)
def auc():
    m = keras.metrics.AUC(from_logits=True, multi_label=True, num_labels=3)
    m.update_state(np.eye(3)[yt].astype("float32"), yp); return float(m.result())
t("AUC logits", auc)
def r2():
    m = keras.metrics.R2Score(class_aggregation="variance_weighted_average", num_regressors=1)
    m.update_state(yp, yp + 0.1); return float(m.result())
t("R2", r2)
def prec():
    m = keras.metrics.Precision(top_k=2, class_id=1)
    m.update_state(np.eye(3)[yt].astype("float32"), np.abs(yp)); return float(m.result())
t("Precision topk", prec)
def stopk():
    m = keras.metrics.SparseTopKCategoricalAccuracy(k=2, from_sorted_ids=False)
    m.update_state(yt, yp); return float(m.result())
t("SparseTopK", stopk)
