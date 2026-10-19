"""Fair 2D vs 1D comparison following TUNING_PROTOCOL.md.

Phase 1 (select): every candidate trained with seed 0; selection by validation ROC-AUC only.
Phase 2 (final):  best 2D and best 1D retrained with seeds 0-2; each evaluated once on the test set.

Usage: python tune_encodings.py
Outputs: out/tuning_selection.json, out/preds_tuned_<name>_s<seed>.csv, out/tuning_final.json
"""
import json, os, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
BATCH, MAX_EPOCHS, PATIENCE, LR = 128, 40, 5, 1e-3
FILTERS = (32, 64, 128)


def head(x):
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    return layers.Dense(1, activation='sigmoid')(x)


def bn_relu(x):
    return layers.ReLU()(layers.BatchNormalization()(x))


# ---------- input transforms: X is (n, 8, 101) uint8, rows 0-3 ref ACGT, rows 4-7 alt ACGT ----------
def in_8x101x1(X):  return X.astype(np.float32)[..., None]
def in_2x101x4(X):  return np.stack([X[:, :4].transpose(0, 2, 1), X[:, 4:].transpose(0, 2, 1)], 1).astype(np.float32)
def in_3x101x4(X):
    r, a = X[:, :4].transpose(0, 2, 1).astype(np.float32), X[:, 4:].transpose(0, 2, 1).astype(np.float32)
    return np.stack([r, a, a - r], 1)
def in_101x8(X):    return X.transpose(0, 2, 1).astype(np.float32)
def in_101x12(X):
    r, a = X[:, :4].transpose(0, 2, 1).astype(np.float32), X[:, 4:].transpose(0, 2, 1).astype(np.float32)
    return np.concatenate([r, a, a - r], 2)


# ---------- candidate architectures ----------
def m2d_a(shape):
    inp = keras.Input(shape); x = inp
    for f in FILTERS:
        x = layers.MaxPooling2D((2, 2))(bn_relu(layers.Conv2D(f, (3, 3), padding='same', use_bias=False)(x)))
    return keras.Model(inp, head(x))


def m2d_b(shape):
    inp = keras.Input(shape)
    x = bn_relu(layers.Conv2D(FILTERS[0], (8, 5), padding='valid', use_bias=False)(layers.ZeroPadding2D(((0, 0), (2, 2)))(inp)))
    x = layers.MaxPooling2D((1, 2))(x)
    for f in FILTERS[1:]:
        x = layers.MaxPooling2D((1, 2))(bn_relu(layers.Conv2D(f, (1, 3), padding='same', use_bias=False)(x)))
    return keras.Model(inp, head(x))


def m2d_rows(kh):
    def build(shape):
        inp = keras.Input(shape); x = inp
        for f in FILTERS:
            x = layers.MaxPooling2D((1, 2))(bn_relu(layers.Conv2D(f, (kh, 3), padding='same', use_bias=False)(x)))
        return keras.Model(inp, head(x))
    return build


def m1d(kernels):
    def build(shape):
        inp = keras.Input(shape); x = inp
        for f, k in zip(FILTERS, kernels):
            x = layers.MaxPooling1D(2)(bn_relu(layers.Conv1D(f, k, padding='same', use_bias=False)(x)))
        return keras.Model(inp, head(x))
    return build


CANDIDATES = {
    '2D-A': ('2d', in_8x101x1, (8, 101, 1), m2d_a),
    '2D-B': ('2d', in_8x101x1, (8, 101, 1), m2d_b),
    '2D-C': ('2d', in_2x101x4, (2, 101, 4), m2d_rows(2)),
    '2D-D': ('2d', in_3x101x4, (3, 101, 4), m2d_rows(3)),
    '1D-A': ('1d', in_101x8, (101, 8), m1d((3, 3, 3))),
    '1D-B': ('1d', in_101x8, (101, 8), m1d((5, 5, 5))),
    '1D-C': ('1d', in_101x12, (101, 12), m1d((3, 3, 3))),
    '1D-D': ('1d', in_101x8, (101, 8), m1d((7, 3, 3))),
}


def fit(name, seed, X, y, idx):
    kind, tf_in, shape, builder = CANDIDATES[name]
    Xi = tf_in(X)
    keras.utils.set_random_seed(seed)
    m = builder(shape)
    m.compile(optimizer=keras.optimizers.Adam(LR), loss='binary_crossentropy')
    t0 = time.time()
    h = m.fit(Xi[idx['train']], y[idx['train']], validation_data=(Xi[idx['val']], y[idx['val']]),
              batch_size=BATCH, epochs=MAX_EPOCHS, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=PATIENCE, restore_best_weights=True)])
    pv = m.predict(Xi[idx['val']], batch_size=1024, verbose=0)[:, 0]
    return m, Xi, dict(val_auc=round(float(roc_auc_score(y[idx['val']], pv)), 4), params=int(m.count_params()),
                       epochs=len(h.history['loss']), seconds=round(time.time() - t0, 1))


def main():
    tf.config.threading.set_inter_op_parallelism_threads(4)
    d = np.load(os.path.join(OUT, 'dataset.npz'))
    X, y = d['X'], d['y'].astype(np.float32)
    meta = pd.read_csv(os.path.join(OUT, 'variants.csv'))
    idx = {k: np.where(meta.split_gene.values == k)[0] for k in ('train', 'val', 'test')}

    sel_path = os.path.join(OUT, 'tuning_selection.json')
    sel = json.load(open(sel_path)) if os.path.exists(sel_path) else {}
    for name in CANDIDATES:
        if name in sel:
            continue
        _, _, info = fit(name, 0, X, y, idx)
        sel[name] = info
        print('select', name, info, flush=True)
        json.dump(sel, open(sel_path, 'w'), indent=2)

    best = {k: max((n for n in sel if n.startswith(k)), key=lambda n: sel[n]['val_auc']) for k in ('2D', '1D')}
    print('selected by validation AUC:', best, flush=True)

    final = {'selected': best, 'runs': {}}
    for kind, name in best.items():
        for seed in range(3):
            m, Xi, info = fit(name, seed, X, y, idx)
            pt = m.predict(Xi[idx['test']], batch_size=1024, verbose=0)[:, 0]
            pd.DataFrame({'id': meta.id.values[idx['test']], 'label': y[idx['test']].astype(int), 'prob': pt}) \
                .to_csv(os.path.join(OUT, f'preds_tuned_{name}_s{seed}.csv'), index=False)
            info['test_auc'] = round(float(roc_auc_score(y[idx['test']], pt)), 4)
            final['runs'][f'{name}_s{seed}'] = info
            print('final', name, seed, info, flush=True)
            json.dump(final, open(os.path.join(OUT, 'tuning_final.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
