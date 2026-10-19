"""Plan B, step 2: train the 2D CNN and the 1D ablation, save test-set predictions.

Experiments (each repeated for every seed):
  gene_2d    - 2D CNN, gene-disjoint split          (primary result)
  gene_1d    - parameter-matched 1D CNN, same split  (encoding ablation)
  random_2d  - 2D CNN, random variant-level split    (leakage comparison)

Usage: python train_models.py [n_seeds] [experiments...]
Outputs (out/): preds_<exp>_s<seed>.csv, history_<exp>_s<seed>.json, model_<exp>_s<seed>.keras,
                gradcam_gene_2d_s0.npz, train_summary.json
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
EXPS = sys.argv[2:] or ['gene_2d', 'gene_1d', 'random_2d']
BATCH, MAX_EPOCHS, PATIENCE, LR = 128, 40, 5, 1e-3


def block2d(x, f):
    x = layers.Conv2D(f, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return layers.MaxPooling2D((2, 2))(x)


def block1d(x, f):
    x = layers.Conv1D(f, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return layers.MaxPooling1D(2)(x)


def head(x):
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    return layers.Dense(1, activation='sigmoid')(x)


def build(kind):
    if kind == '2dD':         # tuned design 2D-D selected by TUNING_PROTOCOL.md (validation AUC only)
        from tune_encodings import m2d_rows
        m = m2d_rows(3)((3, 101, 4))
        m.compile(optimizer=keras.optimizers.Adam(LR), loss='binary_crossentropy',
                  metrics=[keras.metrics.AUC(name='auc')])
        return m
    if kind == '2d':          # 8 x 101 "image": 4 reference + 4 alternate one-hot rows
        inp = keras.Input((8, 101, 1))
        x = inp
        for i, f in enumerate((32, 64, 128)):
            x = block2d(x, f)
    else:                     # same information as a 1D sequence with 8 channels
        inp = keras.Input((101, 8))
        x = inp
        for f in (32, 64, 128):
            x = block1d(x, f)
    m = keras.Model(inp, head(x))
    m.compile(optimizer=keras.optimizers.Adam(LR), loss='binary_crossentropy',
              metrics=[keras.metrics.AUC(name='auc')])
    return m


def to_input(X, kind):
    if kind == '2dD':
        from tune_encodings import in_3x101x4
        return in_3x101x4(X)
    X = X.astype(np.float32)
    return X[..., None] if kind == '2d' else np.transpose(X, (0, 2, 1))


def gradcam_profile(model, X, layer_name):
    """Grad-CAM heatmap per example, reduced to a 1D importance profile over the 101 positions."""
    conv = model.get_layer(layer_name)
    gm = keras.Model(model.inputs, [conv.output, model.output])
    profs = []
    for i in range(0, len(X), 512):
        xb = tf.convert_to_tensor(X[i:i + 512])
        with tf.GradientTape() as tape:
            fmap, pred = gm(xb, training=False)
            score = pred[:, 0]
        grads = tape.gradient(score, fmap)                       # (b, h, w, c)
        w = tf.reduce_mean(grads, axis=(1, 2))                   # (b, c)
        cam = tf.nn.relu(tf.einsum('bhwc,bc->bhw', fmap, w))     # (b, h, w)
        cam = tf.reduce_mean(cam, axis=1).numpy()                # collapse rows -> (b, w)
        cam = np.stack([np.interp(np.linspace(0, cam.shape[1] - 1, 101), np.arange(cam.shape[1]), c) for c in cam])
        cam /= cam.max(axis=1, keepdims=True) + 1e-8
        profs.append(cam)
    return np.concatenate(profs)


def main():
    tf.config.threading.set_inter_op_parallelism_threads(4)
    d = np.load(os.path.join(OUT, 'dataset.npz'))
    X, y = d['X'], d['y'].astype(np.float32)
    meta = pd.read_csv(os.path.join(OUT, 'variants.csv'))
    summ_path = os.path.join(OUT, 'train_summary.json')
    summary = json.load(open(summ_path)) if os.path.exists(summ_path) else {}
    for exp in EXPS:
        split_col = 'split_gene' if exp.startswith('gene') else 'split_random'
        kind = exp.split('_')[1]
        idx = {k: np.where(meta[split_col].values == k)[0] for k in ('train', 'val', 'test')}
        Xi = to_input(X, kind)
        for seed in range(N_SEEDS):
            tag = f'{exp}_s{seed}'
            keras.utils.set_random_seed(seed)
            m = build(kind)
            t0 = time.time()
            h = m.fit(Xi[idx['train']], y[idx['train']], validation_data=(Xi[idx['val']], y[idx['val']]),
                      batch_size=BATCH, epochs=MAX_EPOCHS, verbose=2,
                      callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=PATIENCE,
                                                               restore_best_weights=True)])
            secs = time.time() - t0
            t1 = time.time()
            p = m.predict(Xi[idx['test']], batch_size=1024, verbose=0)[:, 0]
            ms_per_variant = 1000 * (time.time() - t1) / len(idx['test'])
            pd.DataFrame({'id': meta.id.values[idx['test']], 'label': y[idx['test']].astype(int), 'prob': p}) \
                .to_csv(os.path.join(OUT, f'preds_{tag}.csv'), index=False)
            json.dump({k: [float(v) for v in vs] for k, vs in h.history.items()},
                      open(os.path.join(OUT, f'history_{tag}.json'), 'w'))
            m.save(os.path.join(OUT, f'model_{tag}.keras'))
            summary[tag] = dict(params=int(m.count_params()),
                                trainable_params=int(sum(np.prod(w.shape) for w in m.trainable_weights)),
                                epochs_run=len(h.history['loss']),
                                best_epoch=int(np.argmin(h.history['val_loss'])) + 1,
                                train_seconds=round(secs, 1), cpu_ms_per_variant=round(ms_per_variant, 3),
                                n_train=len(idx['train']), n_val=len(idx['val']), n_test=len(idx['test']))
            print(tag, summary[tag], flush=True)
            if tag in ('gene_2d_s0', 'gene_2dD_s0'):
                last_conv = [l.name for l in m.layers if isinstance(l, layers.Conv2D)][-1]
                prof = gradcam_profile(m, Xi[idx['test']], last_conv)
                np.savez_compressed(os.path.join(OUT, f'gradcam_{tag}.npz'), profile=prof,
                                    label=y[idx['test']], prob=p, layer=last_conv)
            json.dump(summary, open(summ_path, 'w'), indent=2)
    summary['config'] = dict(batch=BATCH, max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR, optimizer='Adam',
                             loss='binary_crossentropy', tensorflow=tf.__version__, n_seeds=N_SEEDS)
    json.dump(summary, open(summ_path, 'w'), indent=2)


if __name__ == '__main__':
    main()
