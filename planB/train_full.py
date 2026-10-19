"""Round 2: train 2D-D on the enlarged training set (R2-1) and 2D-D + phyloP hybrid (R2-3).

Usage: python train_full.py <exp> [n_seeds]     exp in {full_2dD, full_2dD_phylop}
Outputs (out/full/): preds_<exp>_s<seed>_{val,test}.csv, history_<exp>_s<seed>.json, model_<exp>_s<seed>.keras,
                     train_full_summary.json
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score
from tune_encodings import head, bn_relu, FILTERS

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, 'out', 'full')
EXP = sys.argv[1]
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
BATCH, MAX_EPOCHS, PATIENCE, LR = 128, 40, 5, 1e-3
USE_PHYLOP = EXP.endswith('phylop')


def trunk_2dD(inp):
    x = inp
    for f in FILTERS:
        x = layers.MaxPooling2D((1, 2))(bn_relu(layers.Conv2D(f, (3, 3), padding='same', use_bias=False)(x)))
    return layers.Flatten()(x)


def build():
    seq_in = keras.Input((3, 101, 4), name='seq')
    feats = [trunk_2dD(seq_in)]
    inputs = [seq_in]
    if USE_PHYLOP:
        ph_in = keras.Input((101, 1), name='phylop')
        y = ph_in
        for f in FILTERS:
            y = layers.MaxPooling1D(2)(bn_relu(layers.Conv1D(f, 3, padding='same', use_bias=False)(y)))
        feats.append(layers.Flatten()(y))
        inputs.append(ph_in)
    x = layers.Concatenate()(feats) if len(feats) > 1 else feats[0]
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    m = keras.Model(inputs, layers.Dense(1, activation='sigmoid')(x))
    m.compile(optimizer=keras.optimizers.Adam(LR), loss='binary_crossentropy')
    return m


def seq_transform(xb):
    """uint8 (b, 8, 101) -> float (b, 3, 101, 4): reference, alternate, alternate - reference."""
    x = tf.cast(xb, tf.float32)
    r = tf.transpose(x[:, :4], (0, 2, 1))
    a = tf.transpose(x[:, 4:], (0, 2, 1))
    return tf.stack([r, a, a - r], axis=1)


def dataset(X, P, y, idx, shuffle, seed):
    ds = tf.data.Dataset.from_tensor_slices((idx,))
    if shuffle:
        ds = ds.shuffle(len(idx), seed=seed, reshuffle_each_iteration=True)
    def gather(i):
        xb = tf.gather(X, i)
        feats = {'seq': seq_transform(xb)}
        if USE_PHYLOP:
            feats['phylop'] = tf.expand_dims(tf.gather(P, i), -1)
        return feats, tf.gather(y, i)
    return ds.batch(BATCH).map(lambda i: gather(i), num_parallel_calls=tf.data.AUTOTUNE).prefetch(4)


def main():
    tf.config.threading.set_inter_op_parallelism_threads(4)
    d = np.load(os.path.join(FULL, 'dataset_full.npz'))
    X = tf.constant(d['X'])
    y_np = d['y'].astype(np.float32)
    y = tf.constant(y_np)
    P = tf.constant(np.load(os.path.join(FULL, 'phylop_full.npy'))) if USE_PHYLOP else tf.zeros((1, 101))
    meta = pd.read_csv(os.path.join(FULL, 'variants_full.csv'))
    idx = {k: np.where(meta.split.values == k)[0] for k in ('train', 'val', 'test')}
    summ_path = os.path.join(FULL, 'train_full_summary.json')
    summary = json.load(open(summ_path)) if os.path.exists(summ_path) else {}
    for seed in range(N_SEEDS):
        tag = f'{EXP}_s{seed}'
        if tag in summary:
            continue
        keras.utils.set_random_seed(seed)
        m = build()
        t0 = time.time()
        h = m.fit(dataset(X, P, y, idx['train'], True, seed), validation_data=dataset(X, P, y, idx['val'], False, 0),
                  epochs=MAX_EPOCHS, verbose=2,
                  callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=PATIENCE, restore_best_weights=True)])
        secs = time.time() - t0
        res = {}
        for split in ('val', 'test'):
            p = m.predict(dataset(X, P, y, idx[split], False, 0), verbose=0)[:, 0]
            pd.DataFrame({'id': meta.id.values[idx[split]], 'label': y_np[idx[split]].astype(int), 'prob': p}) \
                .to_csv(os.path.join(FULL, f'preds_{tag}_{split}.csv'), index=False)
            res[f'{split}_auc'] = round(float(roc_auc_score(y_np[idx[split]], p)), 4)
        json.dump({k: [float(v) for v in vs] for k, vs in h.history.items()},
                  open(os.path.join(FULL, f'history_{tag}.json'), 'w'))
        m.save(os.path.join(FULL, f'model_{tag}.keras'))
        summary[tag] = dict(params=int(m.count_params()), epochs_run=len(h.history['loss']),
                            best_epoch=int(np.argmin(h.history['val_loss'])) + 1, train_seconds=round(secs, 1),
                            n_train=len(idx['train']), **res)
        print(tag, summary[tag], flush=True)
        json.dump(summary, open(summ_path, 'w'), indent=2)


if __name__ == '__main__':
    main()
