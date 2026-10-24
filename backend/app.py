# backend/app.py
"""GeneSpectra API – serves the hybrid 2D-CNN + phyloP model described in the DABCon 2026 paper.

The request/response format is unchanged from the original prototype (POST /predict with
{"dna_sequence": ...}), so the existing frontend works without modification. The model scores a
specific variant, so the text box now takes a GRCh38 variant: chrom:pos:ref:alt (e.g. 17:7675088:C:T,
chr17-7675088-C-T or "17 7675088 C T"). The backend fetches the +/-50 bp reference window and the
phyloP100way track for that position from the UCSC Genome Browser API, builds exactly the input used in
training (planB/), and averages the three trained seed models.
"""
import io
import json
import os
import re
import base64
import ssl
import urllib.request

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- Constants & Initialization ---
FLANK = 50
WINDOW = 2 * FLANK + 1
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, '..', 'model', 'hybrid')
UCSC = 'https://api.genome.ucsc.edu/getData'
try:  # some Python installs (e.g. Windows Store, macOS python.org) lack a CA bundle; certifi supplies one
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()
app = Flask(__name__)
CORS(app)

# --- Model Loading (three seeds, averaged as in the paper) ---
models = []
for name in sorted(os.listdir(MODEL_DIR)) if os.path.isdir(MODEL_DIR) else []:
    if name.endswith('.keras'):
        models.append(keras.models.load_model(os.path.join(MODEL_DIR, name)))
        print(f'Model loaded: {name}')
METRICS = json.load(open(os.path.join(MODEL_DIR, 'metrics.json'))) if os.path.isdir(MODEL_DIR) else {}
grad_model = None
if models:
    last_conv = [l for l in models[0].layers if isinstance(l, keras.layers.Conv2D)][-1]
    grad_model = keras.Model(models[0].inputs, [last_conv.output, models[0].output])
    print(f"Using layer '{last_conv.name}' of the first seed model for Grad-CAM.")


# --- Variant parsing and data retrieval ---
VARIANT_RE = re.compile(r'^\s*(?:CHR)?([0-9]{1,2}|X|Y)[\s:_-]+([0-9]+)[\s:_-]+([ACGT]+)[\s:_>/-]+([ACGT]+)\s*$', re.I)


def parse_variant(text):
    m = VARIANT_RE.match(text.replace(',', ''))
    if not m:
        return None
    chrom, pos, ref, alt = m.group(1).upper(), int(m.group(2)), m.group(3).upper(), m.group(4).upper()
    if abs(len(ref) - len(alt)) > 10 or max(len(ref), len(alt)) > 11 or ref == alt:
        raise ValueError('Only SNVs and small indels (up to 10 bp) are supported, matching the training data.')
    return chrom, pos, ref, alt


def ucsc(endpoint, **params):
    url = f'{UCSC}/{endpoint}?' + ';'.join(f'{k}={v}' for k, v in params.items())
    with urllib.request.urlopen(urllib.request.Request(url, headers={'Accept': 'application/json'}), timeout=30,
                                context=SSL_CTX) as r:
        return json.load(r)


def fetch_inputs(chrom, pos, ref, alt):
    """Reference/alternate windows and phyloP track, built exactly as in planB/prep_data.py."""
    p0 = pos - 1
    seq = ucsc('sequence', genome='hg38', chrom=f'chr{chrom}', start=p0 - FLANK,
               end=p0 + len(ref) + WINDOW)['dna'].upper()
    left, refseq = seq[:FLANK], seq[FLANK:FLANK + len(ref)]
    if refseq != ref:
        raise ValueError(f'Reference allele mismatch: GRCh38 has {refseq} at chr{chrom}:{pos}, not {ref}.')
    ref_win = (left + seq[FLANK:FLANK + WINDOW])[:WINDOW]
    alt_win = (left + alt + seq[FLANK + len(ref):])[:WINDOW]
    phylop = np.zeros(WINDOW, dtype=np.float32)
    track = ucsc('track', genome='hg38', track='phyloP100way', chrom=f'chr{chrom}', start=p0 - FLANK,
                 end=p0 + FLANK + 1).get('phyloP100way', [])
    for item in track:
        for g in range(max(item['start'], p0 - FLANK), min(item['end'], p0 + FLANK + 1)):
            phylop[g - (p0 - FLANK)] = item['value']
    return ref_win, alt_win, phylop


def annotate(chrom, pos, ref, alt):
    """Gene and consequence from Ensembl VEP (optional; the prediction does not use it)."""
    body = json.dumps({'variants': [f'{chrom} {pos} . {ref} {alt} . . .']}).encode()
    for _ in range(2):
        try:
            req = urllib.request.Request('https://rest.ensembl.org/vep/homo_sapiens/region?canonical=1', data=body,
                                         headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=20, context=SSL_CTX) as r:
                v = json.load(r)[0]
            canon = [t for t in v.get('transcript_consequences', []) if t.get('canonical')]
            gene = canon[0].get('gene_symbol', 'Unknown') if canon else 'Unknown'
            return gene, v.get('most_severe_consequence', 'unknown').replace('_', ' ')
        except Exception:
            continue
    return 'Unknown', 'unknown'


# --- Encoding ---
BASES = 'ACGT'


def one_hot(seq):
    x = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq):
        if b in BASES:
            x[i, BASES.index(b)] = 1.0
    return x


def build_input(ref_win, alt_win, phylop):
    r, a = one_hot(ref_win), one_hot(alt_win)
    seq = np.stack([r, a, a - r], 0)[None]          # (1, 3, 101, 4): reference, alternate, difference
    return {'seq': seq, 'phylop': phylop[None, :, None]}


def grad_cam(inputs, pathogenic):
    """Grad-CAM on the last Conv2D layer of the sequence branch; returns (3, 101) rows and a (101,) profile."""
    with tf.GradientTape() as tape:
        fmap, pred = grad_model({k: tf.convert_to_tensor(v) for k, v in inputs.items()}, training=False)
        score = pred[:, 0] if pathogenic else 1 - pred[:, 0]
    grads = tape.gradient(score, fmap)
    w = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.einsum('bhwc,bc->bhw', fmap, w))[0].numpy()     # (3, 25)
    rows = np.stack([np.interp(np.linspace(0, cam.shape[1] - 1, WINDOW), np.arange(cam.shape[1]), c) for c in cam])
    rows = rows / (rows.max() + 1e-8)
    return rows, rows.mean(0) / (rows.mean(0).max() + 1e-8)


# --- Plotting (same dark style as the original prototype) ---
def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2, transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def style(ax, title, size=18):
    """Dark theme with fonts large enough to read when the frontend shrinks the image into a card."""
    ax.set_title(title, color='#67e8f9', fontsize=size + 2)
    ax.tick_params(colors='#c3dae8', labelsize=size)
    ax.xaxis.label.set_color('#67e8f9')
    ax.yaxis.label.set_color('#67e8f9')
    ax.xaxis.label.set_size(size)
    ax.yaxis.label.set_size(size)


BASE_COLORS = {'A': '#ff4d4d', 'C': '#4d4dff', 'G': '#4dff4d', 'T': '#ffff4d', 'N': '#808080'}
ROW_LABELS = ['Ref', 'Alt', 'Diff']
TICKS = [0, 25, 50, 75, 100]          # window index -> -50, -25, 0, +25, +50 bp


def base_matrix(ref_win, alt_win):
    """Rows: reference, alternate, and a difference row that marks changed positions."""
    idx = {b: i for i, b in enumerate('ACGTN')}
    diff = ''.join('N' if r == a else a for r, a in zip(ref_win, alt_win))
    return np.array([[idx.get(b, 4) for b in s] for s in (ref_win, alt_win, diff)])


def axes_101(ax):
    """Shared axis layout for the 3 x 101 panels: short row labels, ticks every 25 bp, variant marked."""
    ax.set_yticks(range(3)); ax.set_yticklabels(ROW_LABELS)
    ax.set_xticks(TICKS); ax.set_xticklabels([f'{i - FLANK:+d}' if i != FLANK else '0' for i in TICKS])
    ax.axvline(FLANK, color='white', ls=':', lw=2)
    ax.set_xlabel('Position from variant (bp)')


def plot_input(ref_win, alt_win):
    cmap = ListedColormap([BASE_COLORS[b] for b in 'ACGTN'])
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='#0a1e33')
    ax.imshow(base_matrix(ref_win, alt_win), cmap=cmap, vmin=0, vmax=4, aspect='auto', interpolation='nearest')
    axes_101(ax)
    handles = [plt.Line2D([0], [0], marker='s', color='w', label=b, markerfacecolor=BASE_COLORS[b], markersize=14)
               for b in 'ACGT']
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False,
              labelcolor='#c3dae8', fontsize=16)
    style(ax, 'Model input (101 bp)')
    return plot_to_base64(fig)


def plot_heatmap(rows, overlay=None):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='#0a1e33')
    if overlay is not None:
        cmap = ListedColormap([BASE_COLORS[b] for b in 'ACGTN'])
        ax.imshow(overlay, cmap=cmap, vmin=0, vmax=4, aspect='auto', interpolation='nearest')
        im = ax.imshow(rows, cmap='magma', aspect='auto', alpha=0.75, interpolation='bilinear')
        title = 'DNA + Grad-CAM overlay'
    else:
        im = ax.imshow(rows, cmap='plasma', aspect='auto', interpolation='bilinear')
        title = 'Grad-CAM heatmap'
    cb = plt.colorbar(im, ax=ax, pad=0.03)
    cb.set_label('Activation', color='#c3dae8', fontsize=16)
    cb.ax.tick_params(colors='#c3dae8', labelsize=14)
    axes_101(ax)
    style(ax, title)
    return plot_to_base64(fig)


def plot_test_confusion():
    cm = np.array(METRICS['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#0a1e33')
    ax.imshow(cm, cmap='viridis')
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f'{v:,}', ha='center', va='center', color='white' if v < cm.max() / 2 else 'black', fontsize=22)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Benign', 'Pathogenic'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Benign', 'Pathogenic'])
    ax.set_xlabel('Predicted label'); ax.set_ylabel('True label')
    style(ax, f'Held-out test set\n(n = {cm.sum():,}, unseen genes)')
    return plot_to_base64(fig)


def plot_letters(seq):
    per_row = 34
    rows = (len(seq) + per_row - 1) // per_row
    fig, ax = plt.subplots(figsize=(per_row * 0.35, rows * 0.8 + 0.6), facecolor='#0a1e33')
    for i, b in enumerate(seq):
        ax.text(i % per_row + 0.5, rows - 0.5 - i // per_row, b, color=BASE_COLORS.get(b, 'gray'),
                fontsize=14, ha='center', va='center', fontweight='bold')
    ax.set_xlim(0, per_row); ax.set_ylim(0, rows); ax.axis('off')
    ax.set_title('Letter Encoded Sequence (alternate allele window)', color='#67e8f9')
    return plot_to_base64(fig)


def plot_base_activations(base_acts):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0a1e33')
    labels = [b for b in 'ATGC' if base_acts.get(b)]
    if labels:
        bp = ax.boxplot([base_acts[b] for b in labels], patch_artist=True, labels=labels, whis=[0, 100],
                        medianprops=dict(color='#67e8f9', linewidth=2))
        for patch, b in zip(bp['boxes'], labels):
            patch.set_facecolor(BASE_COLORS[b]); patch.set_alpha(0.7)
    ax.set_ylabel('Activation Level'); ax.set_facecolor('#0a1e33')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    style(ax, 'Base-specific activations')
    return plot_to_base64(fig)


# --- Main Prediction Route ---
@app.route('/predict', methods=['POST'])
def predict_mutation():
    if not models:
        return jsonify({'error': 'Model not loaded (expected model/hybrid/*.keras)'}), 500
    text = (request.get_json() or {}).get('dna_sequence', '')
    if text.startswith('>'):
        text = text.split('\n', 1)[-1]
    try:
        variant = parse_variant(text)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    if variant is None:
        return jsonify({'error': 'GeneSpectra scores a specific variant. Enter a GRCh38 variant as '
                                 'chrom:pos:ref:alt, for example 17:7675088:C:T.'}), 400
    chrom, pos, ref, alt = variant
    try:
        ref_win, alt_win, phylop = fetch_inputs(chrom, pos, ref, alt)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Could not fetch sequence/conservation from UCSC: {e}'}), 502

    inputs = build_input(ref_win, alt_win, phylop)
    probs = [float(m(inputs, training=False).numpy()[0, 0]) for m in models]
    p = float(np.mean(probs))
    pathogenic = p >= 0.5
    label = 'Pathogenic' if pathogenic else 'Benign'
    rows, profile = grad_cam(inputs, pathogenic)

    activation_profile = [{'position': i - FLANK, 'activation': float(v)} for i, v in enumerate(profile)]
    base_acts = {b: [float(profile[i]) for i, x in enumerate(ref_win) if x == b] for b in BASES}
    base_stats = {}
    for b, vals in base_acts.items():
        if vals:
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            base_stats[b] = {'min': float(np.min(vals)), 'q1': float(q1), 'median': float(med), 'q3': float(q3),
                             'max': float(np.max(vals))}

    gene, consequence = annotate(chrom, pos, ref, alt)
    peak = int(np.argmax(profile)) - FLANK
    t = METRICS['test']
    summary = {
        'gene': gene,
        'sequenceLength': f'{WINDOW} bp window around chr{chrom}:{pos} {ref}>{alt} (GRCh38)',
        'keyFindings': [
            f'Pathogenicity probability {p:.3f} (mean of {len(models)} models; individual: '
            f'{", ".join(f"{x:.3f}" for x in probs)}).',
            f'Consequence: {consequence}.',
            f'phyloP conservation at the variant site: {phylop[FLANK]:.2f}.',
            f'Grad-CAM peak at {peak:+d} bp from the variant.',
        ],
        'performanceMetrics': {
            'Accuracy': f'{100 * t["accuracy"]:.1f}%', 'Precision': f'{100 * t["precision"]:.1f}%',
            'Recall': f'{100 * t["recall"]:.1f}%', 'AUC-ROC': f'{100 * t["roc_auc"]:.1f}%',
            'Evaluated on': f'{t["n"]:,} held-out variants from unseen genes',
        },
        'clinicalSignificance': [
            'Research prototype, not validated for clinical use.',
            'Use alongside ACMG/AMP evidence; scores reflect a model trained on ClinVar labels.',
        ],
    }
    visualizations = {
        'panel_a': plot_input(ref_win, alt_win),
        'panel_b': plot_heatmap(rows),
        'panel_c': plot_heatmap(rows, overlay=base_matrix(ref_win, alt_win)),
        'confusion_matrix': plot_test_confusion(),
        'letter_encoded_output': plot_letters(alt_win),
        'letter_encoded_output_text': alt_win,
        'base_specific_activations': plot_base_activations(base_acts),
    }
    return jsonify({'prediction': {'label': label, 'confidence': p if pathogenic else 1 - p,
                                   'probabilities': [1 - p, p]},
                    'visualizations': visualizations,
                    'chartData': {'activationProfile': activation_profile, 'baseActivations': base_stats},
                    'summary': summary})


if __name__ == '__main__':
    app.run(debug=False, port=5000)
