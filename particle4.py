"""
Global ML pipeline for predicting Beta_slope and Biovolume.

Changes from previous version:
  - Two separate single-output SVGPs (one per target).
  - Beta:      500 inducing points, RBF kernel.
  - Biovolume: 1000 inducing points, Matern kernel (nu=1.5).
  - Feature importance REWRITTEN (sections 11/11b/11c):
      * SHAP — directional, per-feature (TreeSHAP for RF, GradientSHAP for SVGP)
      * Per-variable permutation importance (single-feature shuffle, R² drop)
      * Grouped permutation on correlation clusters (collinearity-robust cross-check)
      * Combined summary bar plot: SHAP mean |value| vs permutation R² drop
      * Old impurity (MDI) and marginal r-drop permutation removed.
  - NESTED CV now fully symmetric across RF, SVGP and RIDGE:
      * Outer loop: K_FOLDS buffered spatial CV — the SAME fold_indices object
        is passed to cv_rf(), cv_svgp_two() and cv_ridge().
      * Inner selection runs INSIDE EVERY outer fold (TUNE_EVERY_FOLD), using
        only that fold's training partition, so every outer score is a genuine
        nested-CV score. The configuration reported for the paper and used for
        the final all-data models is the MODE across the outer folds (ties ->
        earliest fold). Per-fold selections are printed and written to the .mat
        files, so their stability can be reported.
        All three tuners call the shared helper _inner_folds(), so the
        inner train/validation index sets are byte-identical:
          the outer-training blocks are split into INNER_K_FOLDS=5 spatial
          groups and the first INNER_N_USE=1 is held out: 20% of blocks,
          BLOCKED and BUFFERED (BLOCK_DEG/BUFFER_KM, random_state=
          INNER_SEED, different from the outer random_state=0).
          Set INNER_N_USE=INNER_K_FOLDS for a full 5-fold inner CV instead —
          no tuner needs changing, they all loop over whatever is returned.
      * Inner splits are always built from the FULL outer-training lat/lon; the
        INNER_TRAIN_SAMPLE row cap is applied INSIDE the inner training split,
        never before the folds are built, and is now the SAME for all three
        models — so candidates are selected on an equal number of rows and
        capacity-type hyperparameters are not biased by unequal information.
      * X and y scalers are refit on each inner training split for all three
        models (RF via Pipeline + TransformedTargetRegressor), so no inner
        validation block leaks into the standardisation.
      * Selection criterion is the same for all three: mean MSE in LOG-TARGET
        units across the inner folds.
      * Search spaces:
          RF    — grid over n_estimators, max_depth, min_samples_split,
                  min_samples_leaf, max_features (joint 2-output fit).
          SVGP  — grid over SVGP_PARAM_GRID, per target, GP_TUNE_EPOCHS epochs.
          Ridge — grid over RIDGE_ALPHA_GRID (9 alphas), per target.
        SVGP now varies the kernel as well as the inducing-point count, so its
        search is no longer an order of magnitude smaller than the others'.
      * Remaining known asymmetry: the RF is fitted jointly on both targets and
        therefore selects ONE hyperparameter set for beta+biovolume, whereas the
        SVGP and Ridge select per target.
"""

import os
import gc
import time
import numpy as np
import pandas as pd
from scipy import io
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist, squareform
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import MiniBatchKMeans

import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader

try:
    import shap
except ImportError as _e:
    raise ImportError("This pipeline uses SHAP for feature importance: "
                      "pip install shap") from _e

# ── Checkpoint / restart (SIGUSR1-based) ──────────────────────────────────────
import signal, pickle, subprocess

_CHECKPOINT_NOW = False
_JOB_START      = time.time()

CKPT_PATH    = '/home/b/b384036/particle_size_article/particle_checkpoint.pkl'
MODEL_DIR    = '/home/b/b384036/particle_size_article/model_checkpoint'
os.makedirs(MODEL_DIR, exist_ok=True)

# Paths for saved model objects
_RF_PATH          = os.path.join(MODEL_DIR, 'rf_final.pkl')
_SVGP_BETA_PATH   = os.path.join(MODEL_DIR, 'svgp_beta_final.pt')
_SVGP_BIOV_PATH   = os.path.join(MODEL_DIR, 'svgp_biovol_final.pt')
_SCALERS_PATH     = os.path.join(MODEL_DIR, 'scalers.pkl')

def _save_rf(model):
    with open(_RF_PATH, 'wb') as f: pickle.dump(model, f, protocol=4)
    print(f"[ckpt] RF model saved → {_RF_PATH}", flush=True)

def _load_rf():
    with open(_RF_PATH, 'rb') as f: return pickle.load(f)

def _save_svgp(wrapper, path):
    torch.save({'model': wrapper.model.state_dict(),
                'likelihood': wrapper.likelihood.state_dict(),
                'inducing': wrapper.model.variational_strategy.inducing_points,
                'kernel': wrapper.model.covar_module.__class__.__name__,
                }, path)
    print(f"[ckpt] SVGP saved → {path}", flush=True)

def _load_svgp(path, n_feat):
    ck   = torch.load(path, map_location=DEVICE)
    ind  = ck['inducing']
    kname= ck['kernel']
    kern = 'matern15' if 'Matern' in kname else 'rbf'
    m    = SingleSVGP(ind, kernel=kern).to(DEVICE)
    lk   = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
    m.load_state_dict(ck['model'])
    lk.load_state_dict(ck['likelihood'])
    return SingleSVGPWrapper(m, lk, DEVICE)

def _save_scalers(sx, syb, syv):
    with open(_SCALERS_PATH, 'wb') as f:
        pickle.dump({'sx': sx, 'sy_b': syb, 'sy_v': syv}, f, protocol=4)
    print(f"[ckpt] Scalers saved → {_SCALERS_PATH}", flush=True)

def _load_scalers():
    with open(_SCALERS_PATH, 'rb') as f: d = pickle.load(f)
    return d['sx'], d['sy_b'], d['sy_v']

def _signal_handler(signum, frame):
    global _CHECKPOINT_NOW
    _CHECKPOINT_NOW = True
    print("\n[ckpt] *** SIGUSR1 received — will checkpoint after current step ***",
          flush=True)

signal.signal(signal.SIGUSR1, _signal_handler)
print("[ckpt] SIGUSR1 handler registered.", flush=True)

def should_checkpoint():
    if _CHECKPOINT_NOW:
        return True
    margin  = 10 * 60   # 10-min fallback safety margin
    limit   = float(os.environ.get('SLURM_TIME_LIMIT_SECONDS', 7*3600 + 55*60))
    return (limit - (time.time() - _JOB_START)) < margin

# Bump this whenever a change makes half-finished results from an EARLIER run
# scientifically incomparable with results from this one — a different inner
# split, a different selection criterion, a different search grid, a new skill
# metric. ckpt_load() then refuses to resume instead of silently mixing folds
# scored under two different protocols, which produces a plausible-looking
# table that is not reproducible by any single run.
#
# 1: original — 3-fold inner CV, tune on fold 0 only, rbf-only SVGP grid,
#    OLS baseline, RMSE in original units, per-model inner row caps.
# 2: single 80/20 blocked inner split; tuning in EVERY outer fold with the
#    modal configuration reported; 12-config SVGP grid (4 inducing x 3
#    kernels); Ridge baseline with a tuned alpha; selection and reporting both
#    in log units (skill_rmse_log); one shared INNER_TRAIN_SAMPLE.
# 3: buffer is now a real edge-to-edge gap in km (it was a no-op before), so
#    the outer AND inner folds themselves differ from protocol 2.
# 4: K_FOLDS 7 -> 15, INNER_K_FOLDS 5 -> 10, and the reported hyperparameter is
#    the per-parameter mode rather than the modal full configuration.
# 5: BUFFER_MODE='point' — the buffer is enforced at point resolution instead
#    of by discarding whole blocks, so the folds themselves differ again.
CKPT_PROTOCOL = 5


def ckpt_save(state):
    state = dict(state)
    state['protocol'] = CKPT_PROTOCOL
    tmp = CKPT_PATH + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(state, f, protocol=4)
    os.replace(tmp, CKPT_PATH)
    print(f"[ckpt] Saved  stage={state.get('stage','?')}  "
          f"fold={state.get('fold','?')}", flush=True)

def ckpt_load():
    if not os.path.exists(CKPT_PATH):
        return None
    with open(CKPT_PATH, 'rb') as f:
        s = pickle.load(f)
    got = s.get('protocol', 1)
    if got != CKPT_PROTOCOL:
        print("\n" + "!" * 70, flush=True)
        print(f"[ckpt] REFUSING TO RESUME: checkpoint was written under "
              f"protocol {got}, this script is protocol {CKPT_PROTOCOL}.",
              flush=True)
        print("       Folds already scored used a different inner split, "
              "criterion or", flush=True)
        print("       search grid, so resuming would average them with folds "
              "scored under", flush=True)
        print("       the new one. The resulting table would look fine and be "
              "reproducible", flush=True)
        print("       by no single run.", flush=True)
        print(f"       Delete {CKPT_PATH} (and the stale final models, see the "
              "note in", flush=True)
        print("       section 10) and start clean.", flush=True)
        print("!" * 70 + "\n", flush=True)
        raise SystemExit(1)
    print(f"[ckpt] Loaded stage={s.get('stage','?')}  "
          f"fold={s.get('fold','?')}  protocol={got}", flush=True)
    return s

def requeue_and_exit():
    jid = os.environ.get('SLURM_JOB_ID')
    if jid:
        print(f"[ckpt] Requeueing job {jid}...", flush=True)
        try:
            subprocess.run(['scontrol', 'requeue', jid], check=True)
            print(f"[ckpt] Job {jid} requeued.", flush=True)
        except Exception as e:
            print(f"[ckpt] Requeue failed: {e}", flush=True)
    import sys; sys.exit(0)

class _CheckpointSignal(Exception):
    pass


# ============================================================
# CONFIG
# ============================================================
N_DEPTHS    = 241
EPS         = 1e-9
# Outer folds. Raised from 7 once the buffer became real.
#
# The buffer removes any training block touching a validation block, so the
# cost is set by the PERIMETER of the validation set, not by the buffer width.
# With randomly scattered blocks a training block has 8 neighbours and a 1-in-k
# chance each of being validation, so the fraction excluded is ~1-(1-1/k)^8:
# 71% at k=7, but only 45% at k=15. Every block is still validated exactly
# once across the run, so total validation coverage is identical — the extra
# folds buy training data, not test coverage.
#
#   k     train    valid/fold   buffered out   relative compute
#   7     25.5%      14.3%         60.2%           1.0x
#  15     54.3%       6.7%         39.0%           2.1x
#  20     63.5%       5.0%         31.5%           2.9x
#
# Per-fold scores are noisier at 6.7% validation, but there are more of them
# and every point is still scored exactly once, so the pooled estimate is more
# precise rather than less. If the per-fold hyperparameter selections start
# disagreeing (watch "MODAL configuration (n/15 folds)"), that is the smaller
# inner training partitions talking, not a finding.
K_FOLDS     = 11
BLOCK_DEG   = 10
_KM_PER_DEG = 111.32        # metres per degree of latitude, in km
# Buffer in KILOMETRES, measured EDGE-TO-EDGE between blocks — not degrees
# between block centres.
#
# The old form compared centre-to-centre distance in raw degrees against
# BUFFER_DEG = 5. Distinct block centres are at least BLOCK_DEG = 10 deg apart
# by construction, so `d_min > 5` was true for every training block and NOTHING
# was ever excluded: the "buffered" CV had a zero-width buffer, and the
# `buffered out=` column printed 0 in every fold. Validation points sitting on
# a block edge had training points a fraction of a degree away.
#
# Two fixes, both needed:
#   * EDGE-TO-EDGE. Subtract the block extent, so the number means the gap
#     between the blocks rather than between their centres.
#   * KILOMETRES. A degree of longitude is 111 km at the equator and 56 km at
#     60 deg. Measuring in raw degrees makes the buffer physically weakest
#     exactly where coverage is thinnest (Southern Ocean, subpolar North).
#
# Set from the measured autocorrelation of beta anomalies: e-folding scales are
# 281 km (0-200 dbar), 354 km (200-500) and 811 km (500-1500). BUFFER_KM = 0
# already gives a full block of separation (1112 km at the equator, 556 km at
# 60 deg); a positive value pushes the high-latitude case past the deep-band
# scale too. Each step costs training rows — the fold printout reports how many
# and what separation was actually achieved.
BUFFER_KM = 300.0

# How the buffer is enforced.
#   'point'  drop individual training POINTS within BUFFER_KM of any validation
#            point. The gap is then EXACTLY BUFFER_KM by construction.
#   'block'  drop whole training BLOCKS whose edge lies within BUFFER_KM of a
#            validation block's edge.
#
# 'block' is far more expensive for the same separation, because it discards a
# whole 10 deg block (1112 km at the equator) to enforce a 300 km gap — and the
# cost barely responds to BUFFER_KM, since any touching block has an edge gap of
# zero and goes regardless:
#
#   mode    BUFFER_KM   train    dropped   realised gap
#   block       0       54.9%     38.5%       296 km
#   block     300       54.3%     39.0%       383 km
#   point       0       93.3%      0.0%         3 km
#   point     300       84.1%      9.3%       300 km      <- same guarantee,
#   point     600       71.7%     21.6%       600 km         30 pts more data
#
# With 'point', BUFFER_KM means what it says and can be set straight from the
# measured autocorrelation: 281 / 354 / 811 km for the three depth bands.
BUFFER_MODE = 'point'
assert BUFFER_MODE in ('point', 'block')
_EARTH_R_KM = 6371.0


def _xyz_km(lat, lon):
    """Lat/lon -> 3-D Cartesian on a sphere of radius R, for cKDTree."""
    la, lo = np.deg2rad(np.asarray(lat, float)), np.deg2rad(np.asarray(lon, float))
    cl = np.cos(la)
    return np.stack([cl * np.cos(lo), cl * np.sin(lo), np.sin(la)],
                    axis=1) * _EARTH_R_KM


def _chord(arc_km):
    """Great-circle arc -> straight-line chord, which is what cKDTree measures."""
    return 2.0 * _EARTH_R_KM * np.sin(min(arc_km, np.pi * _EARTH_R_KM)
                                      / (2.0 * _EARTH_R_KM))
TARGET_NAMES  = ['Beta_slope', 'Biovolume']
FEATURE_NAMES = ['NO3', 'Chl', 'Bathymetry', 'MLD', 'Salinity',
                 'TEMP', 'O2', 'Thflx', 'NPP', 'Depth',
                 'month_sin', 'month_cos']
OUTPUT_DIR  = './ml_outputs'
SCATTER_DIR = os.path.join(OUTPUT_DIR, 'scatter_plots')
os.makedirs(SCATTER_DIR, exist_ok=True)

N_JOBS_RF_INNER = 1
N_JOBS_RF_OUTER = 16
N_JOBS_RF_FINAL = 16

# Retrain the RF even when a checkpoint reports it complete. Needed after any
# change to rf_tune (e.g. switching RandomizedSearchCV -> GridSearchCV), since
# the cached rf_state would otherwise be reused silently. SVGP progress in the
# same checkpoint is preserved — see the merge logic in cv_rf and Section 9.
# Remember to also delete model_checkpoint/rf_final.pkl, or the FINAL RF is
# reloaded from disk with the old hyperparameters.
FORCE_RERUN_RF = True

# SVGP knobs — per-target
GP_INDUCING = {'beta': 500,  'biovol': 1000}
GP_KERNEL   = {'beta': 'rbf', 'biovol': 'rbf'}
GP_BATCH_SIZE = 4096
GP_EPOCHS     = 100
GP_LR         = 0.01
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── Shared nested-CV protocol (RF, SVGP and Ridge all use these) ─────────────
# Any model-specific deviation from these settings breaks comparability of the
# outer skill scores, so they live in one place and are consumed by
# _inner_folds() below.
# Inner holdout = 1 of INNER_K_FOLDS block groups. Raised from 5 to 10 once the
# buffer became real, because the buffer eats the REMAINDER, not the holdout:
# ~1-(1-1/k)^8 of the surviving blocks touch the holdout and are dropped.
#
#   k    inner train   inner valid   buffered   train:valid
#   5       13.8%         19.8%        66.5%       0.70   <- holdout bigger
#  10       42.9%          9.8%        47.3%       4.38      than the training
#  15       55.2%          6.5%        38.3%       8.52      set
#
# At k=5 the run log showed folds selecting on as few as 11,838 rows while the
# OUTER model was then fitted on 88,000-136,000. Hyperparameters chosen for a
# dataset 3-8x smaller than the one they are applied to are biased toward
# shallower trees and fewer inducing points, i.e. systematically underfit.
# k=10 keeps a holdout of ~10k rows on this data, ample for ranking candidates.
INNER_K_FOLDS   = 10
INNER_N_USE     = 1         # hold out the FIRST group only (20% of blocks)
                            # (set INNER_N_USE = INNER_K_FOLDS for full 5-fold
                            #  inner CV; the tuners loop over whatever is given)
INNER_SEED      = 1         # different block permutation from the outer CV (0)
# Hyperparameters are re-selected INSIDE EVERY outer fold, using only that
# fold's training partition. The configuration reported for the paper (and used
# for the final all-data models) is the MODE across the outer folds. Setting
# TUNE_EVERY_FOLD = False reverts to the cheaper "tune on fold TUNE_ON_FOLD and
# reuse" behaviour, which is NOT fully nested: fold j's validation blocks sit
# inside fold TUNE_ON_FOLD's training partition and so take part in the
# selection that is later scored on them.
TUNE_EVERY_FOLD = True
TUNE_ON_FOLD    = 0         # only used when TUNE_EVERY_FOLD is False
# Optional uniform cap on the inner VALIDATION rows. Applied inside
# _inner_folds(), i.e. identically for RF, SVGP and Ridge, so capping does not
# reintroduce an asymmetry. None = use every validation row.
INNER_VALID_SAMPLE = None   # e.g. 100_000 if the RF inner search gets too slow

# NOMINAL description. The realised split is not 80/20 once the buffer is real:
# the inner holdout is 1 of INNER_K_FOLDS block groups (20% of blocks), but the
# buffer then removes every remaining block that touches one of them, which at
# INNER_K_FOLDS=5 is ~1-(4/5)^8 = 83% of them. The realised inner training
# fraction is therefore ~20%, not 80%. _inner_folds() prints the actual numbers
# on its first call so the two are never confused.
INNER_DESC = (f"1-of-{INNER_K_FOLDS} buffered spatial holdout"
              if INNER_N_USE == 1 else
              f"{INNER_N_USE}-of-{INNER_K_FOLDS} buffered spatial CV")

# Row cap applied INSIDE each inner training split (never before the folds are
# built, so the split geometry is unaffected). Shared by all three models: the
# number of rows a candidate is fitted on during selection is now the same for
# RF, SVGP and Ridge, so capacity-type hyperparameters are chosen under equal
# information. Ridge respects it too, though it would not need the cap.
INNER_TRAIN_SAMPLE = 50_000

# SVGP nested-CV tuning knobs
GP_TUNE_EPOCHS  = 50        # reduced epochs for inner CV (vs GP_EPOCHS for final)
GP_TUNE_SAMPLE  = INNER_TRAIN_SAMPLE
RF_TUNE_SAMPLE  = INNER_TRAIN_SAMPLE
RIDGE_TUNE_SAMPLE = INNER_TRAIN_SAMPLE

RIDGE_ALPHA_GRID = list(np.logspace(-4, 4, 9))


# 4 inducing-point counts x 3 kernels = 12 candidates. The kernel is a real
# degree of freedom again (it was rbf-only), so the SVGP search budget is no
# longer an order of magnitude smaller than the RF's.
SVGP_PARAM_GRID = [{'n_inducing': n, 'kernel': kn}
                   for n in (300, 500, 750, 1000)
                   for kn in ('rbf', 'matern15', 'matern25')]


# Feature-importance knobs
SHAP_RF_EXPLAIN     = 20_000
SHAP_SVGP_EXPLAIN   = 5_000
SHAP_TESTGRID_N     = 150_000
SHAP_SVGP_BG        = 200
SHAP_SVGP_NSAMPLES  = 64
SHAP_SVGP_CHUNK     = 5_000
PERM_SAMPLE_RF      = 20_000   # rows for per-variable permutation (RF)
PERM_SAMPLE_SVGP    = 5_000    # rows for per-variable permutation (SVGP)
GROUP_SAMPLE_RF     = 20_000
GROUP_SAMPLE_SVGP   = 5_000
GROUP_N_REPEATS     = 5
PERM_N_REPEATS      = 5        # repeats for single-feature permutation
CLUSTER_DIST_THRESH = 0.5


# ============================================================
# Inline target diagnostic
# ============================================================
def diagnose_target(y_col, depth, eps, name='Biovolume'):
    y_col = np.asarray(y_col, dtype=np.float64)
    print("\n" + "=" * 60)
    print(f"{name.upper()} TARGET DIAGNOSTIC  (EPS = {eps:.1e})")
    print("=" * 60)

    n = len(y_col)
    n_zero = int((y_col == 0).sum())
    n_lt6  = int((y_col < 1e-6).sum())
    n_lt3  = int((y_col < 1e-3).sum())
    print(f"Total rows:        {n:,}")
    print(f"Exact zeros:       {n_zero:,}  ({100*n_zero/n:.2f}%)")
    print(f"Below 1e-6:        {n_lt6:,}  ({100*n_lt6/n:.2f}%)")
    print(f"Below 1e-3:        {n_lt3:,}  ({100*n_lt3/n:.2f}%)")

    pos = y_col[y_col > 0]
    tail_ratio = depth_ratio = 0.0
    median_pos = np.nan
    if len(pos):
        median_pos = float(np.median(pos))
        print("\nPositive percentiles:")
        for p in [50, 75, 90, 95, 99, 99.5, 99.9, 100]:
            print(f"  {p:5.1f}%: {np.percentile(pos, p):.3e}")
        tail_ratio = float(pos.mean() / median_pos)
        print(f"\n  mean/median ratio: {tail_ratio:.2f}  "
              f"(>5 heavy-tailed, >10 severe)")
        frac_below_eps = float((pos < eps).mean())
        print("\n  EPS-floor interaction:")
        print(f"    median(positive) / EPS = {median_pos/eps:.2f}")
        print(f"    positives below EPS     = {100*frac_below_eps:.1f}%")
        if median_pos < 10 * eps or frac_below_eps > 0.05:
            print("    ** EPS is comparable to the data scale **")
        else:
            print("    EPS is well below the data scale — not the bottleneck.")

    dfb = pd.DataFrame({'depth': np.asarray(depth, dtype=np.float64), 't': y_col})
    dfb['band'] = pd.cut(dfb['depth'], bins=[0, 50, 200, 500, 1000, 6000],
                         labels=['0-50', '50-200', '200-500', '500-1000', '1000+'])
    stats = dfb.groupby('band', observed=True)['t'].agg(
        ['count', 'mean', 'median', 'min', 'max'])
    print("\nBy depth band:")
    print(stats)
    if len(stats) > 1:
        depth_ratio = float(stats['median'].max()
                            / max(stats['median'].min(), 1e-9))
        print(f"\n  median ratio (max band / min band): {depth_ratio:.1f}")

    print("\n  --- summary flags ---")
    if (n_zero / n) > 0.05 or (n_lt3 / n) > 0.10:
        print("  [!] zero/near-zero inflation -> consider two-stage (hurdle) model")
    if tail_ratio > 5:
        print("  [!] heavy tail -> consider winsorizing at the 99th pct in training")
    if depth_ratio > 10:
        print("  [!] depth heterogeneity -> consider per-depth-zone models")
    print("=" * 60)
    return {'median_pos': median_pos, 'tail_ratio': tail_ratio,
            'depth_ratio': depth_ratio}


# ============================================================
# 1) Load + stack all per-depth training files
# ============================================================
def stack_train_var(var_key, ravel=False):
    chunks = []
    for j in range(N_DEPTHS):
        a = io.loadmat(f'data_{var_key}_Rf1_{j}.mat')[var_key].astype(np.float32)
        chunks.append(a.ravel() if ravel else a)
    out = np.vstack(chunks) if not ravel else np.concatenate(chunks)
    del chunks; gc.collect()
    return out


print("Loading and stacking per-depth training files...")
t0 = time.time()
X             = stack_train_var('X')
y             = stack_train_var('y')
lon_all       = stack_train_var('lon',       ravel=True)
lat_all       = stack_train_var('lat',       ravel=True)
month_sin_all = stack_train_var('month_sin', ravel=True)
month_cos_all = stack_train_var('month_cos', ravel=True)
depth_all     = stack_train_var('depth',     ravel=True)
print(f"Stacked X={X.shape}, y={y.shape}  ({time.time()-t0:.1f}s)")


# ============================================================
# 2) Drop NaN rows
# ============================================================
valid = (
    ~np.isnan(X).any(axis=1)
    & ~np.isnan(y).any(axis=1)
)
diagnose_target(y[valid][:, 1], depth_all[valid], EPS, name='Biovolume')

valid = (
    valid
    & (y[:, 0] > 0)
    & (y[:, 1] >= 0)
)
X             = X[valid];             y         = y[valid]
lon_all       = lon_all[valid];       lat_all   = lat_all[valid]
month_sin_all = month_sin_all[valid]; month_cos_all = month_cos_all[valid]
depth_all     = depth_all[valid]
del valid; gc.collect()
print(f"After NaN drop: {X.shape[0]:,} rows")

# Reconstruct integer month from cyclic encoding for metadata saves
month_all = (np.round(
    np.arctan2(month_sin_all, month_cos_all) * 12 / (2 * np.pi)
).astype(int) % 12) + 1


# ============================================================
# 3) Buffered spatial CV
# ============================================================
def buffered_spatial_cv(lat, lon, k_folds=10, block_deg=10, buffer_km=300.0,
                        buffer_mode=None, random_state=0, verbose=True):
    """
    Spatial CV on lat/lon blocks with a real, physical buffer.

    Blocks are block_deg squares. Each fold holds out a random set of blocks;
    any training block whose EDGE lies within buffer_km of a validation block's
    edge is dropped from BOTH sets, so a genuine gap separates the two.

    Separation is computed edge-to-edge and in kilometres (longitude scaled by
    cos(lat)), because the previous centre-to-centre-in-degrees form could
    never exclude anything: block centres are >= block_deg apart, which always
    exceeded a buffer expressed in single-digit degrees.
    """
    lat = np.asarray(lat); lon = np.asarray(lon)
    buffer_mode = BUFFER_MODE if buffer_mode is None else buffer_mode
    _XYZ = _xyz_km(lat, lon) if buffer_mode == 'point' else None
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    block_lon = np.floor((lon180 + 180.0) / block_deg).astype(np.int32)
    block_lat = np.floor((lat   + 90.0)  / block_deg).astype(np.int32)
    blocks    = np.stack([block_lat, block_lon], axis=1)

    uniq_blocks, inverse = np.unique(blocks, axis=0, return_inverse=True)
    n_blocks = len(uniq_blocks)
    block_lat_center = uniq_blocks[:, 0] * block_deg + block_deg / 2.0 - 90.0
    block_lon_center = uniq_blocks[:, 1] * block_deg + block_deg / 2.0 - 180.0

    rng = np.random.RandomState(random_state)
    block_order = rng.permutation(n_blocks)

    kf = KFold(n_splits=k_folds, shuffle=False)
    for fold, (_, valid_block_local) in enumerate(kf.split(block_order)):
        valid_blocks     = block_order[valid_block_local]
        train_blocks_all = np.setdiff1d(np.arange(n_blocks), valid_blocks)

        v_lats = block_lat_center[valid_blocks]
        v_lons = block_lon_center[valid_blocks]
        t_lats = block_lat_center[train_blocks_all]
        t_lons = block_lon_center[train_blocks_all]

        # gap between block EDGES, per axis, in degrees
        d_lat = np.abs(t_lats[:, None] - v_lats[None, :])
        d_lon = np.abs(t_lons[:, None] - v_lons[None, :])
        d_lon = np.minimum(d_lon, 360.0 - d_lon)
        gap_lat = np.maximum(d_lat - block_deg, 0.0)
        gap_lon = np.maximum(d_lon - block_deg, 0.0)

        # convert to km; a degree of longitude shrinks as cos(lat)
        mid_lat = 0.5 * (t_lats[:, None] + v_lats[None, :])
        km_lat = gap_lat * _KM_PER_DEG
        km_lon = gap_lon * _KM_PER_DEG * np.cos(
            np.deg2rad(np.clip(mid_lat, -89.0, 89.0)))
        d_km = np.sqrt(km_lat ** 2 + km_lon ** 2).min(axis=1)

        valid_mask = np.isin(inverse, valid_blocks)
        if buffer_mode == 'block':
            kept_train_blocks = train_blocks_all[d_km > buffer_km]
            train_mask = np.isin(inverse, kept_train_blocks)
        else:
            # POINT mode: keep every non-validation point that is farther than
            # buffer_km from the NEAREST validation point. Blocks still define
            # what is held out; the buffer is then applied at point resolution,
            # so a neighbouring block contributes everything beyond the collar
            # instead of being discarded whole.
            train_mask = ~valid_mask
            if buffer_km > 0 and valid_mask.any() and train_mask.any():
                tree = cKDTree(_XYZ[valid_mask])
                idx = np.flatnonzero(train_mask)
                dd, _ = tree.query(_XYZ[idx], k=1, workers=-1)
                train_mask = np.zeros_like(valid_mask)
                train_mask[idx[dd > _chord(buffer_km)]] = True
                del tree, idx, dd
        n_buffer   = (~valid_mask & ~train_mask).sum()
        # separation actually achieved, measured point-to-point — compare it
        # against the autocorrelation scale rather than trusting the request
        sep_min = float('nan')
        if valid_mask.any() and train_mask.any():
            tree = cKDTree(_XYZ[valid_mask])
            ti = np.flatnonzero(train_mask)
            if len(ti) > 200000:
                ti = np.random.RandomState(7).choice(ti, 200000, replace=False)
            dd, _ = tree.query(_XYZ[ti], k=1, workers=-1)
            sep_min = float(2.0 * _EARTH_R_KM
                            * np.arcsin(np.clip(dd.min() / (2.0 * _EARTH_R_KM),
                                                0.0, 1.0)))
            del tree, ti, dd

        if verbose:
            print(f"  fold {fold+1:2d}: train={train_mask.sum():>7,}  "
                  f"valid={valid_mask.sum():>6,}  buffered out={n_buffer:>6,}"
                  f"  min train-valid gap={sep_min:>6.0f} km")
        yield np.where(train_mask)[0], np.where(valid_mask)[0]


_INNER_REPORTED = False


def _inner_folds(lat_tr, lon_tr):
    """
    Inner split(s) for hyperparameter selection, shared by rf_tune(),
    svgp_tune() and ridge_tune().

    The outer-training blocks are divided into INNER_K_FOLDS spatial groups and
    the first INNER_N_USE of them are returned. With INNER_K_FOLDS=5 and
    INNER_N_USE=1, 20% of the blocks are held out for evaluating candidate
    hyperparameters and the rest train them — MINUS the buffer ring, which at
    INNER_K_FOLDS=5 removes most of what is left, so the realised training
    fraction is ~20%, not 80%. The realised numbers are printed on the first
    call; do not describe this as an 80/20 split. Setting INNER_N_USE=INNER_K_FOLDS turns the same call
    into a full 5-fold inner CV without touching any of the three tuners.

    Unlike a plain random holdout, this one is BLOCKED and BUFFERED, so
    hyperparameters are selected under the same spatial-extrapolation regime in
    which the outer folds score them.

    Called with the FULL outer-training lat/lon (never a subsample), so all
    three models see byte-identical inner train/validation index sets. Any
    speed-capping subsample is applied *inside* the returned training splits by
    the caller, which changes how many rows a model is fitted on but not which
    blocks define the split.

    Returns a list of (train_idx, valid_idx) arrays indexing into the outer
    training set.
    """
    folds = list(buffered_spatial_cv(
        lat_tr, lon_tr,
        k_folds=INNER_K_FOLDS,
        block_deg=BLOCK_DEG,
        buffer_km=BUFFER_KM,
        buffer_mode=BUFFER_MODE,
        random_state=INNER_SEED,
        verbose=False,
    ))[:INNER_N_USE]
    global _INNER_REPORTED
    if not _INNER_REPORTED:
        _INNER_REPORTED = True
        n = len(lat_tr)
        for j, (a, b) in enumerate(folds):
            print(f"  [inner split {j+1}] of this fold's training partition: "
                  f"train {len(a)/n:.1%} ({len(a):,} rows)  "
                  f"valid {len(b)/n:.1%} ({len(b):,})  "
                  f"buffered out {1-(len(a)+len(b))/n:.1%}", flush=True)
        if folds and len(folds[0][0]) < INNER_TRAIN_SAMPLE:
            print(f"  [inner split] NOTE: only {len(folds[0][0]):,} training "
                  f"rows available, below the INNER_TRAIN_SAMPLE cap of "
                  f"{INNER_TRAIN_SAMPLE:,}. All three models still see the same "
                  f"rows, so the comparison holds, but selection is being made "
                  f"on less data than intended.", flush=True)

    if INNER_VALID_SAMPLE is not None:
        capped = []
        for i_fold, (i_tr, i_va) in enumerate(folds):
            if len(i_va) > INNER_VALID_SAMPLE:
                sub  = np.random.RandomState(1000 + i_fold).choice(
                    len(i_va), INNER_VALID_SAMPLE, replace=False)
                i_va = np.sort(i_va[sub])
            capped.append((i_tr, i_va))
        folds = capped
    return folds


def _cfg_key(cfg):
    """Hashable, order-independent key for a hyperparameter configuration."""
    if isinstance(cfg, dict):
        return tuple(sorted((k, str(v)) for k, v in cfg.items()))
    return cfg


# A 162-point joint grid cannot be resolved by one holdout per fold: the run
# log showed a modal configuration supported by 2 folds out of 15. The MARGINAL
# modes were far better determined over the same folds (max_features='sqrt' in
# 14/15, min_samples_split=2 in 10/15, max_depth=20 in 8/15), because each
# marginal pools over the other four axes. Reporting the per-parameter mode
# therefore rests on much more evidence than reporting the joint one.
#
# The cost is that the combination need not be one any fold actually scored.
# Both are always computed and printed, and a warning fires when they differ,
# so the choice is visible rather than silent. Set False to report the joint
# mode instead.
MODAL_PER_PARAM = True


def modal_config(per_fold, label='', per_param=None):
    """
    Modal hyperparameter configuration across the outer folds.

    Each outer fold selects independently on its own training partition; the
    configuration reported for the paper — and used to fit the final all-data
    model — is the one chosen most often. Ties are broken in favour of the
    configuration that appeared in the earliest fold, which is deterministic
    and does not peek at any outer validation score.
    """
    vals = [c for c in per_fold if c is not None]
    if not vals:
        return None
    counts = {}
    for i, c in enumerate(vals):
        key = _cfg_key(c)
        if key not in counts:
            counts[key] = [0, i, c]          # count, first fold, value
        counts[key][0] += 1
    best_key = max(counts, key=lambda k: (counts[k][0], -counts[k][1]))
    n_hit, _, joint = counts[best_key]

    # per-parameter (marginal) mode, for dict-valued configurations
    per_param = MODAL_PER_PARAM if per_param is None else per_param
    marginal, support = None, {}
    if per_param and isinstance(vals[0], dict):
        marginal = {}
        for key in vals[0]:
            cnt = {}
            for i, c in enumerate(vals):
                kk = str(c[key])
                if kk not in cnt:
                    cnt[kk] = [0, i, c[key]]
                cnt[kk][0] += 1
            bk = max(cnt, key=lambda z: (cnt[z][0], -cnt[z][1]))
            marginal[key] = cnt[bk][2]
            support[key] = cnt[bk][0]

    if label:
        print(f"  [{label}] per-fold selections:")
        for i, c in enumerate(per_fold):
            print(f"      fold {i+1}: {c}")
        print(f"  [{label}] joint mode ({n_hit}/{len(vals)} folds): {joint}",
              flush=True)
        if marginal is not None:
            print(f"  [{label}] per-parameter mode (support out of "
                  f"{len(vals)} folds):", flush=True)
            for key in marginal:
                print(f"      {key} = {marginal[key]}   [{support[key]}/"
                      f"{len(vals)}]", flush=True)
            if _cfg_key(marginal) != _cfg_key(joint):
                print(f"  [{label}] NOTE: the per-parameter mode differs from "
                      f"the joint mode. It rests on more folds per axis "
                      f"({min(support.values())}-{max(support.values())}/"
                      f"{len(vals)}) than the joint mode does "
                      f"({n_hit}/{len(vals)}), but the combination itself may "
                      f"not have been scored by any fold.", flush=True)
        chosen = marginal if marginal is not None else joint
        print(f"  [{label}] USING: {chosen}", flush=True)
        if n_hit == 1 and len(vals) > 1 and marginal is None:
            print(f"  [{label}] WARNING: no configuration was selected more "
                  f"than once — the inner selection is unstable across folds.",
                  flush=True)
    return marginal if marginal is not None else joint


def _as_per_fold(x, k, width=None):
    """
    Validate a per-fold hyperparameter record restored from a checkpoint.

    Returns a fresh list of length k, or a list of Nones if the stored object
    predates the per-fold scheme (e.g. a single dict written by an older run).
    """
    empty = [([None] * width if width else None) for _ in range(k)]
    if not isinstance(x, list) or len(x) != k:
        return empty
    if width is not None and not all(
            (e is None) or (isinstance(e, list) and len(e) == width) for e in x):
        return empty
    return [list(e) if isinstance(e, list) else e for e in x]


def _subsample_train_split(i_tr, cap, seed):
    """Speed cap applied inside an inner training split (validation untouched)."""
    if cap is None or len(i_tr) <= cap:
        return i_tr
    sub = np.random.RandomState(seed).choice(len(i_tr), cap, replace=False)
    return i_tr[sub]


print("\nBuilding buffered spatial folds:")
fold_indices = list(buffered_spatial_cv(lat_all, lon_all,
                                        k_folds=K_FOLDS,
                                        block_deg=BLOCK_DEG,
                                        buffer_km=BUFFER_KM,
                                        random_state=0,
                                        verbose=True))


# ============================================================
# 4) Log-transform targets
# ============================================================
y_log = np.log(y + EPS).astype(np.float32)
del y; gc.collect()


# ============================================================
# 5) RANDOM FOREST
# ============================================================
def rf_tune(X_tr, y_log_tr, lat_tr, lon_tr, n_iter=9):
    """
    Grid search over RF hyperparameters on the shared inner split(s) from
    _inner_folds() — protocol-identical to svgp_tune() and ridge_tune().
    With the default knobs that is a single 1-of-5 buffered spatial holdout.

    Takes RAW (unscaled) X and log-targets: feature standardisation is done by
    a Pipeline and target standardisation by a TransformedTargetRegressor, so
    both scalers are refit on each inner training split (no leakage from the
    inner validation blocks) exactly as in svgp_tune()/ridge_tune().

    Because TransformedTargetRegressor inverse-transforms before scoring, the
    reported inner MSE is in LOG-TARGET UNITS — the same space used by the
    other two tuners.
    """
    inner_folds = _inner_folds(lat_tr, lon_tr)
    inner_cv = [(_subsample_train_split(i_tr, RF_TUNE_SAMPLE, seed=i_fold), i_va)
                for i_fold, (i_tr, i_va) in enumerate(inner_folds)]

    # GridSearchCV ships X_tr to every worker, but only the rows named in
    # inner_cv are ever touched: the training rows are capped at
    # INNER_TRAIN_SAMPLE and the validation rows are one inner block. Passing
    # the whole outer training partition (millions of rows) instead was
    # inflating each worker's footprint several-fold for no benefit, which is
    # what produces joblib's "a worker stopped while some jobs were given to
    # the executor" warning. Compact to the used rows and remap the indices.
    _used = np.unique(np.concatenate([np.concatenate([t, v])
                                      for t, v in inner_cv]))
    _remap = np.full(len(X_tr), -1, dtype=np.int64)
    _remap[_used] = np.arange(len(_used))
    X_fit = np.ascontiguousarray(X_tr[_used])
    y_fit = np.ascontiguousarray(y_log_tr[_used])
    inner_cv = [(_remap[t], _remap[v]) for t, v in inner_cv]

    print(f"  [RF tune] {INNER_DESC}  "
          f"(train rows: {[len(t) for t, _ in inner_cv]}  "
          f"valid rows: {[len(v) for _, v in inner_cv]})", flush=True)
    print(f"  [RF tune] shipping {len(_used):,} of {len(X_tr):,} rows to the "
          f"workers ({100*len(_used)/max(len(X_tr),1):.0f}%)", flush=True)

    param_dist = {
        'regressor__rf__n_estimators':      [100, 200, 300],
        'regressor__rf__max_depth':         [10, 15, 20],
        'regressor__rf__min_samples_split': [2, 4, 8],
        'regressor__rf__min_samples_leaf':  [2, 4, 8],
        'regressor__rf__max_features':      ['sqrt', 0.5],
    }
    est = TransformedTargetRegressor(
        regressor=Pipeline([
            ('sx', StandardScaler()),
            ('rf', RandomForestRegressor(random_state=10,
                                         n_jobs=N_JOBS_RF_INNER)),
        ]),
        transformer=StandardScaler(),
    )
    search = GridSearchCV(
        est, param_dist,
        cv=inner_cv,
        scoring='neg_mean_squared_error',
        n_jobs=N_JOBS_RF_OUTER,
        verbose=1,
    )
    #search = RandomizedSearchCV(
    #    est, param_dist,
    #    n_iter=n_iter,
    #    cv=inner_cv,                         # spatially blocked inner folds
    #    scoring='neg_mean_squared_error',
    #    n_jobs=N_JOBS_RF_OUTER,
    #    verbose=1,
    #    random_state=0,
    #)
    search.fit(X_fit, y_fit)
    # strip the 'regressor__rf__' prefix so rf_train() can consume the dict
    best = {k.split('__')[-1]: v for k, v in search.best_params_.items()}
    print(f"  [RF tune] best params: {best}  "
          f"(inner MSE={-search.best_score_:.4f}, log units)", flush=True)
    del search; gc.collect()
    return best


def svgp_tune(X_tr, y_log_tr, lat_tr, lon_tr, target_idx, target_name,
              _resume=None, _on_config=None):
    """
    Exhaustive grid search over SVGP_PARAM_GRID on the shared inner split(s)
    from _inner_folds() — protocol-identical to rf_tune() and ridge_tune()
    (same splits, same MSE-in-log-units criterion). With the default knobs
    that is a single 1-of-5 buffered spatial holdout.

    Uses GP_TUNE_EPOCHS (not GP_EPOCHS) and subsamples to GP_TUNE_SAMPLE
    rows per inner training split for speed. Returns best
    {'n_inducing', 'kernel'}.
    """
    inner_folds = _inner_folds(lat_tr, lon_tr)

    # Config-level resume. train_single_svgp() can raise _CheckpointSignal at
    # any epoch, and with TUNE_EVERY_FOLD that can now happen deep into a job.
    # Without this, every requeue would restart the fold's tuning from config 0
    # and a fold whose full grid exceeds one job's wall clock would requeue for
    # ever, making no progress. `_resume` carries the configs already scored;
    # `_on_config` hands each new score back so the caller can checkpoint it.
    done = dict(_resume) if _resume else {}
    if done:
        print(f"  [SVGP tune] resuming {target_name}: "
              f"{len(done)}/{len(SVGP_PARAM_GRID)} configs already scored",
              flush=True)

    print(f"  [SVGP tune] target={target_name}  "
          f"grid={len(SVGP_PARAM_GRID)} configs × {INNER_DESC}", flush=True)

    y_col       = y_log_tr[:, target_idx]
    best_mse    = np.inf
    best_params = SVGP_PARAM_GRID[0]

    for cfg_idx, cfg in enumerate(SVGP_PARAM_GRID):
        if cfg_idx in done:
            mean_mse = done[cfg_idx]
            print(f"    cfg={cfg}  inner_MSE={mean_mse:.4f}  [restored]",
                  flush=True)
            if mean_mse < best_mse:
                best_mse, best_params = mean_mse, cfg
            continue
        fold_mses = []

        for i_fold, (i_tr, i_va) in enumerate(inner_folds):
            # speed cap applied inside the training split (same helper as RF)
            i_tr_s = _subsample_train_split(i_tr, GP_TUNE_SAMPLE, seed=i_fold)

            sx_i = StandardScaler().fit(X_tr[i_tr_s])
            Xt_i = sx_i.transform(X_tr[i_tr_s]).astype(np.float32)
            Xv_i = sx_i.transform(X_tr[i_va ]).astype(np.float32)

            sy_i = StandardScaler().fit(y_col[i_tr_s].reshape(-1, 1))
            yt_i = sy_i.transform(y_col[i_tr_s].reshape(-1, 1)).ravel().astype(np.float32)
            yv_i = sy_i.transform(y_col[i_va ].reshape(-1, 1)).ravel().astype(np.float32)

            try:
                wrap = train_single_svgp(
                    Xt_i, yt_i,
                    n_inducing=cfg['n_inducing'],
                    kernel=cfg['kernel'],
                    label=f"{target_name}_tune_c{cfg_idx}_f{i_fold}",
                    verbose=False,
                    n_epochs_override=GP_TUNE_EPOCHS,
                )
            except _CheckpointSignal:
                raise   # propagate — outer loop handles requeue

            # Score in LOG-TARGET UNITS (inverse-transform first) so the inner
            # criterion is identical to rf_tune()/ridge_tune().
            pred_s   = wrap.predict_np(Xv_i).reshape(-1, 1)
            pred_log = sy_i.inverse_transform(pred_s).ravel()
            mse_i    = float(mean_squared_error(y_col[i_va], pred_log))
            fold_mses.append(mse_i)

            del wrap, Xt_i, Xv_i, yt_i, yv_i
            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

        mean_mse = float(np.mean(fold_mses))
        done[cfg_idx] = mean_mse
        if _on_config is not None:
            _on_config(cfg_idx, mean_mse)     # lets the caller checkpoint it
        print(f"    cfg={cfg}  inner_MSE={mean_mse:.4f}  "
              f"(folds: {[f'{v:.4f}' for v in fold_mses]})", flush=True)

        if mean_mse < best_mse:
            best_mse    = mean_mse
            best_params = cfg

    print(f"  [SVGP tune] best for {target_name}: {best_params}  "
          f"(MSE={best_mse:.4f})", flush=True)
    return best_params


def rf_train(X_tr, y_tr, params=None, tuner=None):
    p = params if params is not None else {}
    rf = RandomForestRegressor(**p, random_state=10, n_jobs=N_JOBS_RF_FINAL)
    rf.fit(X_tr, y_tr)
    return rf


# ============================================================
# 6) SINGLE-OUTPUT SVGP
# ============================================================
class SingleSVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, kernel='rbf'):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()

        d = inducing_points.size(-1)
        if kernel == 'rbf':
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=d)
        elif kernel == 'matern15':
            base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=d)
        elif kernel == 'matern25':
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d)
        else:
            raise ValueError(f"unknown kernel {kernel}")
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


class SingleSVGPWrapper:
    def __init__(self, model, likelihood, device):
        self.model = model
        self.likelihood = likelihood
        self.device = device

    def predict_np(self, X, batch_size=8192):
        self.model.eval(); self.likelihood.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        out = np.empty(X.shape[0], dtype=np.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, X.shape[0], batch_size):
                pred = self.likelihood(self.model(X_t[i:i+batch_size]))
                out[i:i+batch_size] = pred.mean.cpu().numpy()
        return out

    def predict_np_with_std(self, X, batch_size=8192):
        self.model.eval(); self.likelihood.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        mu  = np.empty(X.shape[0], dtype=np.float32)
        std = np.empty(X.shape[0], dtype=np.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, X.shape[0], batch_size):
                pred = self.likelihood(self.model(X_t[i:i+batch_size]))
                mu [i:i+batch_size] = pred.mean.cpu().numpy()
                std[i:i+batch_size] = pred.stddev.cpu().numpy()
        return mu, std

    def predict_np_decomposed(self, X, batch_size=8192):
        self.model.eval(); self.likelihood.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        mu       = np.empty(X.shape[0], dtype=np.float32)
        std_epi  = np.empty(X.shape[0], dtype=np.float32)
        std_tot  = np.empty(X.shape[0], dtype=np.float32)
        noise_var = float(self.likelihood.noise.detach().cpu().numpy().item())
        std_alea_scalar = float(np.sqrt(noise_var))

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, X.shape[0], batch_size):
                xb = X_t[i:i+batch_size]
                latent = self.model(xb)
                mu     [i:i+batch_size] = latent.mean.cpu().numpy()
                std_epi[i:i+batch_size] = latent.stddev.cpu().numpy()
                pred = self.likelihood(latent)
                std_tot[i:i+batch_size] = pred.stddev.cpu().numpy()

        std_alea = np.full_like(std_epi, std_alea_scalar)
        return mu, std_epi, std_alea, std_tot


def train_single_svgp(X_tr, y_tr_1d, n_inducing, kernel='rbf',
                      label='', verbose=True, n_epochs_override=None):
    n, d = X_tr.shape
    n_ind = min(n_inducing, n)
    t0    = time.time()
    n_ep  = n_epochs_override if n_epochs_override is not None else GP_EPOCHS

    km = MiniBatchKMeans(n_clusters=n_ind, batch_size=10000,
                         random_state=0, n_init=3).fit(X_tr)
    inducing = torch.tensor(km.cluster_centers_, dtype=torch.float32)

    model      = SingleSVGP(inducing, kernel=kernel).to(DEVICE)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
    mll        = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n)
    optimizer  = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=GP_LR)

    X_t = torch.tensor(X_tr,    dtype=torch.float32)
    y_t = torch.tensor(y_tr_1d, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=GP_BATCH_SIZE,
                        shuffle=True, num_workers=0,
                        pin_memory=(DEVICE.type == 'cuda'))

    model.train(); likelihood.train()
    for epoch in range(n_ep):
        ep_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            optimizer.zero_grad()
            loss = -mll(model(xb), yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * xb.size(0)
        if verbose and (epoch % 10 == 0 or epoch == n_ep - 1):
            print(f"    [{label}] epoch {epoch:3d}  loss={ep_loss/n:.4f}")
        if should_checkpoint():
            raise _CheckpointSignal()

    print(f"  [{label}] trained in {time.time()-t0:.1f}s "
          f"({n_ind} inducing, kernel={kernel})")
    return SingleSVGPWrapper(model, likelihood, DEVICE)


# ============================================================
# 7) CV runner — RF
# ============================================================
def cv_rf(fold_indices, X, y_log, lat, lon,   # lat/lon needed for inner spatial CV
          start_fold=0, _skill_init=None, _params_init=None):
    """
    Outer buffered spatial CV for RF (fold_indices shared with cv_svgp_two()
    and cv_ridge()).
    With TUNE_EVERY_FOLD, rf_tune() runs INSIDE EVERY outer fold on that
    fold's training partition only (shared _inner_folds() split — by default a
    single 1-of-5 buffered spatial holdout), so every outer score is a genuine
    nested-CV score. The configuration reported for the paper and used for the
    final all-data model is the MODE across folds.

    Returns (corr, corr_log, spear, rmse, modal_params, params_per_fold).
    """
    k = len(fold_indices)
    skill_corr     = (_skill_init['corr']     if _skill_init
                      else np.zeros((k, 2), dtype=np.float32))
    skill_corr_log = (_skill_init['corr_log'] if _skill_init
                      else np.zeros((k, 2), dtype=np.float32))
    skill_spear    = (_skill_init['spear']    if _skill_init
                      else np.zeros((k, 2), dtype=np.float32))
    skill_rmse     = (_skill_init['rmse']     if _skill_init
                      else np.zeros((k, 2), dtype=np.float32))
    # RMSE in LOG-TARGET UNITS. This is the metric the models are actually
    # selected on (inner MSE is in log units), and it is not dominated by the
    # few largest biovolume points the way original-units RMSE is. `.get` so a
    # checkpoint written before this array existed still restores.
    skill_rmse_log = ((_skill_init.get('rmse_log')
                       if _skill_init.get('rmse_log') is not None
                       else np.zeros((k, 2), dtype=np.float32)) if _skill_init
                      else np.zeros((k, 2), dtype=np.float32))
    params_per_fold = _as_per_fold(_params_init, k)

    for fold, (tr_idx, va_idx) in enumerate(fold_indices):
        if fold < start_fold:
            print(f"[ckpt] RF fold {fold+1} already done — skipping", flush=True)
            continue

        t0 = time.time()
        print(f"\n[RF] fold {fold+1}/{k}  "
              f"train={len(tr_idx):,}  valid={len(va_idx):,}")

        sx = StandardScaler().fit(X[tr_idx])
        sy = StandardScaler().fit(y_log[tr_idx])
        Xt = sx.transform(X[tr_idx]).astype(np.float32)
        Xv = sx.transform(X[va_idx]).astype(np.float32)
        yt = sy.transform(y_log[tr_idx]).astype(np.float32)
        yv = sy.transform(y_log[va_idx]).astype(np.float32)

        # ── inner selection on THIS fold's training partition only ───────────
        # Raw X / log-targets: rf_tune refits both scalers inside each inner
        # split, exactly as svgp_tune() and ridge_tune() do.
        if TUNE_EVERY_FOLD or params_per_fold[TUNE_ON_FOLD] is None:
            params_per_fold[fold] = rf_tune(
                X[tr_idx], y_log[tr_idx],
                lat_tr=lat[tr_idx],
                lon_tr=lon[tr_idx],
            )
        else:
            params_per_fold[fold] = params_per_fold[TUNE_ON_FOLD]
        best_params = params_per_fold[fold]
        model = rf_train(Xt, yt, params=best_params)

        pred_s   = model.predict(Xv)
        pred_log = sy.inverse_transform(pred_s)
        y_va_log = sy.inverse_transform(yv)
        pred_orig  = np.maximum(np.exp(pred_log) - EPS, 0.0)
        y_va_orig  = np.maximum(np.exp(y_va_log) - EPS, 0.0)

        for q in range(2):
            skill_corr[fold, q]     = np.corrcoef(y_va_orig[:, q], pred_orig[:, q])[0, 1]
            skill_corr_log[fold, q] = np.corrcoef(y_va_log[:, q], pred_log[:, q])[0, 1]
            skill_spear[fold, q]    = spearmanr(y_va_orig[:, q], pred_orig[:, q]).correlation
            skill_rmse[fold, q]     = np.sqrt(mean_squared_error(y_va_orig[:, q],
                                                                  pred_orig[:, q]))
            skill_rmse_log[fold, q] = np.sqrt(mean_squared_error(y_va_log[:, q],
                                                                 pred_log[:, q]))
            n_plot = min(5000, len(pred_orig))
            idx = np.random.RandomState(fold*10+q).choice(len(pred_orig),
                                                          n_plot, replace=False)
            plt.figure(figsize=(5, 5))
            plt.scatter(y_va_orig[idx, q], pred_orig[idx, q],
                        s=4, alpha=0.3, edgecolors='none')
            lo, hi = y_va_orig[idx, q].min(), y_va_orig[idx, q].max()
            plt.plot([lo, hi], [lo, hi], 'r--', lw=1.5)
            plt.xlabel('Observed'); plt.ylabel('Predicted')
            plt.title(f"RF fold {fold} {TARGET_NAMES[q]}\n"
                      f"r={skill_corr[fold, q]:.2f}  "
                      f"RMSE={skill_rmse[fold, q]:.3f}")
            plt.grid(alpha=0.3)
            plt.savefig(f"{SCATTER_DIR}/RF_fold{fold}_{TARGET_NAMES[q]}.png",
                        dpi=120, bbox_inches='tight')
            plt.close()

        print(f"  r(orig): beta={skill_corr[fold, 0]:.3f}  "
              f"biovol={skill_corr[fold, 1]:.3f}")
        print(f"  r(log):  beta={skill_corr_log[fold, 0]:.3f}  "
              f"biovol={skill_corr_log[fold, 1]:.3f}")
        print(f"  rho:     beta={skill_spear[fold, 0]:.3f}  "
              f"biovol={skill_spear[fold, 1]:.3f}")
        print(f"  RMSE:    beta={skill_rmse[fold, 0]:.3f}  "
              f"biovol={skill_rmse[fold, 1]:.3f}  (original units)")
        print(f"  RMSElog: beta={skill_rmse_log[fold, 0]:.3f}  "
              f"biovol={skill_rmse_log[fold, 1]:.3f}  ({time.time()-t0:.1f}s)")
        del Xt, Xv, yt, yv, model, pred_s, pred_log, pred_orig
        gc.collect()

        # ── checkpoint after every RF fold ───────────────────────────────────
        # Merge into the existing checkpoint rather than replacing it: a bare
        # ckpt_save({...}) here would drop svgp_params_per_fold and svgp_skill,
        # forcing the SVGP to re-tune and restart from fold 0.
        _cs = ckpt_load() or {}
        _cs.update({
            'stage': 'rf_running', 'fold': fold + 1,
            'rf_skill': {'corr': skill_corr, 'corr_log': skill_corr_log,
                         'spear': skill_spear, 'rmse': skill_rmse,
                         'rmse_log': skill_rmse_log},
            'rf_params_per_fold': params_per_fold,
        })
        ckpt_save(_cs)
        if should_checkpoint():
            print(f"[ckpt] Time limit — requeueing after RF fold {fold+1}",
                  flush=True)
            requeue_and_exit()

    print(f"\n[RF] mean r(orig): beta={skill_corr[:, 0].mean():.3f}  "
          f"biovol={skill_corr[:, 1].mean():.3f}")
    print(f"[RF] mean r(log):  beta={skill_corr_log[:, 0].mean():.3f}  "
          f"biovol={skill_corr_log[:, 1].mean():.3f}")
    print(f"[RF] mean rho:     beta={skill_spear[:, 0].mean():.3f}  "
          f"biovol={skill_spear[:, 1].mean():.3f}")
    print(f"[RF] mean RMSE:    beta={skill_rmse[:, 0].mean():.3f}  "
          f"biovol={skill_rmse[:, 1].mean():.3f}  (original units)")
    print(f"[RF] mean RMSElog: beta={skill_rmse_log[:, 0].mean():.3f}  "
          f"biovol={skill_rmse_log[:, 1].mean():.3f}  <- the selection metric")
    modal_params = modal_config(params_per_fold, label='RF')
    return (skill_corr, skill_corr_log, skill_spear, skill_rmse, skill_rmse_log,
            modal_params, params_per_fold)


# ============================================================
# 8) CV runner — SVGP
# ============================================================
def cv_svgp_two(fold_indices, X, y_log, lat, lon,   # lat/lon for inner spatial CV
                start_fold=0, _skill_init=None, _rf_state=None,
                _svgp_best_params_init=None, _tune_progress_init=None):
    """
    Outer buffered spatial CV for SVGP (same fold_indices as cv_rf() and
    cv_ridge()).
    With TUNE_EVERY_FOLD, svgp_tune() runs INSIDE EVERY outer fold, per
    target, searching SVGP_PARAM_GRID on the shared _inner_folds() split of
    that fold's training partition — a full nested CV symmetric with cv_rf()
    and cv_ridge(). The architecture reported for the paper and used for the
    final all-data models is the MODE across folds, per target.

    Returns (corr, corr_log, spear, rmse, modal_params_per_target,
             params_per_fold) where params_per_fold[fold] = [cfg_beta,
             cfg_biovol].
    """
    k = len(fold_indices)
    skill_corr     = (_skill_init['corr']     if _skill_init
                      else np.zeros((k, 2), dtype=np.float32))
    skill_corr_log = (_skill_init['corr_log'] if _skill_init
                      else np.zeros((k, 2), dtype=np.float32))
    skill_spear    = (_skill_init['spear']    if _skill_init
                      else np.zeros((k, 2), dtype=np.float32))
    skill_rmse     = (_skill_init['rmse']     if _skill_init
                      else np.zeros((k, 2), dtype=np.float32))
    skill_rmse_log = ((_skill_init.get('rmse_log')
                       if _skill_init.get('rmse_log') is not None
                       else np.zeros((k, 2), dtype=np.float32)) if _skill_init
                      else np.zeros((k, 2), dtype=np.float32))

    # params_per_fold[fold] = [cfg_beta, cfg_biovol]; folds already completed
    # before a requeue are restored from the checkpoint.
    params_per_fold = _as_per_fold(_svgp_best_params_init, k, width=2)

    # tune_progress[(fold, q)] = {cfg_idx: inner_MSE} for configs already scored
    # in a PREVIOUS job. svgp_tune() can be interrupted by _CheckpointSignal
    # part-way through its grid, and without this the fold would restart tuning
    # from scratch on every requeue.
    tune_progress = (dict(_tune_progress_init)
                     if isinstance(_tune_progress_init, dict) else {})

    targets = [('beta', 0), ('biovol', 1)]

    for fold, (tr_idx, va_idx) in enumerate(fold_indices):
        if fold < start_fold:
            print(f"[ckpt] SVGP fold {fold+1} already done — skipping",
                  flush=True)
            continue

        t0 = time.time()
        print(f"\n[SVGP] fold {fold+1}/{k}  "
              f"train={len(tr_idx):,}  valid={len(va_idx):,}")

        sx = StandardScaler().fit(X[tr_idx])
        Xt = sx.transform(X[tr_idx]).astype(np.float32)
        Xv = sx.transform(X[va_idx]).astype(np.float32)

        for name, q in targets:
            # ── inner selection on THIS fold's training partition only ──────
            if TUNE_EVERY_FOLD or params_per_fold[TUNE_ON_FOLD][q] is None:
                # train_single_svgp() raises _CheckpointSignal when the wall
                # clock runs down, and svgp_tune() re-raises it. That call sits
                # OUTSIDE the handler below, so before this try/except the
                # signal escaped to module scope and killed the job instead of
                # requeueing it. Harmless while tuning only ever ran on fold 0
                # at the start of the stage; with TUNE_EVERY_FOLD it runs in
                # every fold, including ones that start near the limit.
                _tp = tune_progress.setdefault((fold, q), {})

                def _save_cfg(cfg_idx, mse, _tp=_tp):
                    _tp[cfg_idx] = mse

                try:
                    params_per_fold[fold][q] = svgp_tune(
                        X[tr_idx], y_log[tr_idx],
                        lat_tr=lat[tr_idx],
                        lon_tr=lon[tr_idx],
                        target_idx=q,
                        target_name=name,
                        _resume=_tp,
                        _on_config=_save_cfg,
                    )
                except _CheckpointSignal:
                    print(f"  [ckpt] signal during SVGP tuning, fold {fold}, "
                          f"target {name}: {len(_tp)}/{len(SVGP_PARAM_GRID)} "
                          f"configs scored and saved", flush=True)
                    ckpt_save({
                        'stage': 'svgp_running', 'fold': fold,
                        'svgp_skill': {'corr': skill_corr,
                                       'corr_log': skill_corr_log,
                                       'spear': skill_spear, 'rmse': skill_rmse,
                         'rmse_log': skill_rmse_log},
                        'svgp_params_per_fold': params_per_fold,
                        'svgp_tune_progress': tune_progress,
                        'rf_state': _rf_state,
                    })
                    requeue_and_exit()
                # tuning for this (fold, target) is finished; drop its progress
                # so the checkpoint does not carry stale grid scores
                tune_progress.pop((fold, q), None)
            else:
                params_per_fold[fold][q] = params_per_fold[TUNE_ON_FOLD][q]

            cfg   = params_per_fold[fold][q]
            n_ind = cfg['n_inducing']
            kern  = cfg['kernel']

            y_tr_col = y_log[tr_idx][:, [q]]
            y_va_col = y_log[va_idx][:, [q]]
            sy_q = StandardScaler().fit(y_tr_col)
            yt_q = sy_q.transform(y_tr_col).ravel().astype(np.float32)
            yv_q = sy_q.transform(y_va_col).ravel().astype(np.float32)

            try:
                wrap = train_single_svgp(Xt, yt_q, n_ind, kernel=kern,
                                         label=name, verbose=True)
            except _CheckpointSignal:
                # Signal fired mid-epoch — save and requeue; repeat this fold
                ckpt_save({
                    'stage': 'svgp_running', 'fold': fold,
                    'svgp_skill': {'corr': skill_corr, 'corr_log': skill_corr_log,
                                   'spear': skill_spear, 'rmse': skill_rmse,
                         'rmse_log': skill_rmse_log},
                    'svgp_params_per_fold': params_per_fold,   # persist tuning
                    'svgp_tune_progress': tune_progress,
                    'rf_state': _rf_state,
                })
                requeue_and_exit()

            pred_s     = wrap.predict_np(Xv).reshape(-1, 1)
            pred_log   = sy_q.inverse_transform(pred_s).ravel()
            y_va_log_q = sy_q.inverse_transform(yv_q.reshape(-1, 1)).ravel()
            pred_orig  = np.maximum(np.exp(pred_log)   - EPS, 0.0)
            y_va_orig  = np.maximum(np.exp(y_va_log_q) - EPS, 0.0)

            skill_corr[fold, q]     = np.corrcoef(y_va_orig, pred_orig)[0, 1]
            skill_corr_log[fold, q] = np.corrcoef(y_va_log_q, pred_log)[0, 1]
            skill_spear[fold, q]    = spearmanr(y_va_orig, pred_orig).correlation
            skill_rmse[fold, q]     = np.sqrt(mean_squared_error(y_va_orig, pred_orig))
            skill_rmse_log[fold, q] = np.sqrt(mean_squared_error(y_va_log_q, pred_log))

            n_plot = min(5000, len(pred_orig))
            idx = np.random.RandomState(fold*10+q).choice(len(pred_orig),
                                                          n_plot, replace=False)
            plt.figure(figsize=(5, 5))
            plt.scatter(y_va_orig[idx], pred_orig[idx],
                        s=4, alpha=0.3, edgecolors='none')
            lo, hi = float(y_va_orig[idx].min()), float(y_va_orig[idx].max())
            plt.plot([lo, hi], [lo, hi], 'r--', lw=1.5)
            plt.xlabel('Observed'); plt.ylabel('Predicted')
            plt.title(f"SVGP fold {fold} {TARGET_NAMES[q]} "
                      f"({n_ind} ind, {kern})\n"
                      f"r={skill_corr[fold, q]:.2f}  "
                      f"RMSE={skill_rmse[fold, q]:.3f}")
            plt.grid(alpha=0.3)
            plt.savefig(f"{SCATTER_DIR}/SVGP_fold{fold}_{TARGET_NAMES[q]}.png",
                        dpi=120, bbox_inches='tight')
            plt.close()

            del wrap, pred_s, pred_log, pred_orig, y_va_orig
            del y_tr_col, y_va_col, yt_q, yv_q
            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

        print(f"  r(orig): beta={skill_corr[fold, 0]:.3f}  "
              f"biovol={skill_corr[fold, 1]:.3f}")
        print(f"  r(log):  beta={skill_corr_log[fold, 0]:.3f}  "
              f"biovol={skill_corr_log[fold, 1]:.3f}")
        print(f"  rho:     beta={skill_spear[fold, 0]:.3f}  "
              f"biovol={skill_spear[fold, 1]:.3f}")
        print(f"  RMSE:    beta={skill_rmse[fold, 0]:.3f}  "
              f"biovol={skill_rmse[fold, 1]:.3f}  (original units)")
        print(f"  RMSElog: beta={skill_rmse_log[fold, 0]:.3f}  "
              f"biovol={skill_rmse_log[fold, 1]:.3f}  ({time.time()-t0:.1f}s)")
        del Xt, Xv
        gc.collect()

        # ── checkpoint after every SVGP fold ─────────────────────────────────
        ckpt_save({
            'stage': 'svgp_running', 'fold': fold + 1,
            'svgp_skill': {'corr': skill_corr, 'corr_log': skill_corr_log,
                           'spear': skill_spear, 'rmse': skill_rmse,
                         'rmse_log': skill_rmse_log},
            'svgp_params_per_fold': params_per_fold,   # persist across requeues
            'svgp_tune_progress': tune_progress,
            'rf_state': _rf_state,
        })
        if should_checkpoint():
            print(f"[ckpt] Time limit — requeueing after SVGP fold {fold+1}",
                  flush=True)
            requeue_and_exit()

    print(f"\n[SVGP] mean r(orig): beta={skill_corr[:, 0].mean():.3f}  "
          f"biovol={skill_corr[:, 1].mean():.3f}")
    print(f"[SVGP] mean r(log):  beta={skill_corr_log[:, 0].mean():.3f}  "
          f"biovol={skill_corr_log[:, 1].mean():.3f}")
    print(f"[SVGP] mean rho:     beta={skill_spear[:, 0].mean():.3f}  "
          f"biovol={skill_spear[:, 1].mean():.3f}")
    print(f"[SVGP] mean RMSE:    beta={skill_rmse[:, 0].mean():.3f}  "
          f"biovol={skill_rmse[:, 1].mean():.3f}  (original units)")
    print(f"[SVGP] mean RMSElog: beta={skill_rmse_log[:, 0].mean():.3f}  "
          f"biovol={skill_rmse_log[:, 1].mean():.3f}  <- the selection metric")
    modal_per_target = [
        modal_config([pf[q] if pf is not None else None for pf in params_per_fold],
                     label=f"SVGP {name}")
        for name, q in targets
    ]
    return (skill_corr, skill_corr_log, skill_spear, skill_rmse, skill_rmse_log,
            modal_per_target, params_per_fold)


# ============================================================
# 8b) CV runner — RIDGE REGRESSION baseline
# ============================================================
def ridge_tune(X_tr, y_log_tr, lat_tr, lon_tr, target_idx, target_name):
    """
    Grid search over RIDGE_ALPHA_GRID on the shared inner split(s) from
    _inner_folds() — protocol-identical to rf_tune() and svgp_tune().

    Same splits, same per-split refitting of the X and y scalers, same
    criterion (MSE in log-target units), same INNER_TRAIN_SAMPLE row cap.
    Returns the best alpha.
    """
    inner_folds = _inner_folds(lat_tr, lon_tr)

    print(f"  [Ridge tune] target={target_name}  "
          f"grid={len(RIDGE_ALPHA_GRID)} alphas × {INNER_DESC}", flush=True)

    y_col = y_log_tr[:, target_idx]
    # mse[alpha, fold]; scalers are fit once per fold and reused across alphas
    mse = np.full((len(RIDGE_ALPHA_GRID), len(inner_folds)), np.nan)

    for i_fold, (i_tr, i_va) in enumerate(inner_folds):
        # same row cap and same helper as RF/SVGP, so all three select on an
        # equal number of training rows
        i_tr = _subsample_train_split(i_tr, RIDGE_TUNE_SAMPLE, seed=i_fold)

        sx_i = StandardScaler().fit(X_tr[i_tr])
        Xt_i = sx_i.transform(X_tr[i_tr]).astype(np.float32)
        Xv_i = sx_i.transform(X_tr[i_va]).astype(np.float32)

        sy_i = StandardScaler().fit(y_col[i_tr].reshape(-1, 1))
        yt_i = sy_i.transform(y_col[i_tr].reshape(-1, 1)).ravel().astype(np.float32)

        for a_idx, alpha in enumerate(RIDGE_ALPHA_GRID):
            m = Ridge(alpha=alpha, random_state=10)
            m.fit(Xt_i, yt_i)
            pred_log = sy_i.inverse_transform(
                m.predict(Xv_i).reshape(-1, 1)).ravel()
            mse[a_idx, i_fold] = float(mean_squared_error(y_col[i_va], pred_log))
            del m

        del sx_i, sy_i, Xt_i, Xv_i, yt_i
        gc.collect()

    for a_idx, alpha in enumerate(RIDGE_ALPHA_GRID):
        print(f"    alpha={alpha:>10.4g}  inner_MSE={mse[a_idx].mean():.4f}  "
              f"(folds: {[f'{v:.4f}' for v in mse[a_idx]]})", flush=True)

    best_idx   = int(np.argmin(mse.mean(axis=1)))
    best_alpha = RIDGE_ALPHA_GRID[best_idx]
    best_mse   = float(mse[best_idx].mean())

    print(f"  [Ridge tune] best for {target_name}: alpha={best_alpha:.4g}  "
          f"(MSE={best_mse:.4f})", flush=True)
    return best_alpha


def cv_ridge(fold_indices, X, y_log, lat, lon, _alphas_init=None):
    """
    Outer buffered spatial CV for the Ridge baseline — same fold_indices as
    cv_rf() and cv_svgp_two().

    With TUNE_EVERY_FOLD, ridge_tune() selects alpha per target INSIDE EVERY
    outer fold on that fold's training partition only (shared _inner_folds()
    split), a full nested CV symmetric with the other two models. The alpha
    reported for the paper is the MODE across folds, per target.

    Returns (corr, corr_log, spear, rmse, modal_alphas, alphas_per_fold).
    """
    k = len(fold_indices)
    skill_corr     = np.zeros((k, 2), dtype=np.float32)
    skill_corr_log = np.zeros((k, 2), dtype=np.float32)
    skill_spear    = np.zeros((k, 2), dtype=np.float32)
    skill_rmse     = np.zeros((k, 2), dtype=np.float32)
    skill_rmse_log = np.zeros((k, 2), dtype=np.float32)

    # alphas_per_fold[fold] = [alpha_beta, alpha_biovol]
    alphas_per_fold = _as_per_fold(_alphas_init, k, width=2)

    for fold, (tr_idx, va_idx) in enumerate(fold_indices):
        t0 = time.time()
        print(f"\n[Ridge] fold {fold+1}/{k}  "
              f"train={len(tr_idx):,}  valid={len(va_idx):,}")

        sx = StandardScaler().fit(X[tr_idx])
        Xt = sx.transform(X[tr_idx]).astype(np.float32)
        Xv = sx.transform(X[va_idx]).astype(np.float32)

        for q, tname in enumerate(['beta', 'biovol']):
            # ── inner selection on THIS fold's training partition only ──────
            if TUNE_EVERY_FOLD or alphas_per_fold[TUNE_ON_FOLD][q] is None:
                alphas_per_fold[fold][q] = ridge_tune(
                    X[tr_idx], y_log[tr_idx],
                    lat_tr=lat[tr_idx],
                    lon_tr=lon[tr_idx],
                    target_idx=q,
                    target_name=tname,
                )
            else:
                alphas_per_fold[fold][q] = alphas_per_fold[TUNE_ON_FOLD][q]
            alpha = alphas_per_fold[fold][q]

            y_tr_col = y_log[tr_idx][:, [q]]
            y_va_col = y_log[va_idx][:, [q]]
            sy_q = StandardScaler().fit(y_tr_col)
            yt_q = sy_q.transform(y_tr_col).ravel().astype(np.float32)
            yv_q = sy_q.transform(y_va_col).ravel().astype(np.float32)

            model  = Ridge(alpha=alpha, random_state=10)
            model.fit(Xt, yt_q)

            pred_s     = model.predict(Xv).reshape(-1, 1)
            pred_log   = sy_q.inverse_transform(pred_s).ravel()
            y_va_log_q = sy_q.inverse_transform(yv_q.reshape(-1, 1)).ravel()
            pred_orig  = np.maximum(np.exp(pred_log)   - EPS, 0.0)
            y_va_orig  = np.maximum(np.exp(y_va_log_q) - EPS, 0.0)

            skill_corr[fold, q]     = np.corrcoef(y_va_orig, pred_orig)[0, 1]
            skill_corr_log[fold, q] = np.corrcoef(y_va_log_q, pred_log)[0, 1]
            skill_spear[fold, q]    = spearmanr(y_va_orig, pred_orig).correlation
            skill_rmse[fold, q]     = np.sqrt(mean_squared_error(y_va_orig, pred_orig))
            skill_rmse_log[fold, q] = np.sqrt(mean_squared_error(y_va_log_q, pred_log))

            n_plot = min(5000, len(pred_orig))
            idx = np.random.RandomState(fold*10+q).choice(len(pred_orig),
                                                          n_plot, replace=False)
            plt.figure(figsize=(5, 5))
            plt.scatter(y_va_orig[idx], pred_orig[idx],
                        s=4, alpha=0.3, edgecolors='none', color='#7E57C2')
            lo, hi = float(y_va_orig[idx].min()), float(y_va_orig[idx].max())
            plt.plot([lo, hi], [lo, hi], 'r--', lw=1.5)
            plt.xlabel('Observed'); plt.ylabel('Predicted')
            plt.title(f"Ridge fold {fold} {TARGET_NAMES[q]} "
                      f"(alpha={alpha:.4g})\n"
                      f"r={skill_corr[fold, q]:.2f}  "
                      f"RMSE={skill_rmse[fold, q]:.3f}")
            plt.grid(alpha=0.3)
            plt.savefig(f"{SCATTER_DIR}/Ridge_fold{fold}_"
                        f"{TARGET_NAMES[q]}.png",
                        dpi=120, bbox_inches='tight')
            plt.close()

            del y_tr_col, y_va_col, yt_q, yv_q
            del model, pred_s, pred_log, pred_orig, y_va_orig
            gc.collect()

        print(f"  r(orig): beta={skill_corr[fold, 0]:.3f}  "
              f"biovol={skill_corr[fold, 1]:.3f}")
        print(f"  r(log):  beta={skill_corr_log[fold, 0]:.3f}  "
              f"biovol={skill_corr_log[fold, 1]:.3f}")
        print(f"  rho:     beta={skill_spear[fold, 0]:.3f}  "
              f"biovol={skill_spear[fold, 1]:.3f}")
        print(f"  RMSE:    beta={skill_rmse[fold, 0]:.3f}  "
              f"biovol={skill_rmse[fold, 1]:.3f}  (original units)")
        print(f"  RMSElog: beta={skill_rmse_log[fold, 0]:.3f}  "
              f"biovol={skill_rmse_log[fold, 1]:.3f}  ({time.time()-t0:.1f}s)")
        del Xt, Xv
        gc.collect()

    print(f"\n[Ridge] mean r(orig): beta={skill_corr[:, 0].mean():.3f}  "
          f"biovol={skill_corr[:, 1].mean():.3f}")
    print(f"[Ridge] mean r(log):  beta={skill_corr_log[:, 0].mean():.3f}  "
          f"biovol={skill_corr_log[:, 1].mean():.3f}")
    print(f"[Ridge] mean rho:     beta={skill_spear[:, 0].mean():.3f}  "
          f"biovol={skill_spear[:, 1].mean():.3f}")
    print(f"[Ridge] mean RMSE:    beta={skill_rmse[:, 0].mean():.3f}  "
          f"biovol={skill_rmse[:, 1].mean():.3f}  (original units)")
    print(f"[Ridge] mean RMSElog: beta={skill_rmse_log[:, 0].mean():.3f}  "
          f"biovol={skill_rmse_log[:, 1].mean():.3f}  <- the selection metric")
    modal_alphas = [
        modal_config([af[q] if af is not None else None for af in alphas_per_fold],
                     label=f"Ridge {tname}")
        for q, tname in enumerate(['beta', 'biovol'])
    ]
    return (skill_corr, skill_corr_log, skill_spear, skill_rmse, skill_rmse_log,
            modal_alphas, alphas_per_fold)


# ============================================================
# 9) Run CV  (checkpoint-aware)
# ============================================================
_state = ckpt_load()

# ── RF CV ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RANDOM FOREST  (nested CV)")
_n_rf_cand = 3 * 3 * 3 * 3 * 2      # n_est x depth x split x leaf x max_features
print(f"  hyperparameter selection: "
      f"{'every outer fold' if TUNE_EVERY_FOLD else f'outer fold {TUNE_ON_FOLD} only'}")
print(f"  inner sel.: {INNER_DESC} "
      f"(BLOCK={BLOCK_DEG}°, BUFFER={BUFFER_KM:.0f} km edge-to-edge, seed={INNER_SEED})")
print(f"  budget: {_n_rf_cand} candidates x {INNER_N_USE} inner split(s) x "
      f"{K_FOLDS if TUNE_EVERY_FOLD else 1} outer fold(s) = "
      f"{_n_rf_cand * INNER_N_USE * (K_FOLDS if TUNE_EVERY_FOLD else 1):,} inner fits")
print("="*60)

if (_state and _state.get('stage') in ('rf_done', 'svgp_running', 'svgp_done')
        and not FORCE_RERUN_RF):
    print("[ckpt] RF CV already complete — restoring from checkpoint")
    rf_corr        = _state['rf_state']['corr']
    rf_corr_log    = _state['rf_state']['corr_log']
    rf_spear       = _state['rf_state']['spear']
    rf_rmse        = _state['rf_state']['rmse']
    rf_rmse_log    = _state['rf_state'].get('rmse_log', np.zeros_like(rf_rmse))
    rf_best_params = _state['rf_state']['best_params']
    rf_params_per_fold = _state['rf_state'].get('params_per_fold',
                                                [None] * K_FOLDS)
else:
    if FORCE_RERUN_RF:
        print("[ckpt] FORCE_RERUN_RF set — retraining RF; "
              "SVGP progress in the checkpoint is preserved", flush=True)
    _start_rf   = _state.get('fold', 0)        if _state and _state.get('stage') == 'rf_running' else 0
    _skill_rf   = _state.get('rf_skill')       if _state and _state.get('stage') == 'rf_running' else None
    _params_rf  = _state.get('rf_params_per_fold') if _state and _state.get('stage') == 'rf_running' else None
    (rf_corr, rf_corr_log, rf_spear, rf_rmse, rf_rmse_log,
     rf_best_params, rf_params_per_fold) = cv_rf(
        fold_indices, X, y_log,
        lat=lat_all, lon=lon_all,       # needed for inner buffered spatial CV
        start_fold=_start_rf,
        _skill_init=_skill_rf,
        _params_init=_params_rf,
    )
    _rf_state = {'corr': rf_corr, 'corr_log': rf_corr_log,
                 'spear': rf_spear, 'rmse': rf_rmse, 'rmse_log': rf_rmse_log,
                 'best_params': rf_best_params,
                 'params_per_fold': rf_params_per_fold}
    # Merge, and do not rewind 'stage' if the SVGP has already started —
    # otherwise the SVGP branch below would restart from fold 0. Note _state
    # was read once at the top of this section and is deliberately not
    # refreshed, so the SVGP branch still sees its original stage/fold.
    _cs = ckpt_load() or {}
    _cs['rf_state'] = _rf_state
    if _cs.get('stage') not in ('svgp_running', 'svgp_done'):
        _cs['stage'], _cs['fold'] = 'rf_done', K_FOLDS
    ckpt_save(_cs)

io.savemat(os.path.join(OUTPUT_DIR, 'skill_rf_global.mat'), {
    'skill_corr':     rf_corr, 'skill_rmse': rf_rmse,
    'skill_rmse_log': rf_rmse_log,      # <- plot THIS; same space as selection
    'skill_corr_log': rf_corr_log, 'skill_spearman': rf_spear,
    'feature_names': np.array(FEATURE_NAMES, dtype=object),
    # hyperparameters selected independently in each outer fold, plus the mode
    'params_per_fold': np.array([str(p) for p in rf_params_per_fold],
                                dtype=object),
    'params_modal':    str(rf_best_params),
    'cv_block_deg':  BLOCK_DEG, 'cv_buffer_km': BUFFER_KM,
})

# ── SVGP CV ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SPARSE VARIATIONAL GP  (nested CV)")
print(f"  inner grid: {len(SVGP_PARAM_GRID)} configs "
      f"(inducing ∈ {sorted({c['n_inducing'] for c in SVGP_PARAM_GRID})}, "
      f"kernel ∈ {sorted({c['kernel'] for c in SVGP_PARAM_GRID})})")
print(f"  inner sel.: {INNER_DESC} "
      f"(BLOCK={BLOCK_DEG}°, BUFFER={BUFFER_KM:.0f} km edge-to-edge, seed={INNER_SEED})")
print(f"  inner epochs: {GP_TUNE_EPOCHS}  (full run uses {GP_EPOCHS})")
print(f"  hyperparameter selection: "
      f"{'every outer fold' if TUNE_EVERY_FOLD else f'outer fold {TUNE_ON_FOLD} only'}")
_n_gp_fit = (len(SVGP_PARAM_GRID) * INNER_N_USE * 2
             * (K_FOLDS if TUNE_EVERY_FOLD else 1))
print(f"  budget: {len(SVGP_PARAM_GRID)} configs x {INNER_N_USE} inner split(s) "
      f"x 2 targets x {K_FOLDS if TUNE_EVERY_FOLD else 1} outer fold(s) = "
      f"{_n_gp_fit:,} SVGP fits at {GP_TUNE_EPOCHS} epochs each")
print(f"          plus {2 * K_FOLDS} full-length outer fits at {GP_EPOCHS} "
      f"epochs. This is the dominant cost of the whole script; if it overruns "
      f"the queue, halve SVGP_PARAM_GRID before touching K_FOLDS.")
print("="*60)

_rf_state_carry = {'corr': rf_corr, 'corr_log': rf_corr_log,
                   'spear': rf_spear, 'rmse': rf_rmse, 'rmse_log': rf_rmse_log,
                   'best_params': rf_best_params,
                   'params_per_fold': rf_params_per_fold}

if _state and _state.get('stage') == 'svgp_done':
    print("[ckpt] SVGP CV already complete — restoring from checkpoint")
    svgp_corr          = _state['svgp_skill']['corr']
    svgp_corr_log      = _state['svgp_skill']['corr_log']
    svgp_spear         = _state['svgp_skill']['spear']
    svgp_rmse          = _state['svgp_skill']['rmse']
    svgp_rmse_log      = _state['svgp_skill'].get('rmse_log',
                                                  np.zeros_like(svgp_rmse))
    svgp_best_params   = _state.get('svgp_best_params', [None, None])
    svgp_params_per_fold = _state.get('svgp_params_per_fold',
                                      [[None, None]] * K_FOLDS)
else:
    _start_svgp        = _state.get('fold', 0)          if _state and _state.get('stage') == 'svgp_running' else 0
    _skill_svgp        = _state.get('svgp_skill')       if _state and _state.get('stage') == 'svgp_running' else None
    _svgp_best_params  = _state.get('svgp_params_per_fold') if _state and _state.get('stage') == 'svgp_running' else None
    _svgp_tune_prog    = _state.get('svgp_tune_progress') if _state and _state.get('stage') == 'svgp_running' else None
    (svgp_corr, svgp_corr_log, svgp_spear,
     svgp_rmse, svgp_rmse_log, svgp_best_params,
     svgp_params_per_fold) = cv_svgp_two(
        fold_indices, X, y_log,
        lat=lat_all, lon=lon_all,           # needed for inner buffered spatial CV
        start_fold=_start_svgp,
        _skill_init=_skill_svgp,
        _rf_state=_rf_state_carry,
        _svgp_best_params_init=_svgp_best_params,
        _tune_progress_init=_svgp_tune_prog,
    )
    ckpt_save({'stage': 'svgp_done', 'fold': K_FOLDS,
               'rf_state': _rf_state_carry,
               'svgp_best_params': svgp_best_params,
               'svgp_params_per_fold': svgp_params_per_fold,
               'svgp_skill': {'corr': svgp_corr, 'corr_log': svgp_corr_log,
                               'spear': svgp_spear, 'rmse': svgp_rmse,
                               'rmse_log': svgp_rmse_log}})

io.savemat(os.path.join(OUTPUT_DIR, 'skill_svgp_global.mat'), {
    'skill_corr':     svgp_corr, 'skill_rmse': svgp_rmse,
    'skill_rmse_log': svgp_rmse_log,
    'skill_corr_log': svgp_corr_log, 'skill_spearman': svgp_spear,
    'feature_names':  np.array(FEATURE_NAMES, dtype=object),
    # tuned architecture (from inner CV on fold 0)
    'n_inducing_beta':   svgp_best_params[0]['n_inducing'] if svgp_best_params[0] else GP_INDUCING['beta'],
    'n_inducing_biovol': svgp_best_params[1]['n_inducing'] if svgp_best_params[1] else GP_INDUCING['biovol'],
    'kernel_beta':       svgp_best_params[0]['kernel']     if svgp_best_params[0] else GP_KERNEL['beta'],
    'kernel_biovol':     svgp_best_params[1]['kernel']     if svgp_best_params[1] else GP_KERNEL['biovol'],
    # architecture selected independently in each outer fold
    'n_inducing_per_fold': np.array(
        [[(c['n_inducing'] if c else -1) for c in pf] for pf in svgp_params_per_fold]),
    'kernel_per_fold': np.array(
        [[(c['kernel'] if c else '') for c in pf] for pf in svgp_params_per_fold],
        dtype=object),
    'cv_block_deg':  BLOCK_DEG, 'cv_buffer_km': BUFFER_KM,
})

# ── Ridge baseline ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RIDGE REGRESSION  (nested CV)")
print(f"  inner grid: {len(RIDGE_ALPHA_GRID)} alphas "
      f"(1e{np.log10(RIDGE_ALPHA_GRID[0]):.0f} … 1e{np.log10(RIDGE_ALPHA_GRID[-1]):.0f})")
print(f"  inner sel.: {INNER_DESC} "
      f"(BLOCK={BLOCK_DEG}°, BUFFER={BUFFER_KM:.0f} km edge-to-edge, seed={INNER_SEED})")
print(f"  hyperparameter selection: "
      f"{'every outer fold' if TUNE_EVERY_FOLD else f'outer fold {TUNE_ON_FOLD} only'}")
print("="*60)
(lin_corr, lin_corr_log, lin_spear, lin_rmse, lin_rmse_log,
 ridge_alphas, ridge_alphas_per_fold) = cv_ridge(
    fold_indices, X, y_log,
    lat=lat_all, lon=lon_all,       # needed for inner buffered spatial CV
)
io.savemat(os.path.join(OUTPUT_DIR, 'skill_linear_global.mat'), {
    'skill_corr':     lin_corr, 'skill_rmse': lin_rmse,
    'skill_rmse_log': lin_rmse_log,
    'skill_corr_log': lin_corr_log, 'skill_spearman': lin_spear,
    'feature_names':  np.array(FEATURE_NAMES, dtype=object),
    # tuned regularisation (from inner CV on fold 0)
    'ridge_alpha_beta':   float(ridge_alphas[0]),
    'ridge_alpha_biovol': float(ridge_alphas[1]),
    # alpha selected independently in each outer fold
    'ridge_alpha_per_fold': np.array(
        [[(a if a is not None else np.nan) for a in af]
         for af in ridge_alphas_per_fold], dtype=float),
    'cv_block_deg':   BLOCK_DEG, 'cv_buffer_km': BUFFER_KM,
})


# ============================================================
# 10) Final models on ALL data
# ============================================================
print("\n" + "="*60)
print("FINAL MODELS ON ALL DATA")
print("="*60)

# Always refit scalers (cheap, needed to reconstruct X_s / y_s)
if os.path.exists(_SCALERS_PATH):
    sx_fin, sy_fin_b, sy_fin_v = _load_scalers()
    print("[ckpt] Scalers loaded from disk")
else:
    sx_fin   = StandardScaler().fit(X)
    sy_fin_b = StandardScaler().fit(y_log[:, [0]])
    sy_fin_v = StandardScaler().fit(y_log[:, [1]])
    _save_scalers(sx_fin, sy_fin_b, sy_fin_v)

X_s = sx_fin.transform(X).astype(np.float32)
y_s = np.column_stack([
    sy_fin_b.transform(y_log[:, [0]]).ravel(),
    sy_fin_v.transform(y_log[:, [1]]).ravel(),
]).astype(np.float32)
del X, y_log; gc.collect()

# ── Final RF ──────────────────────────────────────────────────────────────────
if os.path.exists(_RF_PATH):
    print("[ckpt] Loading final RF from disk — skipping training")
    rf_final = _load_rf()
else:
    print("Training final RF...")
    rf_final = rf_train(X_s, y_s, params=rf_best_params)
    _save_rf(rf_final)
    _cs = ckpt_load() or {}; _cs['stage'] = 'final_rf_done'; ckpt_save(_cs)

if should_checkpoint():
    print("[ckpt] Time limit after final RF — requeueing", flush=True)
    requeue_and_exit()

# ── Resolve final SVGP architecture from tuned params ────────────────────────
# Use tuned values from inner CV; fall back to config defaults if not available.
_fin_n_ind_beta   = (svgp_best_params[0]['n_inducing'] if svgp_best_params[0]
                     else GP_INDUCING['beta'])
_fin_kern_beta    = (svgp_best_params[0]['kernel']     if svgp_best_params[0]
                     else GP_KERNEL['beta'])
_fin_n_ind_biovol = (svgp_best_params[1]['n_inducing'] if svgp_best_params[1]
                     else GP_INDUCING['biovol'])
_fin_kern_biovol  = (svgp_best_params[1]['kernel']     if svgp_best_params[1]
                     else GP_KERNEL['biovol'])
print(f"  Final SVGP beta:   n_inducing={_fin_n_ind_beta},   kernel={_fin_kern_beta}")
print(f"  Final SVGP biovol: n_inducing={_fin_n_ind_biovol}, kernel={_fin_kern_biovol}")

# ── Final SVGP beta ───────────────────────────────────────────────────────────
if os.path.exists(_SVGP_BETA_PATH):
    print("[ckpt] Loading final SVGP beta from disk — skipping training")
    svgp_beta_final = _load_svgp(_SVGP_BETA_PATH, X_s.shape[1])
else:
    print("Training final SVGP for beta...")
    try:
        svgp_beta_final = train_single_svgp(
            X_s, y_s[:, 0], _fin_n_ind_beta,
            kernel=_fin_kern_beta, label='beta-final')
        _save_svgp(svgp_beta_final, _SVGP_BETA_PATH)
        _cs = ckpt_load() or {}; _cs['stage'] = 'final_svgp_beta_done'; ckpt_save(_cs)
    except _CheckpointSignal:
        print("[ckpt] Time limit during final SVGP beta — requeueing", flush=True)
        requeue_and_exit()

if should_checkpoint():
    print("[ckpt] Time limit after final SVGP beta — requeueing", flush=True)
    requeue_and_exit()

# ── Final SVGP biovol ─────────────────────────────────────────────────────────
if os.path.exists(_SVGP_BIOV_PATH):
    print("[ckpt] Loading final SVGP biovol from disk — skipping training")
    svgp_biovol_final = _load_svgp(_SVGP_BIOV_PATH, X_s.shape[1])
else:
    print("Training final SVGP for biovolume...")
    try:
        svgp_biovol_final = train_single_svgp(
            X_s, y_s[:, 1], _fin_n_ind_biovol,
            kernel=_fin_kern_biovol, label='biovol-final')
        _save_svgp(svgp_biovol_final, _SVGP_BIOV_PATH)
        _cs = ckpt_load() or {}; _cs['stage'] = 'final_svgp_biovol_done'; ckpt_save(_cs)
    except _CheckpointSignal:
        print("[ckpt] Time limit during final SVGP biovol — requeueing", flush=True)
        requeue_and_exit()

if should_checkpoint():
    print("[ckpt] Time limit after final SVGP biovol — requeueing", flush=True)
    requeue_and_exit()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================
_fi_rng = np.random.default_rng(0)
n_feat  = len(FEATURE_NAMES)
fnames  = np.array(FEATURE_NAMES)
X_orig  = sx_fin.inverse_transform(X_s)


def _fi_subsample(n_take, seed):
    n = X_s.shape[0]
    k = min(n_take, n)
    return np.random.RandomState(seed).choice(n, k, replace=False)


def _r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-12)


# ============================================================
# 11) SHAP — RandomForest (TreeSHAP)
# ============================================================
# NOTE: RF SHAP explanation set uses TEST points for consistency with SVGP.
# Computation is deferred to after test data is loaded (Section 12).
_shap_rf_mat = os.path.join(OUTPUT_DIR, 'shap_rf_train.mat')
if os.path.exists(_shap_rf_mat):
    print("[ckpt] shap_rf_train.mat exists — loading SHAP absmean, skipping recompute")
    _tmp = io.loadmat(_shap_rf_mat)
    rf_shap_absmean = {TARGET_NAMES[0]: _tmp['shap_absmean_beta'].ravel(),
                       TARGET_NAMES[1]: _tmp['shap_absmean_biovol'].ravel()}
    _rf_shap_needed = False
else:
    print("[11] RF SHAP deferred — will run after test data is loaded")
    _rf_shap_needed = True
    rf_shap_absmean = {}

if should_checkpoint():
    print("[ckpt] Time limit after SHAP RF check — requeueing", flush=True)
    requeue_and_exit()



# ============================================================
# 11b) SHAP — SVGP (GradientSHAP)
# ============================================================
class _SVGPMeanModule(torch.nn.Module):
    def __init__(self, gp_model):
        super().__init__()
        self.gp_model = gp_model

    def forward(self, x):
        return self.gp_model(x).mean.unsqueeze(-1)

# NOTE: SVGP SHAP explanation set uses TEST points (not training points)
# so that the explained distribution matches the reconstruction domain.
# The background set uses training points as the reference distribution.
# Computation is deferred to after test data is loaded (Section 12).
_shap_svgp_mat = os.path.join(OUTPUT_DIR, 'shap_svgp_train.mat')
if os.path.exists(_shap_svgp_mat):
    print("[ckpt] shap_svgp_train.mat exists — loading absmean, skipping recompute")
    _tmp = io.loadmat(_shap_svgp_mat)
    svgp_shap = {TARGET_NAMES[0]: _tmp['shap_absmean_beta'].ravel(),
                 TARGET_NAMES[1]: _tmp['shap_absmean_biovol'].ravel()}
    _svgp_shap_needed = False
else:
    print("[11b] SVGP SHAP deferred — will run after test data is loaded")
    _svgp_shap_needed = True
    svgp_shap = {}

if should_checkpoint():
    print("[ckpt] Time limit after SHAP SVGP check — requeueing", flush=True)
    requeue_and_exit()


# ============================================================
# 11c) Per-variable permutation importance
#      Permute one feature at a time; measure R² drop.
#      Repeated PERM_N_REPEATS times for stability.
# ============================================================
print("\nPer-variable permutation importance (single-feature shuffle)...")


def single_perm_importance(predict_fn, X, y_true, n_repeats=5, seed=0):
    """
    For each feature independently: shuffle that column, measure R² drop.
    Returns (base_r2, mean_drop per feature, std_drop per feature).
    """
    rng    = np.random.RandomState(seed)
    base   = _r2(y_true, predict_fn(X))
    n_feat = X.shape[1]
    drop_mean = np.zeros(n_feat)
    drop_std  = np.zeros(n_feat)
    for f in range(n_feat):
        drops = []
        for _ in range(n_repeats):
            Xp       = X.copy()
            Xp[:, f] = Xp[rng.permutation(len(Xp)), f]
            drops.append(base - _r2(y_true, predict_fn(Xp)))
        drop_mean[f] = np.mean(drops)
        drop_std[f]  = np.std(drops)
    return base, drop_mean, drop_std


pi_rf_idx   = _fi_subsample(PERM_SAMPLE_RF,   seed=20)
pi_svgp_idx = _fi_subsample(PERM_SAMPLE_SVGP, seed=21)

perm_models = [
    ('RF',   TARGET_NAMES[0], lambda A: rf_final.predict(A)[:, 0], pi_rf_idx,   0),
    ('RF',   TARGET_NAMES[1], lambda A: rf_final.predict(A)[:, 1], pi_rf_idx,   1),
    ('SVGP', TARGET_NAMES[0], svgp_beta_final.predict_np,          pi_svgp_idx, 0),
    ('SVGP', TARGET_NAMES[1], svgp_biovol_final.predict_np,        pi_svgp_idx, 1),
]

perm_results = {}
for mdl, tname, pfn, gi, q in perm_models:
    base, dmean, dstd = single_perm_importance(
        pfn, X_s[gi], y_s[gi, q],
        n_repeats=PERM_N_REPEATS, seed=q
    )
    perm_results[f'{mdl}_{tname}'] = (base, dmean, dstd)
    top = fnames[np.argmax(dmean)]
    print(f"  [{mdl} {tname}] base R²={base:.3f}  top feature: {top}")

# Figure: per-variable permutation importance (2x2)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Per-variable permutation importance  (single-feature R² drop)',
             fontsize=14, fontweight='bold')
clr_pos = '#2166ac'
clr_neg = '#d6604d'

for ax, (mdl, tname, _, gi, q) in zip(axes.flat, perm_models):
    base, dmean, dstd = perm_results[f'{mdl}_{tname}']
    order      = np.argsort(dmean)
    fnames_ord = fnames[order]
    colors     = [clr_pos if v >= 0 else clr_neg for v in dmean[order]]
    ax.barh(fnames_ord, dmean[order], xerr=dstd[order],
            color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('R² drop  (higher = more important)', fontsize=10)
    ax.set_title(f'{mdl} — {tname}   (base R²={base:.3f})',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig(f"{SCATTER_DIR}/permutation_importance_single.png",
            dpi=140, bbox_inches='tight')
plt.close()
print("✅  Saved permutation_importance_single.png")


# ============================================================
# 11d) Grouped permutation importance (collinearity-robust)
# ============================================================
print("\nGrouped permutation importance (correlation-clustered)...")

# ── Physically motivated feature groups ──────────────────────────────────────
# Groups are defined a priori based on oceanographic interpretation
# rather than data-driven clustering, to avoid collinearity-driven
# arbitrary groupings and to produce physically interpretable results.
FEATURE_GROUPS = {
    'Productivity':   ['NPP', 'Chl', 'NO3'],
    'Physical_state': ['TEMP', 'Salinity', 'O2', 'MLD', 'Thflx'],
    'Geometry':       ['Depth', 'Bathymetry'],
    'Seasonality':    ['month_sin', 'month_cos'],
}

feat_to_idx = {f: i for i, f in enumerate(FEATURE_NAMES)}
groups = []
for grp_name, feat_list in FEATURE_GROUPS.items():
    cols = [feat_to_idx[f] for f in feat_list if f in feat_to_idx]
    if cols:
        groups.append((grp_name, cols))

# dummy cluster_id for backward compatibility with mat file saves
cluster_id = np.zeros(len(FEATURE_NAMES), dtype=int)
for grp_idx, (_, cols) in enumerate(groups):
    for c in cols:
        cluster_id[c] = grp_idx

print(f"  {len(groups)} physically defined feature groups:")
for lbl, cols in groups:
    print(f"    [{lbl}]: {[FEATURE_NAMES[c] for c in cols]}")


def grouped_perm(predict_fn, X, y_true, groups, n_repeats, seed=0):
    rng  = np.random.RandomState(seed)
    base = _r2(y_true, predict_fn(X))
    out_mean, out_std = [], []
    for lbl, cols in groups:
        drops = []
        for _ in range(n_repeats):
            Xp          = X.copy()
            perm        = rng.permutation(len(Xp))
            Xp[:, cols] = Xp[perm][:, cols]
            drops.append(base - _r2(y_true, predict_fn(Xp)))
        out_mean.append(np.mean(drops))
        out_std.append(np.std(drops))
    return base, np.array(out_mean), np.array(out_std)


group_labels    = [g[0] for g in groups]
grouped_results = {}

# One colour per physical group — consistent across all four panels
_GRP_COLORS = {
    'Productivity':   '#2ca02c',   # green
    'Physical_state': '#1f77b4',   # blue
    'Geometry':       '#d62728',   # red
    'Seasonality':    '#ff7f0e',   # orange
}

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Grouped permutation importance  (R² drop, physically defined groups)',
             fontsize=13, fontweight='bold')
for ax, (mdl, tname, pfn, gi, q) in zip(axes.flat, perm_models):
    base, gmean, gstd = grouped_perm(
        pfn, X_s[gi], y_s[gi, q], groups, GROUP_N_REPEATS, seed=q)
    grouped_results[f'{mdl}_{tname}'] = (gmean, gstd)
    order  = np.argsort(gmean)
    labels_ord = np.array(group_labels)[order]
    colors_ord = [_GRP_COLORS.get(l, 'grey') for l in labels_ord]
    ax.barh(labels_ord, gmean[order],
            xerr=gstd[order], color=colors_ord, alpha=0.85,
            edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('R² drop  (grouped permutation)', fontsize=10)
    ax.set_title(f'{mdl} — {tname}   (base R²={base:.3f})',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.tick_params(labelsize=9)
    print(f"  [{mdl} {tname}] base R²={base:.3f}")
plt.tight_layout()
plt.savefig(f"{SCATTER_DIR}/grouped_permutation_importance.png",
            dpi=140, bbox_inches='tight')
plt.close()
print("✅  Saved grouped_permutation_importance.png")


# ============================================================
# 11e) Combined summary: SHAP mean |value| + per-variable perm R² drop
#      Side-by-side horizontal bars, both normalised to [0,1]
#      so the two metrics are visually comparable on the same axis.
# ============================================================
print("\nBuilding combined SHAP + permutation importance figure...")

shap_absmean = {
    f'RF_{TARGET_NAMES[0]}':   rf_shap_absmean.get(TARGET_NAMES[0],
                                np.zeros(len(FEATURE_NAMES), dtype=np.float32)),
    f'RF_{TARGET_NAMES[1]}':   rf_shap_absmean.get(TARGET_NAMES[1],
                                np.zeros(len(FEATURE_NAMES), dtype=np.float32)),
    f'SVGP_{TARGET_NAMES[0]}': svgp_shap.get(TARGET_NAMES[0],
                                np.zeros(len(FEATURE_NAMES), dtype=np.float32)),
    f'SVGP_{TARGET_NAMES[1]}': svgp_shap.get(TARGET_NAMES[1],
                                np.zeros(len(FEATURE_NAMES), dtype=np.float32)),
}


def _norm01(arr):
    rng = arr.max() - arr.min()
    return (arr - arr.min()) / rng if rng > 1e-12 else np.zeros_like(arr)


fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Feature importance: mean |SHAP| vs single-feature permutation R² drop\n'
             '(both normalised to [0–1] for visual comparison)',
             fontsize=13, fontweight='bold')

bar_h    = 0.35
feat_idx = np.arange(n_feat)
clr_shap = '#2166ac'
clr_perm = '#d6604d'

for ax, (mdl, tname, _, gi, q) in zip(axes.flat, perm_models):
    key = f'{mdl}_{tname}'

    shap_vals = _norm01(shap_absmean[key])
    perm_vals = _norm01(np.maximum(perm_results[key][1], 0.0))

    # sort by average rank across the two metrics
    order      = np.argsort(0.5 * shap_vals + 0.5 * perm_vals)
    fnames_ord = fnames[order]
    shap_ord   = shap_vals[order]
    perm_ord   = perm_vals[order]

    y_pos = feat_idx
    ax.barh(y_pos - bar_h/2, shap_ord, bar_h,
            color=clr_shap, alpha=0.85, label='Mean |SHAP|',
            edgecolor='black', linewidth=0.4)
    ax.barh(y_pos + bar_h/2, perm_ord, bar_h,
            color=clr_perm,  alpha=0.85, label='Perm. R² drop',
            edgecolor='black', linewidth=0.4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(fnames_ord, fontsize=9)
    ax.set_xlabel('Normalised importance  [0–1]', fontsize=10)
    ax.set_title(f'{mdl} — {tname}', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1.15)
    ax.grid(axis='x', alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')

    base_r2 = perm_results[key][0]
    ax.text(0.98, 0.02, f'base R² = {base_r2:.3f}',
            transform=ax.transAxes, fontsize=8,
            ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray',
                      boxstyle='round'))

plt.tight_layout()
plt.savefig(f"{SCATTER_DIR}/feature_importance_combined.png",
            dpi=140, bbox_inches='tight')
plt.close()
print("✅  Saved feature_importance_combined.png")


# Save all importance metrics
io.savemat(os.path.join(OUTPUT_DIR, 'feature_importance_all.mat'), {
    'feature_names':          np.array(FEATURE_NAMES, dtype=object),
    'group_labels':           np.array(group_labels,  dtype=object),
    'cluster_id':             cluster_id,
    'cluster_dist_threshold': float(CLUSTER_DIST_THRESH),
    **{f'perm_drop_mean_{k}': v[1] for k, v in perm_results.items()},
    **{f'perm_drop_std_{k}':  v[2] for k, v in perm_results.items()},
    **{f'perm_base_r2_{k}':   v[0] for k, v in perm_results.items()},
    **{f'group_drop_mean_{k}': v[0] for k, v in grouped_results.items()},
    **{f'group_drop_std_{k}':  v[1] for k, v in grouped_results.items()},
    **{f'shap_absmean_{k}': v for k, v in shap_absmean.items()},
})
print("✅  Saved feature_importance_all.mat")
print("Feature importance done (SHAP + per-variable perm + grouped perm).")


# ============================================================
# 12) Test set: load, predict, save
# ============================================================
print("\n" + "="*60)
print("PREDICTING ON TEST SET")
print("="*60)


def stack_test_var(var_key, ravel=False):
    chunks = []
    for j in range(N_DEPTHS):
        a = io.loadmat(f'data_{var_key}_test_{j}.mat')[var_key].astype(np.float32)
        chunks.append(a.ravel() if ravel else a)
    out = np.vstack(chunks) if not ravel else np.concatenate(chunks)
    del chunks; gc.collect()
    return out


X_test         = stack_test_var('X')
lon_test       = stack_test_var('lon',       ravel=True)
lat_test       = stack_test_var('lat',       ravel=True)
month_sin_test = stack_test_var('month_sin', ravel=True)
month_cos_test = stack_test_var('month_cos', ravel=True)
depth_test     = stack_test_var('depth',     ravel=True)

# Reconstruct integer month from sin/cos for reshape_to_4d
# atan2 returns angle in [-pi, pi], convert back to month 1-12
month_test = (np.round(
    np.arctan2(month_sin_test, month_cos_test) * 12 / (2 * np.pi)
).astype(int) % 12) + 1

print(f"Test rows: {X_test.shape[0]:,}")
print(f"  month_sin_test: {month_sin_test.shape[0]:,}")
print(f"  month_cos_test: {month_cos_test.shape[0]:,}")

# Use the minimum length across all arrays to handle any size mismatches
# from the test preprocessing script
n_min = min(X_test.shape[0], lon_test.shape[0], lat_test.shape[0],
            month_sin_test.shape[0], month_cos_test.shape[0],
            depth_test.shape[0])
X_test         = X_test[:n_min]
lon_test       = lon_test[:n_min]
lat_test       = lat_test[:n_min]
month_sin_test = month_sin_test[:n_min]
month_cos_test = month_cos_test[:n_min]
depth_test     = depth_test[:n_min]

# Reconstruct integer month from sin/cos
month_test = (np.round(
    np.arctan2(month_sin_test, month_cos_test) * 12 / (2 * np.pi)
).astype(int) % 12) + 1

keep           = ~np.isnan(X_test).any(axis=1)
X_test         = X_test[keep];         lon_test       = lon_test[keep]
lat_test       = lat_test[keep];       month_test     = month_test[keep]
month_sin_test = month_sin_test[keep]; month_cos_test = month_cos_test[keep]
depth_test     = depth_test[keep]
del keep; gc.collect()
print(f"After NaN drop: {X_test.shape[0]:,}")

X_test_s = sx_fin.transform(X_test).astype(np.float32)
del X_test; gc.collect()


# ============================================================
# 11b deferred) SVGP GradientSHAP — explanation from TEST points
# ============================================================
if _svgp_shap_needed:
    print("\nSHAP — SVGP (GradientSHAP, test points as explanation set)...")
    torch.manual_seed(0)

    # Background: sample from training data
    idx_bg  = _fi_subsample(SHAP_SVGP_BG, seed=10)
    bg_t    = torch.tensor(X_s[idx_bg], dtype=torch.float32, device=DEVICE)

    # Explanation: sample from test grid
    n_test_avail = X_test_s.shape[0]
    idx_exp_t = np.random.RandomState(11).choice(
        n_test_avail, min(SHAP_SVGP_EXPLAIN, n_test_avail), replace=False)
    Xexp_t      = torch.tensor(X_test_s[idx_exp_t], dtype=torch.float32, device=DEVICE)
    Xexp_svgp_o = sx_fin.inverse_transform(X_test_s[idx_exp_t])  # original scale

    svgp_shap   = {}
    svgp_sv_raw = {}
    for wrapper, tname in [(svgp_beta_final,   TARGET_NAMES[0]),
                           (svgp_biovol_final, TARGET_NAMES[1])]:
        print(f"  GradientSHAP for {tname} "
              f"({len(idx_exp_t)} test rows, nsamples={SHAP_SVGP_NSAMPLES})...")
        wrapper.model.eval(); wrapper.likelihood.eval()
        mean_mod = _SVGPMeanModule(wrapper.model).to(DEVICE).eval()

        with gpytorch.settings.fast_pred_var(state=False):
            expl  = shap.GradientExplainer(mean_mod, bg_t)
            parts = []
            for s in range(0, len(idx_exp_t), SHAP_SVGP_CHUNK):
                xb  = Xexp_t[s:s + SHAP_SVGP_CHUNK]
                svc = expl.shap_values(xb, nsamples=SHAP_SVGP_NSAMPLES)
                svc = svc[0] if isinstance(svc, list) else svc
                parts.append(np.asarray(svc).reshape(xb.shape[0], -1))
        sv = np.concatenate(parts, axis=0).astype(np.float32)

        svgp_sv_raw[tname] = sv
        svgp_shap[tname]   = np.abs(sv).mean(axis=0)

        plt.figure(figsize=(7, 5))
        shap.summary_plot(sv, Xexp_svgp_o, feature_names=FEATURE_NAMES, show=False,
                          plot_size=None)
        plt.title(f'SVGP SHAP — {tname}  (test grid explanation)')
        plt.tight_layout()
        plt.savefig(f"{SCATTER_DIR}/shap_svgp_{tname}_beeswarm.png", dpi=140,
                    bbox_inches='tight')
        plt.close()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    io.savemat(_shap_svgp_mat, {
        'shap_values_beta':    svgp_sv_raw[TARGET_NAMES[0]],
        'shap_values_biovol':  svgp_sv_raw[TARGET_NAMES[1]],
        'shap_absmean_beta':   svgp_shap[TARGET_NAMES[0]],
        'shap_absmean_biovol': svgp_shap[TARGET_NAMES[1]],
        'X_explained':         Xexp_svgp_o.astype(np.float32),
        'lat':   lat_test[idx_exp_t].astype(np.float32),
        'lon':   lon_test[idx_exp_t].astype(np.float32),
        'depth': depth_test[idx_exp_t].astype(np.float32),
        'month': month_test[idx_exp_t].astype(np.int32),
        'feature_names': np.array(FEATURE_NAMES, dtype=object),
        'target_names':  np.array(TARGET_NAMES, dtype=object),
        'n_explained': len(idx_exp_t), 'background': SHAP_SVGP_BG,
    })
    print("  saved shap_svgp_*_beeswarm.png + shap_svgp_train.mat")

    # Update feature_importance_all.mat with real SVGP SHAP values
    shap_absmean.update({
        f'SVGP_{TARGET_NAMES[0]}': svgp_shap[TARGET_NAMES[0]],
        f'SVGP_{TARGET_NAMES[1]}': svgp_shap[TARGET_NAMES[1]],
    })
    io.savemat(os.path.join(OUTPUT_DIR, 'feature_importance_all.mat'), {
        'feature_names':          np.array(FEATURE_NAMES, dtype=object),
        'group_labels':           np.array(group_labels,  dtype=object),
        'cluster_id':             cluster_id,
        'cluster_dist_threshold': float(CLUSTER_DIST_THRESH),
        **{f'perm_drop_mean_{k}': v[1] for k, v in perm_results.items()},
        **{f'perm_drop_std_{k}':  v[2] for k, v in perm_results.items()},
        **{f'perm_base_r2_{k}':   v[0] for k, v in perm_results.items()},
        **{f'group_drop_mean_{k}': v[0] for k, v in grouped_results.items()},
        **{f'group_drop_std_{k}':  v[1] for k, v in grouped_results.items()},
        **{f'shap_absmean_{k}': v for k, v in shap_absmean.items()},
    })
    print("✅  Updated feature_importance_all.mat with SVGP SHAP values")

    if should_checkpoint():
        print("[ckpt] Time limit after SVGP SHAP — requeueing", flush=True)
        requeue_and_exit()


# ============================================================
# ============================================================
# 11) deferred) RF TreeSHAP — explanation from TEST points
# ============================================================
if _rf_shap_needed:
    print("\nSHAP — RandomForest (TreeSHAP, test points as explanation set)...")
    n_test_avail = X_test_s.shape[0]
    idx_rf = np.random.RandomState(0).choice(
        n_test_avail, min(SHAP_RF_EXPLAIN, n_test_avail), replace=False)
    Xexp_s = X_test_s[idx_rf]
    Xexp_o = sx_fin.inverse_transform(Xexp_s)  # original scale for plots

    expl_rf = shap.TreeExplainer(rf_final, feature_perturbation='tree_path_dependent')
    sv_rf   = expl_rf.shap_values(Xexp_s, check_additivity=False, approximate=True)

    if isinstance(sv_rf, list):
        sv_rf_list = sv_rf
    elif sv_rf.ndim == 3:
        sv_rf_list = [sv_rf[:, :, q] for q in range(sv_rf.shape[2])]
    else:
        sv_rf_list = [sv_rf]

    rf_shap_absmean = {}
    for q, tname in enumerate(TARGET_NAMES):
        sv = sv_rf_list[q]
        rf_shap_absmean[tname] = np.abs(sv).mean(axis=0)
        plt.figure(figsize=(7, 5))
        shap.summary_plot(sv, Xexp_o, feature_names=FEATURE_NAMES, show=False,
                          plot_size=None)
        plt.title(f'RF SHAP — {tname}  (test grid explanation)')
        plt.tight_layout()
        plt.savefig(f"{SCATTER_DIR}/shap_rf_{tname}_beeswarm.png", dpi=140,
                    bbox_inches='tight')
        plt.close()

    sv_rf_stack = np.stack(sv_rf_list, axis=-1).astype(np.float32)
    io.savemat(_shap_rf_mat, {
        'shap_values':         sv_rf_stack,
        'shap_absmean_beta':   rf_shap_absmean[TARGET_NAMES[0]],
        'shap_absmean_biovol': rf_shap_absmean[TARGET_NAMES[1]],
        'X_explained':         Xexp_o.astype(np.float32),
        'lat':       lat_test[idx_rf].astype(np.float32),
        'lon':       lon_test[idx_rf].astype(np.float32),
        'depth':     depth_test[idx_rf].astype(np.float32),
        'month':     month_test[idx_rf].astype(np.int32),
        'month_sin': month_sin_test[idx_rf].astype(np.float32),
        'month_cos': month_cos_test[idx_rf].astype(np.float32),
        'feature_names': np.array(FEATURE_NAMES, dtype=object),
        'target_names':  np.array(TARGET_NAMES, dtype=object),
        'n_explained': len(idx_rf),
    })
    print("  saved shap_rf_*_beeswarm.png + shap_rf_train.mat")

    # Update shap_absmean and feature_importance_all.mat with RF SHAP values
    shap_absmean.update({
        f'RF_{TARGET_NAMES[0]}': rf_shap_absmean[TARGET_NAMES[0]],
        f'RF_{TARGET_NAMES[1]}': rf_shap_absmean[TARGET_NAMES[1]],
    })
    io.savemat(os.path.join(OUTPUT_DIR, 'feature_importance_all.mat'), {
        'feature_names':          np.array(FEATURE_NAMES, dtype=object),
        'group_labels':           np.array(group_labels,  dtype=object),
        'cluster_id':             cluster_id,
        'cluster_dist_threshold': float(CLUSTER_DIST_THRESH),
        **{f'perm_drop_mean_{k}': v[1] for k, v in perm_results.items()},
        **{f'perm_drop_std_{k}':  v[2] for k, v in perm_results.items()},
        **{f'perm_base_r2_{k}':   v[0] for k, v in perm_results.items()},
        **{f'group_drop_mean_{k}': v[0] for k, v in grouped_results.items()},
        **{f'group_drop_std_{k}':  v[1] for k, v in grouped_results.items()},
        **{f'shap_absmean_{k}': v for k, v in shap_absmean.items()},
    })
    print("✅  Updated feature_importance_all.mat with RF SHAP values")

    if should_checkpoint():
        print("[ckpt] Time limit after RF SHAP — requeueing", flush=True)
        requeue_and_exit()


# 12b) SHAP on the TEST grid
# ============================================================
print("\nSHAP on the TEST grid (dense driver maps)...")
n_test = X_test_s.shape[0]
idx_t  = np.random.RandomState(42).choice(n_test, min(SHAP_TESTGRID_N, n_test),
                                           replace=False)
Xt_s   = X_test_s[idx_t]
test_coords = {
    'lat':       lat_test[idx_t].astype(np.float32),
    'lon':       lon_test[idx_t].astype(np.float32),
    'depth':     depth_test[idx_t].astype(np.float32),
    'month':     month_test[idx_t].astype(np.int32),
    'month_sin': month_sin_test[idx_t].astype(np.float32),
    'month_cos': month_cos_test[idx_t].astype(np.float32),
    'feature_names': np.array(FEATURE_NAMES, dtype=object),
    'target_names':  np.array(TARGET_NAMES, dtype=object),
    'n_explained': len(idx_t), 'grid': np.array('test', dtype=object),
}

print(f"  TreeSHAP (RF) on {len(idx_t):,} test rows...")
expl_rf_t = shap.TreeExplainer(rf_final, feature_perturbation='tree_path_dependent')
svt = expl_rf_t.shap_values(Xt_s, check_additivity=False, approximate=True)
if isinstance(svt, list):
    svt_list = svt
elif svt.ndim == 3:
    svt_list = [svt[:, :, q] for q in range(svt.shape[2])]
else:
    svt_list = [svt]
sv_rf_test = np.stack(svt_list, axis=-1).astype(np.float32)
io.savemat(os.path.join(OUTPUT_DIR, 'shap_rf.mat'), {
    'shap_values':         sv_rf_test,
    'shap_absmean_beta':   np.abs(svt_list[0]).mean(axis=0),
    'shap_absmean_biovol': np.abs(svt_list[1]).mean(axis=0),
    **test_coords,
})
print("  saved shap_rf.mat (test grid)")

torch.manual_seed(0)
bg_idx = np.random.RandomState(7).choice(n_test, min(SHAP_SVGP_BG, n_test),
                                         replace=False)
bg_t  = torch.tensor(X_test_s[bg_idx], dtype=torch.float32, device=DEVICE)
Xt_t  = torch.tensor(Xt_s,             dtype=torch.float32, device=DEVICE)
svgp_test = {}
for wrapper, tname in [(svgp_beta_final,   TARGET_NAMES[0]),
                       (svgp_biovol_final, TARGET_NAMES[1])]:
    print(f"  GradientSHAP (SVGP) on test grid — {tname}...")
    wrapper.model.eval(); wrapper.likelihood.eval()
    mean_mod = _SVGPMeanModule(wrapper.model).to(DEVICE).eval()
    with gpytorch.settings.fast_pred_var(state=False):
        expl  = shap.GradientExplainer(mean_mod, bg_t)
        parts = []
        for s in range(0, len(idx_t), SHAP_SVGP_CHUNK):
            xb  = Xt_t[s:s + SHAP_SVGP_CHUNK]
            svc = expl.shap_values(xb, nsamples=SHAP_SVGP_NSAMPLES)
            svc = svc[0] if isinstance(svc, list) else svc
            parts.append(np.asarray(svc).reshape(xb.shape[0], -1))
    svgp_test[tname] = np.concatenate(parts, axis=0).astype(np.float32)
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

io.savemat(os.path.join(OUTPUT_DIR, 'shap_svgp.mat'), {
    'shap_values_beta':    svgp_test[TARGET_NAMES[0]],
    'shap_values_biovol':  svgp_test[TARGET_NAMES[1]],
    'shap_absmean_beta':   np.abs(svgp_test[TARGET_NAMES[0]]).mean(axis=0),
    'shap_absmean_biovol': np.abs(svgp_test[TARGET_NAMES[1]]).mean(axis=0),
    'background': SHAP_SVGP_BG,
    **test_coords,
})
print("  saved shap_svgp.mat (test grid)")
del Xt_t, bg_t
if DEVICE.type == 'cuda':
    torch.cuda.empty_cache()
gc.collect()


# ============================================================
# 13) Predict on test set
# ============================================================
# ── Skip predictions if output already exists ─────────────────────────────────
_pred_rf_path   = os.path.join(OUTPUT_DIR, 'prediction_rf_global_long.mat')
_pred_svgp_path = os.path.join(OUTPUT_DIR, 'prediction_svgp_global_long.mat')
_skip_predict   = os.path.exists(_pred_rf_path) and os.path.exists(_pred_svgp_path)
if _skip_predict:
    print("[ckpt] Prediction .mat files already exist — skipping prediction")
    print("\n✅ All done (predictions already saved from previous run).")
    print(f"Output directory: {OUTPUT_DIR}")
    for _cp in [CKPT_PATH, CKPT_PATH + '.tmp']:
        if os.path.exists(_cp): os.remove(_cp)
    print("[ckpt] Checkpoint deleted.", flush=True)
    import sys; sys.exit(0)

print("Predicting with RF...")
rf_pred_s       = rf_final.predict(X_test_s)
rf_pred_b_log   = sy_fin_b.inverse_transform(rf_pred_s[:, [0]]).ravel()
rf_pred_v_log   = sy_fin_v.inverse_transform(rf_pred_s[:, [1]]).ravel()
rf_pred_beta_orig   = np.maximum(np.exp(rf_pred_b_log) - EPS, 0.0).astype(np.float32)
rf_pred_biovol_orig = np.maximum(np.exp(rf_pred_v_log) - EPS, 0.0).astype(np.float32)
rf_predictions = np.column_stack([rf_pred_beta_orig, rf_pred_biovol_orig])
del rf_pred_s, rf_pred_b_log, rf_pred_v_log
del rf_pred_beta_orig, rf_pred_biovol_orig
gc.collect()

print("Predicting with SVGP (beta)...")
svgp_pred_b_s, std_epi_b_s, std_alea_b_s, std_tot_b_s = \
    svgp_beta_final.predict_np_decomposed(X_test_s)
print("Predicting with SVGP (biovol)...")
svgp_pred_v_s, std_epi_v_s, std_alea_v_s, std_tot_v_s = \
    svgp_biovol_final.predict_np_decomposed(X_test_s)
del X_test_s; gc.collect()

svgp_pred_b_log = sy_fin_b.inverse_transform(svgp_pred_b_s.reshape(-1, 1)).ravel()
svgp_pred_v_log = sy_fin_v.inverse_transform(svgp_pred_v_s.reshape(-1, 1)).ravel()
svgp_pred_beta_orig   = np.maximum(np.exp(svgp_pred_b_log) - EPS, 0.0).astype(np.float32)
svgp_pred_biovol_orig = np.maximum(np.exp(svgp_pred_v_log) - EPS, 0.0).astype(np.float32)


def back_transform_std(std_scaled, mu_orig, y_scaler):
    std_log = std_scaled * y_scaler.scale_[0]
    return (mu_orig * std_log).astype(np.float32)


svgp_std_epi_beta    = back_transform_std(std_epi_b_s,  svgp_pred_beta_orig,   sy_fin_b)
svgp_std_alea_beta   = back_transform_std(std_alea_b_s, svgp_pred_beta_orig,   sy_fin_b)
svgp_std_tot_beta    = back_transform_std(std_tot_b_s,  svgp_pred_beta_orig,   sy_fin_b)
svgp_std_epi_biovol  = back_transform_std(std_epi_v_s,  svgp_pred_biovol_orig, sy_fin_v)
svgp_std_alea_biovol = back_transform_std(std_alea_v_s, svgp_pred_biovol_orig, sy_fin_v)
svgp_std_tot_biovol  = back_transform_std(std_tot_v_s,  svgp_pred_biovol_orig, sy_fin_v)

svgp_predictions = np.column_stack([svgp_pred_beta_orig, svgp_pred_biovol_orig])
svgp_std_epi     = np.column_stack([svgp_std_epi_beta,   svgp_std_epi_biovol])
svgp_std_alea    = np.column_stack([svgp_std_alea_beta,  svgp_std_alea_biovol])
svgp_std_total   = np.column_stack([svgp_std_tot_beta,   svgp_std_tot_biovol])
svgp_predictions_std = svgp_std_total.astype(np.float32)

for q, tname in enumerate(TARGET_NAMES):
    eps2 = (svgp_std_epi[:, q]**2).mean()
    al2  = (svgp_std_alea[:, q]**2).mean()
    tot2 = eps2 + al2
    print(f"  [SVGP {tname:<12s}]  avg variance: "
          f"epistemic={eps2:.3e} ({100*eps2/tot2:.1f}%)  "
          f"aleatoric={al2:.3e}  ({100*al2/tot2:.1f}%)")

del svgp_pred_b_s, svgp_pred_v_s
del std_epi_b_s, std_alea_b_s, std_tot_b_s
del std_epi_v_s, std_alea_v_s, std_tot_v_s
del svgp_pred_b_log, svgp_pred_v_log
del svgp_pred_beta_orig, svgp_pred_biovol_orig
del svgp_std_epi_beta,  svgp_std_alea_beta,  svgp_std_tot_beta
del svgp_std_epi_biovol, svgp_std_alea_biovol, svgp_std_tot_biovol
gc.collect()


# ============================================================
# 14) Save: long-table + 4D
# ============================================================
def save_long(name, pred, std=None, std_epistemic=None, std_aleatoric=None):
    out = {
        'predictions': pred,
        'lat':       lat_test,
        'lon':       lon_test,
        'month':     month_test,
        'month_sin': month_sin_test,
        'month_cos': month_cos_test,
        'depth':     depth_test,
        'target_names': np.array(TARGET_NAMES, dtype=object),
    }
    if std is not None:
        out['predictions_std'] = std
    if std_epistemic is not None:
        out['predictions_std_epistemic'] = std_epistemic
    if std_aleatoric is not None:
        out['predictions_std_aleatoric'] = std_aleatoric
    io.savemat(os.path.join(OUTPUT_DIR, f'prediction_{name}_long.mat'), out)
    print(f"  saved long-table: prediction_{name}_long.mat  ({pred.shape})")


def reshape_to_4d(name, pred, std=None):
    depths_axis = np.sort(np.unique(depth_test))
    months_axis = np.arange(1, 13, dtype=np.int32)
    N_dep, N_mon = len(depths_axis), len(months_axis)
    expected = N_dep * N_mon

    df = pd.DataFrame({
        'lat': lat_test, 'lon': lon_test,
        'depth': depth_test, 'month': month_test.astype(int),
        'beta': pred[:, 0], 'biovol': pred[:, 1],
    })
    if std is not None:
        df['beta_std']   = std[:, 0]
        df['biovol_std'] = std[:, 1]

    counts    = df.groupby(['lat', 'lon']).size()
    n_complete = (counts == expected).sum()
    print(f"  [{name}] locations: {len(counts):,}; complete: {n_complete:,}")
    if n_complete == 0:
        print(f"  [{name}] no complete locations; skipping 4D save.")
        return

    complete = counts[counts == expected].index
    df_c = df.set_index(['lat', 'lon']).loc[complete].reset_index()
    df_c = df_c.sort_values(['lat', 'lon', 'depth', 'month']).reset_index(drop=True)
    N_loc = len(complete)

    beta_4d   = df_c['beta'  ].values.reshape(N_loc, N_dep, N_mon)
    biovol_4d = df_c['biovol'].values.reshape(N_loc, N_dep, N_mon)
    preds_4d  = np.stack([beta_4d, biovol_4d], axis=-1).astype(np.float32)
    loc_lat   = df_c.iloc[::expected]['lat'].values.astype(np.float32)
    loc_lon   = df_c.iloc[::expected]['lon'].values.astype(np.float32)

    out = {
        'predictions': preds_4d,
        'lat': loc_lat, 'lon': loc_lon,
        'depth': depths_axis.astype(np.float32),
        'month': months_axis,
        'target_names': np.array(TARGET_NAMES, dtype=object),
    }
    if std is not None:
        beta_std_4d   = df_c['beta_std'  ].values.reshape(N_loc, N_dep, N_mon)
        biovol_std_4d = df_c['biovol_std'].values.reshape(N_loc, N_dep, N_mon)
        out['predictions_std'] = np.stack(
            [beta_std_4d, biovol_std_4d], axis=-1).astype(np.float32)

    io.savemat(os.path.join(OUTPUT_DIR, f'prediction_{name}_4d.mat'), out)
    print(f"  saved 4D: prediction_{name}_4d.mat  shape={preds_4d.shape}")


save_long('rf_global',   rf_predictions)
reshape_to_4d('rf_global', rf_predictions)

save_long('svgp_global',   svgp_predictions,
          std=svgp_predictions_std,
          std_epistemic=svgp_std_epi.astype(np.float32),
          std_aleatoric=svgp_std_alea.astype(np.float32))
reshape_to_4d('svgp_global', svgp_predictions, std=svgp_predictions_std)

print("\n✅ All done.")
print(f"Output directory: {OUTPUT_DIR}")

# ── Delete checkpoint — run completed successfully ────────────────────────────
for _cp in [CKPT_PATH, CKPT_PATH + '.tmp']:
    if os.path.exists(_cp):
        os.remove(_cp)
print("[ckpt] Checkpoint deleted — run complete.", flush=True)
