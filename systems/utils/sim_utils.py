import numpy as np
import pandas as pd

def _common(u, v):
    "return indices where both u and v are non-NaN"
    mask = ~np.isnan(u) & ~np.isnan(v)
    return mask, u[mask], v[mask]

def pearson(u,v):
    mask, u_c, v_c = _common(u,v)
    if len(u_c) < 2: return 0.0
    u_c -= u_c.mean(); v_c -= v_c.mean()
    denom = np.sqrt((u_c**2).sum() * (v_c**2).sum())
    if denom == 0: return 0.0
    return (u_c * v_c).sum() / denom

def constrained_pearson(u, v, shrinkage = 10):
    "pearson with shrinkage towards zero when few co-ratings"
    mask, u_c, v_c = _common(u,v)
    n = len(u_c)
    if n < 2: return 0.0
    raw = pearson(u, v)
    return (n * raw) / (n + shrinkage)

def cosine(u, v):
    mask, u_c, v_c = _common(u,v)
    denom = np.linalg.norm(u_c) * np.linalg.norm(v_c)
    return (u_c @ v_c) / denom if denom else 0.0

def jaccard(u, v):
    mask_u = ~np.isnan(u); mask_v = ~np.isnan(v)
    intersection = np.sum(mask_u & mask_v)
    union = np.sum(mask_u | mask_v)
    return intersection / union if union else 0.0

def euclidean(u, v):
    mask, u_c, v_c = _common(u, v)
    if len(u_c) == 0: return 0.0
    dist = np.linalg.norm(u_c - v_c)
    # convert distance to similarity in [0,1]
    return 1 / (1 + dist)

def manhattan(u, v):
    mask, u_c, v_c = _common(u, v)
    if len(u_c) == 0: return 0.0
    dist = np.abs(u_c - v_c).sum()
    return 1 / (1 + dist)

def vectorized_cosine(df):
    mat = df.fillna(0).values
    norms = np.linalg.norm(mat, axis = 1)
    sim = mat.dot(mat.T) / (norms[:, None] * norms[None, :])
    return pd.DataFrame(sim, index=df.index, columns=df.index)

def vectorized_pearson(df):
    means = df.mean(axis=1)
    centered = df.sub(means, axis=0).fillna(0).values
    norms = np.linalg.norm(centered, axis=1)
    sim = centered.dot(centered.T) / (norms[:, None] * norms[None, :])
    return pd.DataFrame(sim, index=df.index, columns=df.index)
