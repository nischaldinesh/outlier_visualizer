from collections import Counter, defaultdict
from typing import Optional, Dict, Tuple, Iterable, List
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull


import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter

try:
    import alphashape
    from shapely.ops import unary_union
    from shapely.geometry import MultiPolygon, Polygon
except Exception:
    alphashape = None
    unary_union = None
    MultiPolygon = None
    Polygon = None

import importlib
try:
    detectors = importlib.import_module("detectors")
except Exception:
    detectors = None


# -------------------- Helper Functions --------------------

def auto_eps(X2d: np.ndarray, k: int = 10, q: float = 95.0) -> float:
    k = min(k, len(X2d) - 1) if len(X2d) > 1 else 1
    if k < 1:
        return 0.5
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X2d)
    dists, _ = nbrs.kneighbors(X2d)
    kth = dists[:, -1]
    eps = float(np.percentile(kth, q))
    if eps <= 0.0:
        eps = float(np.median(kth[kth > 0])) if np.any(kth > 0) else 0.5
    return eps


def clamp_perplexity(perp: float, n: int) -> float:
    if n <= 3:
        return 1.0
    upper = max(2.0, (n - 1) / 3.0)
    return float(np.clip(perp, 2.0, upper))


def tsne_embed(X_std: np.ndarray, random_state: int, perplexity: float, metric: str) -> np.ndarray:
    import inspect
    n = X_std.shape[0]
    perp = clamp_perplexity(perplexity, n)
    defaults = dict(
        n_components=2,
        perplexity=perp,
        learning_rate=300.0,
        early_exaggeration=12.0,
        init="pca",
        random_state=random_state,
        metric=metric,
        verbose=0,
    )
    supported = set(inspect.signature(TSNE.__init__).parameters.keys())
    kwargs = {k: v for k, v in defaults.items() if k in supported}
    if "angle" in supported:
        kwargs["angle"] = 0.5
        if "method" in supported:
            kwargs["method"] = "barnes_hut"
    if "n_iter" in supported:
        kwargs["n_iter"] = 1500
    tsne = TSNE(**kwargs)
    return tsne.fit_transform(X_std)


def convex_hull_poly(points: np.ndarray) -> np.ndarray:
    hull = ConvexHull(points)
    return points[hull.vertices]


def alpha_shape_polygon(points: np.ndarray, alpha: Optional[float] = None):
    if alphashape is None:
        raise ImportError("alphashape not available")
    import shapely.geometry as geom
    try:
        if alpha is None or (isinstance(alpha, float) and alpha == 0.0):
            poly = alphashape.alphashape(points, 0.0)
            if poly.is_empty:
                a = alphashape.optimizealpha(points)
                poly = alphashape.alphashape(points, a)
        else:
            poly = alphashape.alphashape(points, alpha)
        if poly.is_empty or poly.geom_type == "GeometryCollection":
            hull_pts = convex_hull_poly(points)
            return geom.Polygon(hull_pts)
        return poly
    except Exception:
        hull_pts = convex_hull_poly(points)
        return geom.Polygon(hull_pts)


def polygon_area_perimeter(poly) -> Tuple[float, float, int]:
    if MultiPolygon and isinstance(poly, MultiPolygon):
        poly = unary_union(poly)
    area = float(poly.area)
    perim = float(poly.length)
    holes = 0
    if getattr(poly, "geom_type", "") == "Polygon":
        holes = len(poly.interiors)
    elif getattr(poly, "geom_type", "") == "MultiPolygon":
        holes = sum(len(g.interiors) for g in poly.geoms)
    return area, perim, holes


def shape_metrics_2d(points: np.ndarray, alpha: Optional[float] = None) -> Dict[str, float]:
    try:
        poly_alpha = alpha_shape_polygon(points, alpha=alpha)
        area_a, perim_a, holes = polygon_area_perimeter(poly_alpha)
    except Exception:
        hull = ConvexHull(points)
        area_a = float(hull.volume)
        perim_pts = points[hull.vertices]
        perim_a = sum(float(np.linalg.norm(perim_pts[i] - perim_pts[(i + 1) % len(perim_pts)]))
                      for i in range(len(perim_pts)))
        holes = 0.0

    hull_pts = convex_hull_poly(points)
    x = hull_pts[:, 0]; y = hull_pts[:, 1]
    area_h = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    solidity = float(area_a / area_h) if area_h > 0 else 0.0
    compactness = float(4.0 * np.pi * area_a / (perim_a ** 2)) if perim_a > 0 else 0.0
    return {"solidity": solidity, "compactness": compactness,
            "area_alpha": area_a, "area_hull": area_h}


def aspect_ratio_from_cov(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 1.0
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    vals = np.clip(np.linalg.eigvals(cov).real, 1e-12, None)
    return float(np.sqrt(vals.max() / vals.min()))


def pca_var_ratio(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 1.0
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    vals = np.clip(np.linalg.eigvals(cov).real, 1e-12, None)
    vals_sorted = np.sort(vals)[::-1]
    return float(vals_sorted[0] / (vals_sorted.sum() + 1e-12))


def poly_r2(x: np.ndarray, y: np.ndarray, deg: int) -> float:
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    if len(x) < deg + 1 or np.isclose(np.var(y), 0.0):
        return 0.0
    try:
        coef = np.polyfit(x, y, deg=deg)
        yhat = np.polyval(coef, x)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
        return float(1.0 - ss_res / ss_tot)
    except Exception:
        try:
            coef = np.polyfit(y, x, deg=deg)
            xhat = np.polyval(coef, y)
            ss_res = float(np.sum((x - xhat) ** 2))
            ss_tot = float(np.sum((x - np.mean(x)) ** 2)) + 1e-12
            return float(1.0 - ss_res / ss_tot)
        except Exception:
            return 0.0


def classify_shape(solidity: float, aspect_ratio: float) -> str:
    if solidity < 0.85:
        return "concave/irregular"
    if aspect_ratio <= 1.5:
        return "spherical"
    if aspect_ratio <= 3.0:
        return "elliptical"
    return "elongated"


def classify_density(d: float) -> str:
    if d > 15:
        return "Extremely Dense"
    elif d > 8:
        return "Dense"
    elif d > 3:
        return "Moderate"
    elif d > 1:
        return "Sparse"
    else:
        return "Very Sparse"


# ---------------- NEW: Distribution typing ----------------
DIST_RANDOM = "Random"
DIST_LINEAR = "linear correlation"
DIST_OVERLAP = "overlap"
DIST_MANIFOLD = "manifolds types"

def classify_distribution_local(points: np.ndarray) -> str:
    if points.shape[0] < 3:
        return DIST_RANDOM

    x, y = points[:, 0], points[:, 1]
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(x, y)[0, 1]
    corr = float(0.0 if np.isnan(corr) else abs(corr))
    var1_ratio = pca_var_ratio(points)
    r2_lin = poly_r2(x, y, deg=1)
    r2_quad = poly_r2(x, y, deg=2)
    delta = r2_quad - r2_lin

    if corr >= 0.82 or var1_ratio >= 0.85 or r2_lin >= 0.85:
        return DIST_LINEAR
    if (delta >= 0.12 and r2_quad >= 0.80) and corr < 0.85:
        return DIST_MANIFOLD
    return DIST_RANDOM


def make_cluster_polygon(points: np.ndarray, alpha: Optional[float]) -> Optional[Polygon]:
    if Polygon is None:
        return None
    if alphashape is not None:
        try:
            poly = alpha_shape_polygon(points, alpha=alpha)
            return poly
        except Exception:
            pass
    try:
        order = ConvexHull(points).vertices
        return Polygon(points[order])
    except Exception:
        return None


def detect_overlaps(cluster_points: Dict[str, np.ndarray], alpha: Optional[float]) -> set:
    ids = sorted(cluster_points.keys(), key=lambda z: int(z))
    overlapped = set()
    polys: Dict[str, Optional[Polygon]] = {cid: make_cluster_polygon(cluster_points[cid], alpha) for cid in ids}
    areas: Dict[str, float] = {}
    for cid, p in polys.items():
        try:
            areas[cid] = float(p.area) if p is not None else 0.0
        except Exception:
            areas[cid] = 0.0

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ci, cj = ids[i], ids[j]
            pi, pj = polys[ci], polys[cj]

            overlapping = False
            if (pi is not None) and (pj is not None):
                try:
                    inter_area = pi.intersection(pj).area
                    denom = max(1e-9, min(areas[ci], areas[cj]))
                    frac = inter_area / denom
                    if frac >= 0.08:
                        overlapping = True
                except Exception:
                    overlapping = False

            if (pi is None or pj is None) and not overlapping:
                P_i = cluster_points[ci]; P_j = cluster_points[cj]
                c_i = P_i.mean(axis=0); c_j = P_j.mean(axis=0)
                r_i = float(np.median(np.linalg.norm(P_i - c_i, axis=1)) * 1.6)
                r_j = float(np.median(np.linalg.norm(P_j - c_j, axis=1)) * 1.6)
                d = float(np.linalg.norm(c_i - c_j))
                if d <= 0.90 * (r_i + r_j):
                    overlapping = True

            if overlapping:
                overlapped.add(ci); overlapped.add(cj)

    return overlapped


def build_groups(labels_array):
    groups = {}
    for i, cid in enumerate(labels_array):
        groups.setdefault(str(int(cid)), []).append(i)
    return groups


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    a = float(np.clip(alpha, 0.0, 1.0))
    return f"rgba({r},{g},{b},{a})"


def polygon_fill_traces_for_points(
    pts: np.ndarray,
    base_hex: str,
    fill_alpha: float,
    alpha_param: Optional[float] = None,
) -> List[go.Scatter]:
    traces: List[go.Scatter] = []
    if pts.shape[0] < 3:
        return traces

    poly = None
    if alphashape is not None and Polygon is not None:
        try:
            poly = alpha_shape_polygon(pts, alpha=alpha_param)
        except Exception:
            poly = None

    if poly is None or (hasattr(poly, "is_empty") and poly.is_empty):
        hull = ConvexHull(pts)
        order = hull.vertices
        x = pts[order, 0].tolist() + [pts[order[0], 0]]
        y = pts[order, 1].tolist() + [pts[order[0], 1]]
        traces.append(
            go.Scatter(
                x=x, y=y, mode="lines",
                fill="toself",
                fillcolor=hex_to_rgba(base_hex, fill_alpha),
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name="_region",
            )
        )
        return traces

    polys: Iterable[Polygon]
    if MultiPolygon and isinstance(poly, MultiPolygon):
        if unary_union is not None:
            poly = unary_union(poly)
        polys = getattr(poly, "geoms", [])
    else:
        polys = [poly]

    for p in polys:
        ext = p.exterior.coords.xy
        x = list(ext[0]); y = list(ext[1])
        traces.append(
            go.Scatter(
                x=x, y=y, mode="lines",
                fill="toself",
                fillcolor=hex_to_rgba(base_hex, fill_alpha),
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name="_region",
            )
        )
    return traces


# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="Comparative Analytical Framework", layout="wide")
st.title("Comparative Analytical Framework for Outlier Detection Algorithms")


if "shape_filters" not in st.session_state:
    st.session_state.shape_filters = []
if "density_filters" not in st.session_state:
    st.session_state.density_filters = []
if "distribution_filters" not in st.session_state:
    st.session_state.distribution_filters = []
if "show_explore" not in st.session_state:
    st.session_state.show_explore = False

with st.sidebar:
    st.header("Upload & Settings")
    file = st.file_uploader("Please upload a CSV file", type=["csv"])
    limit = st.number_input("Row limit (0 = all)", min_value=0, value=10000, step=1000)

    st.subheader("Visualization (t-SNE)")
    tsne_perp = st.slider("Perplexity", min_value=2, max_value=100, value=30, step=1)
    tsne_metric = st.selectbox("Distance Metric", ["euclidean", "cosine", "manhattan"], index=0)
    seed = st.number_input("Random seed", value=42)

    st.subheader("Clustering (DBSCAN)")
    db_min_samples = st.slider("DBSCAN min_samples", min_value=3, max_value=50, value=10)
    db_eps = st.number_input("DBSCAN eps (0=auto)", min_value=0.0, value=0.0, step=0.05)

    st.subheader("α-shape")
    alpha_val = st.number_input("Alpha (empty=auto)", value=0.0, step=0.1, format="%.1f")
    alpha = None if alpha_val == 0.0 else float(alpha_val)

    st.subheader("Shading")
    shade_alpha = st.slider("Shading opacity", min_value=0.0, max_value=0.6, value=0.22, step=0.02,
                            help="Opacity of shaded region for clusters matching selected characteristics.")

if file is None:
    st.stop()

if "file_signature" not in st.session_state:
    st.session_state.file_signature = None

raw = pd.read_csv(file)

signature = (getattr(file, "name", None), getattr(file, "size", None))
if st.session_state.file_signature != signature:
    st.session_state.file_signature = signature
    st.session_state.shape_filters = []
    st.session_state.density_filters = []
    st.session_state.distribution_filters = []
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and (
            key.startswith("shape-filter-")
            or key.startswith("density-filter-")
            or key.startswith("distribution-filter-")
        ):
            del st.session_state[key]

if limit and limit > 0:
    raw = raw.head(limit)

X = raw.select_dtypes(include=[np.number]).copy()
keep_mask = ~X.isna().any(axis=1)
X = X.loc[keep_mask]

if X.shape[0] < 3 or X.shape[1] < 1:
    st.error("Not enough usable numeric data after cleaning.")
    st.stop()

# ---------------- Determine cluster labels ----------------
label_candidates = ["label", "labels", "class", "Class", "target", "y", "digit"]
label_col = next((c for c in label_candidates if c in raw.columns), None)

labels_used: np.ndarray
if label_col is not None:
    provided = raw.loc[keep_mask, label_col]
    if pd.api.types.is_numeric_dtype(provided):
        codes = provided.astype("Int64").fillna(-1).astype(int).to_numpy()
        labels_used = codes
    else:
        codes, _ = pd.factorize(provided.astype("string"), sort=True)
        labels_used = codes
else:
    scaler = StandardScaler()
    X_std_tmp = scaler.fit_transform(X.values)
    X2d_tmp = tsne_embed(X_std=X_std_tmp, random_state=int(seed), perplexity=float(tsne_perp), metric=tsne_metric)
    eps_val = None if db_eps <= 0.0 else float(db_eps)
    if eps_val is None:
        eps_val = auto_eps(X2d_tmp, k=10, q=95.0)
    labels_used = DBSCAN(eps=eps_val, min_samples=int(db_min_samples)).fit_predict(X2d_tmp).astype(int)

# Compute t-SNE (final)
scaler = StandardScaler()
X_std = scaler.fit_transform(X.values)
X2d = tsne_embed(X_std=X_std, random_state=int(seed), perplexity=float(tsne_perp), metric=tsne_metric)

# -------- RELABEL to 1-based positive IDs --------
uniq_order = sorted(pd.unique(labels_used))
if -1 in uniq_order:
    uniq_order = [l for l in uniq_order if l != -1] + [-1]
id_map = {old: i for i, old in enumerate(uniq_order, start=1)}
labels_relabeled = np.array([id_map[int(l)] for l in labels_used], dtype=int)

clusters = build_groups(labels_relabeled)
if not clusters:
    st.warning("No clusters found.")
    st.stop()

# ---------------- Per-cluster analysis ----------------
rows = []
cluster_points = {}
for cid, idxs in clusters.items():
    pts = X2d[np.array(idxs)]
    cluster_points[str(int(cid))] = pts

    npts = len(pts)
    if npts < 3:
        shape = "too-small"
        density_val = 0.0
        dist_type = DIST_RANDOM
    else:
        ar = aspect_ratio_from_cov(pts)
        metrics = shape_metrics_2d(pts, alpha=alpha)
        shape = classify_shape(metrics["solidity"], ar)
        area = metrics["area_alpha"]
        density_val = npts / area if area > 0 else 0.0
        dist_type = classify_distribution_local(pts)

    rows.append({
        "cluster_id": int(cid),
        "shape": shape,
        "density_val": density_val,
        "density_label": classify_density(density_val),
        "distribution": dist_type
    })

# ---------------- Overall dataset analysis (full X2d) ----------------
overall = {
    "shape": "N/A",
    "solidity": 0.0,
    "aspect_ratio": 1.0,
    "density_val": 0.0,
    "density_label": "N/A",
    "distribution": DIST_RANDOM,
}
all_pts = X2d
n_all = all_pts.shape[0]
if n_all >= 3:
    ar_all = aspect_ratio_from_cov(all_pts)
    metrics_all = shape_metrics_2d(all_pts, alpha=alpha)
    shape_all = classify_shape(metrics_all["solidity"], ar_all)
    area_all = metrics_all["area_alpha"]
    density_all = n_all / area_all if area_all > 0 else 0.0

    global_dist = classify_distribution_local(all_pts)

    overall.update({
        "shape": shape_all,
        "solidity": float(metrics_all["solidity"]),
        "aspect_ratio": float(ar_all),
        "density_val": float(density_all),
        "density_label": classify_density(density_all),
        "distribution": global_dist,
    })

# ---------------- Detect overlaps & override overall if needed ----------------
overlap_ids = detect_overlaps(cluster_points, alpha=alpha)
if overlap_ids:
    for r in rows:
        if str(int(r["cluster_id"])) in overlap_ids:
            r["distribution"] = DIST_OVERLAP
    overall["distribution"] = DIST_OVERLAP

# ------------------ Top Layout: Left (t-SNE) / Right (Details & Explore) ------------------
left, right = st.columns([2, 1.2])

with right:
    st.subheader("Data Characteristics")
    st.markdown(f"- **Dataset size (after clean):** {X.shape[0]}")

    
    shape_counts = Counter(r["shape"] for r in rows)
    shape_summary = sorted(shape_counts.items(), key=lambda x: (-x[1], x[0]))

    density_counts = Counter(r["density_label"] for r in rows)
    density_summary = sorted(density_counts.items(), key=lambda x: (-x[1], x[0]))

 
    col_shapes, col_dens = st.columns(2)

    # ---------- SHAPES ----------
    with col_shapes:
        st.markdown("- **Cluster Shapes:**")
        for i, (shape, count) in enumerate(shape_summary, 1):
            key = f"shape-filter-{shape}"
            if key not in st.session_state:
                st.session_state[key] = shape in st.session_state.shape_filters
            cols = st.columns([0.8, 0.2])
            with cols[0]:
                st.markdown(f"{i}. **{shape.capitalize()}** ({count})")
            with cols[1]:
                st.checkbox("select", key=key, label_visibility="collapsed")
        st.session_state.shape_filters = [
            shape for shape, _ in shape_summary if st.session_state.get(f"shape-filter-{shape}", False)
        ]
        with st.expander("per-cluster details", expanded=False):
            for r in sorted(rows, key=lambda x: int(x["cluster_id"])):
                st.markdown(f"- Cluster **{r['cluster_id']}** : {r['shape']}")
        if n_all >= 3:
            st.markdown(f"_Overall shape:_ **{overall['shape'].capitalize()}** ")

    # ---------- DENSITIES ----------
    with col_dens:
        st.markdown("- **Cluster Densities:**")
        for i, (dlabel, count) in enumerate(density_summary, 1):
            key = f"density-filter-{dlabel}"
            if key not in st.session_state:
                st.session_state[key] = dlabel in st.session_state.density_filters
            cols = st.columns([0.8, 0.2])
            with cols[0]:
                st.markdown(f"{i}. **{dlabel}** ({count})")
            with cols[1]:
                st.checkbox("select", key=key, label_visibility="collapsed")
        st.session_state.density_filters = [
            dlabel for dlabel, _ in density_summary if st.session_state.get(f"density-filter-{dlabel}", False)
        ]
        with st.expander("per-cluster details", expanded=False):
            for r in sorted(rows, key=lambda x: int(x["cluster_id"])):
                st.markdown(f"- Cluster **{r['cluster_id']}** : {r['density_label']}")
        if n_all >= 3:
            st.markdown(f"_Overall density:_ **{overall['density_label']}** ")

    
    dist_counts = Counter(r["distribution"] for r in rows)
    dist_summary = sorted(dist_counts.items(), key=lambda x: (-x[1], x[0]))
    st.markdown("- **Cluster Distributions:**")
    for i, (dist, count) in enumerate(dist_summary, 1):
        key = f"distribution-filter-{dist}"
        if key not in st.session_state:
            st.session_state[key] = dist in st.session_state.distribution_filters
        cols = st.columns([0.8, 0.2])
        with cols[0]:
            st.markdown(f"{i}. **{dist}** ({count})")
        with cols[1]:
            st.checkbox("select", key=key, label_visibility="collapsed")
    st.session_state.distribution_filters = [
        dist for dist, _ in dist_summary if st.session_state.get(f"distribution-filter-{dist}", False)
    ]
    with st.expander("per-cluster details", expanded=False):
        for r in sorted(rows, key=lambda x: int(x["cluster_id"])):
            st.markdown(f"- Cluster **{r['cluster_id']}** : {r['distribution']}")
    if n_all >= 3:
        st.markdown(f"_Overall distribution:_ **{overall['distribution']}** ")

    # -------- Explore --------
    # if st.button("Explore"):
    #     st.session_state.show_explore = True

    if st.session_state.show_explore:
        st.subheader("Explore Outlier Detectors")
        if detectors is None:
            st.error("Missing `detectors.py`. Please place it alongside main.py.")
        else:
            heat_tech = st.radio(
                "Heatmap Technique",
                ["Raw", "Threshold", "Interpolated", "Binary", "Ranked"],
                index=0,
            )
            cmap_choice = st.selectbox(
                "Colormap (Inlier → Outlier)",
                ["viridis", "plasma", "terrain", "coolwarm", "turbo", "cividis"],
                index=0,
            )
            st.session_state["_explore_opts"] = {
                "heat_tech": heat_tech,
                "colormap": cmap_choice,
            }

# ---------------- Left panel: t-SNE scatter ----------------
with left:
    PALETTE = [
        "#0B3D91","#1E3A8A","#003F5C","#005F73","#083344","#14532D","#1B5E20",
        "#3D550C","#6A040F","#8B0000","#7C2D12","#4E342E","#4A148C","#3B0764",
        "#880E4F","#1F2937","#000000",
    ]
    labels_as_str = pd.Series(labels_relabeled).astype(str).values
    uniq = sorted(pd.unique(labels_as_str))
    color_map = {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(uniq)}

    shape_filters = set(st.session_state.get("shape_filters", []))
    density_filters = set(st.session_state.get("density_filters", []))
    distribution_filters = set(st.session_state.get("distribution_filters", []))
    filters_active = bool(shape_filters or density_filters or distribution_filters)

    info_by_lab = {str(r["cluster_id"]): r for r in rows}

    def matches_filters(lab: str) -> bool:
        if not filters_active:
            return False
        r = info_by_lab.get(lab)
        if r is None:
            return False
        conds = []
        if shape_filters:
            conds.append(r["shape"] in shape_filters)
        if density_filters:
            conds.append(r["density_label"] in density_filters)
        if distribution_filters:
            conds.append(r["distribution"] in distribution_filters)
        return any(conds) if conds else False

    fig = go.Figure()

    for lab in sorted(uniq, key=lambda x: int(x) if x.isdigit() else x):
        if matches_filters(lab):
            pts = X2d[labels_as_str == lab]
            if pts.shape[0] >= 3:
                for t in polygon_fill_traces_for_points(
                    pts=pts,
                    base_hex=color_map[lab],
                    fill_alpha=float(shade_alpha),
                    alpha_param=alpha,
                ):
                    t.update(legendgroup=f"cluster-{lab}", showlegend=False, hoverinfo="skip")
                    fig.add_trace(t)

    for lab in sorted(uniq, key=lambda x: int(x) if x.isdigit() else x):
        pts = X2d[labels_as_str == lab]
        marker = dict(size=7 if matches_filters(lab) else 5,
                      opacity=1.0 if matches_filters(lab) else 0.9,
                      color=color_map[lab],
                      line=dict(width=1, color="#FFFFFF") if matches_filters(lab) else None)
        fig.add_trace(
            go.Scattergl(
                x=pts[:, 0], y=pts[:, 1],
                mode="markers",
                name="",
                showlegend=False,
                marker=marker,
                legendgroup=f"cluster-{lab}",
            )
        )

    title_suffix = " — Highlighting selections" if filters_active else ""
    fig.update_layout(
        title=f"Clusters (t-SNE){title_suffix}",
        showlegend=False,
        height=760,
        margin=dict(t=40, r=10, b=10, l=10),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    st.plotly_chart(fig, use_container_width=True)

    if filters_active:
        parts = []
        if shape_filters:
            parts.append("shape: " + ", ".join(sorted(s.capitalize() for s in shape_filters)))
        if density_filters:
            parts.append("density: " + ", ".join(sorted(d for d in density_filters)))
        if distribution_filters:
            parts.append("distribution: " + ", ".join(sorted(d for d in distribution_filters)))
        st.caption("Shaded = matches → " + " | ".join(parts))
    else:
        st.caption("No selections — showing all clusters (no shading).")

# ------------------ FULL-WIDTH SECTION: Algorithm Plots (2×2) ------------------
if st.session_state.get("show_explore") and detectors is not None and st.session_state.get("_explore_opts"):
    try:
        detectors.ensure_pyod()
    except ImportError as e:
        st.error(str(e))
    else:
        st.divider()
        st.subheader("Algorithm Heatmaps (Inlier → Outlier)")

        def render_heatmap_image(scores: np.ndarray, title: str):
            s = scores.astype(float)
            s = (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-8)

            method = st.session_state["_explore_opts"]["heat_tech"].lower()
            if method == "threshold":
                s = (s >= np.percentile(s, 95)).astype(float)
            elif method == "binary":
                s = (s >= 0.5).astype(float)
            elif method == "ranked":
                order = np.argsort(s)
                ranks = np.empty_like(order, dtype=float)
                ranks[order] = np.linspace(0, 1, num=len(s), endpoint=True)
                s = ranks

            xlim = (X2d[:, 0].min(), X2d[:, 0].max())
            ylim = (X2d[:, 1].min(), X2d[:, 1].max())
            gx = np.linspace(xlim[0], xlim[1], 60)
            gy = np.linspace(ylim[0], ylim[1], 60)
            xx, yy = np.meshgrid(gx, gy)

            interpolator = RBFInterpolator(X2d, s, neighbors=20, smoothing=0.1)
            zz = interpolator(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            if method in ["threshold", "interpolated", "ranked"]:
                zz = gaussian_filter(zz, sigma=1.5)

            cmap = st.session_state["_explore_opts"]["colormap"]

            fig, ax = plt.subplots(figsize=(4.4, 3.4), dpi=120)
            cset = ax.contourf(xx, yy, zz, levels=60, cmap=cmap)
            ax.scatter(X2d[:, 0], X2d[:, 1], s=4, c="black", alpha=0.55, edgecolors="none")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=12, color="black")
            for spine in ax.spines.values():
                spine.set_visible(False)

            cb = fig.colorbar(cset, ax=ax, fraction=0.046, pad=0.04)
            cb.set_ticks([]); cb.ax.tick_params(length=0, labelsize=0)
            cb.set_label("likelihood (inlier → outlier)", fontsize=8)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        algos = ["CBLOF", "HBOS", "MCD", "OCSVM"]
        rows_cols = [st.columns(2), st.columns(2)]

        for i, algo in enumerate(algos):
            det = detectors.get_detector(algo, random_state=int(seed))
            try:
                scores = detectors.fit_and_score(det, X_std)
            except Exception as e:
                target_col = rows_cols[0][i] if i < 2 else rows_cols[1][i - 2]
                with target_col:
                    st.error(f"{algo} failed: {e}")
                continue

            target_col = rows_cols[0][i] if i < 2 else rows_cols[1][i - 2]
            with target_col:
                render_heatmap_image(scores, f"{algo} — {st.session_state['_explore_opts']['heat_tech']}")

