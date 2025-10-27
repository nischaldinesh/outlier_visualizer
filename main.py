from collections import Counter
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
from scipy.stats import skew, kurtosis, entropy

try:
    import alphashape
    from shapely.ops import unary_union
    from shapely.geometry import MultiPolygon
except Exception:
    alphashape = None
    unary_union = None
    MultiPolygon = None


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
        if alpha is None:
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
    if poly.geom_type == "Polygon":
        holes = len(poly.interiors)
    elif poly.geom_type == "MultiPolygon":
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
    x = hull_pts[:, 0]
    y = hull_pts[:, 1]
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


def classify_shape(solidity: float, aspect_ratio: float) -> str:
    if solidity < 0.85:
        return "concave/irregular"
    if aspect_ratio <= 1.5:
        return "spherical"
    if aspect_ratio <= 3.0:
        return "elliptical"
    return "elongated"


def classify_density(d):
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


def classify_distribution_type(skewness, kurt, entropy_val):
    if entropy_val > 3.0 and abs(skewness) < 0.5 and abs(kurt - 3) < 1:
        return "Uniform"
    elif abs(skewness) < 0.3 and 2.5 <= kurt <= 3.5:
        return "Normal"
    elif abs(skewness) > 0.5:
        return "Skewed"
    elif 2.0 <= kurt <= 6.0 and entropy_val < 3.0:
        return "Multimodal"
    elif kurt > 4.5:
        return "Heavy-Tailed"
    elif entropy_val < 2.5:
        return "Clustered"
    else:
        return "Normal"


def build_groups(labels_array):
    """Include ALL labels; do not skip anything."""
    groups = {}
    for i, cid in enumerate(labels_array):
        groups.setdefault(str(int(cid)), []).append(i)
    return groups


# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="Comparative Analytical Framework", layout="wide")
st.title("Comparative Analytical Framework for Outlier Detection Algorithms")

status = st.empty()
if "status_msg" not in st.session_state:
    st.session_state.status_msg = "Upload a CSV to begin."
status.info(st.session_state.status_msg)

if "selected_cluster" not in st.session_state:
    st.session_state.selected_cluster = None
if "shape_filters" not in st.session_state:
    st.session_state.shape_filters = []
if "density_filters" not in st.session_state:
    st.session_state.density_filters = []
if "distribution_filters" not in st.session_state:
    st.session_state.distribution_filters = []

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


selected_dist = "All"

if file is None:
    st.stop()

if "file_signature" not in st.session_state:
    st.session_state.file_signature = None

st.session_state.status_msg = "Processing..."
status.warning(st.session_state.status_msg)

raw = pd.read_csv(file)

signature = (getattr(file, "name", None), getattr(file, "size", None))
if st.session_state.file_signature != signature:
    st.session_state.selected_cluster = None
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
    status.error("Not enough usable numeric data after cleaning.")
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

# Compute t-SNE (final, for plotting & analytics)
scaler = StandardScaler()
X_std = scaler.fit_transform(X.values)
X2d = tsne_embed(X_std=X_std, random_state=int(seed), perplexity=float(tsne_perp), metric=tsne_metric)

# -------- RELABEL to 1-based positive IDs for uniform plotting --------
uniq_order = sorted(pd.unique(labels_used))
if -1 in uniq_order:
    uniq_order = [l for l in uniq_order if l != -1] + [-1]
id_map = {old: i for i, old in enumerate(uniq_order, start=1)}
labels_relabeled = np.array([id_map[int(l)] for l in labels_used], dtype=int)

clusters = build_groups(labels_relabeled)
if not clusters:
    status.warning("No clusters found.")
    st.stop()

selected_cluster = st.session_state.get("selected_cluster")
if selected_cluster and selected_cluster not in clusters:
    st.session_state.selected_cluster = None
    selected_cluster = None

# ---------------- Per-cluster analysis ----------------
rows = []
for cid, idxs in clusters.items():
    pts = X2d[np.array(idxs)]
    npts = len(pts)
    if npts < 3:
        shape = "too-small"
        density_val = 0.0
        dist_type = "N/A"
    else:
        ar = aspect_ratio_from_cov(pts)
        metrics = shape_metrics_2d(pts, alpha=alpha)
        shape = classify_shape(metrics["solidity"], ar)
        area = metrics["area_alpha"]
        density_val = npts / area if area > 0 else 0.0

        sk = skew(pts[:, 0]) + skew(pts[:, 1])
        kurt_val = kurtosis(pts[:, 0]) + kurtosis(pts[:, 1])
        hist, _ = np.histogram(pts[:, 0], bins=10, density=True)
        p = hist / np.sum(hist) if np.sum(hist) > 0 else np.full_like(hist, 1 / len(hist), dtype=float)
        ent = entropy(p, base=2)
        dist_type = classify_distribution_type(sk, kurt_val, ent)

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
    "distribution": "N/A",
}
all_pts = X2d
n_all = all_pts.shape[0]
if n_all >= 3:
    ar_all = aspect_ratio_from_cov(all_pts)
    metrics_all = shape_metrics_2d(all_pts, alpha=alpha)
    shape_all = classify_shape(metrics_all["solidity"], ar_all)
    area_all = metrics_all["area_alpha"]
    density_all = n_all / area_all if area_all > 0 else 0.0

    sk_all = skew(all_pts[:, 0]) + skew(all_pts[:, 1])
    kurt_all = kurtosis(all_pts[:, 0]) + kurtosis(all_pts[:, 1])
    hist_all, _ = np.histogram(all_pts[:, 0], bins=10, density=True)
    p_all = hist_all / np.sum(hist_all) if np.sum(hist_all) > 0 else np.full_like(hist_all, 1 / len(hist_all), dtype=float)
    ent_all = entropy(p_all, base=2)
    dist_all = classify_distribution_type(sk_all, kurt_all, ent_all)

    overall.update({
        "shape": shape_all,
        "solidity": float(metrics_all["solidity"]),
        "aspect_ratio": float(ar_all),
        "density_val": float(density_all),
        "density_label": classify_density(density_all),
        "distribution": dist_all,
    })

st.session_state.status_msg = "Successful."
status.success(st.session_state.status_msg)

left, right = st.columns([2, 1.2])

# ------------------ Right Panel (Interactive Summaries) ------------------
with right:
    st.subheader("Data Characteristics")
    st.markdown(f"- **Dataset size (after clean):** {X.shape[0]}")
    st.markdown("\n")

    # ---- Shapes ----
    shape_counts = Counter(r["shape"] for r in rows)
    shape_summary = sorted(shape_counts.items(), key=lambda x: (-x[1], x[0]))

    st.markdown("- **Cluster Shapes:**")
    for i, (shape, count) in enumerate(shape_summary, 1):
        key = f"shape-filter-{shape}"
        if key not in st.session_state:
            st.session_state[key] = shape in st.session_state.shape_filters
        cols = st.columns([0.8, 0.2])
        with cols[0]:
            st.markdown(f"{i}. **{shape.capitalize()}** ({count})")
        with cols[1]:
            st.checkbox("", key=key)
    st.session_state.shape_filters = [
        shape for shape, _ in shape_summary if st.session_state.get(f"shape-filter-{shape}", False)
    ]

    with st.expander("Show per-cluster shape assignments"):
        highlighted = st.session_state.get("selected_cluster")
        for r in sorted(rows, key=lambda x: int(x["cluster_id"])):
            lab = str(r["cluster_id"])
            label_text = f"Cluster {lab} -> {r['shape']}"
            button_type = "primary" if highlighted == lab else "secondary"
            if st.button(label_text, key=f"shape-{lab}", type=button_type):
                st.session_state.selected_cluster = None if highlighted == lab else lab
                highlighted = st.session_state.selected_cluster

    if n_all >= 3:
        st.markdown(
            f"_Overall shape:_ **{overall['shape'].capitalize()}** "
            # f"(solidity: {overall['solidity']:.2f}, aspect ratio: {overall['aspect_ratio']:.2f})"
        )
    else:
        st.markdown("_Overall shape:_ *Not enough points to compute reliably.*")

    st.markdown("---")

    # ---- Densities ----
    density_counts = Counter(r["density_label"] for r in rows)
    density_summary = sorted(density_counts.items(), key=lambda x: (-x[1], x[0]))

    st.markdown("- **Cluster Densities:**")
    for i, (dlabel, count) in enumerate(density_summary, 1):
        key = f"density-filter-{dlabel}"
        if key not in st.session_state:
            st.session_state[key] = dlabel in st.session_state.density_filters
        cols = st.columns([0.8, 0.2])
        with cols[0]:
            st.markdown(f"{i}. **{dlabel}** ({count})")
        with cols[1]:
            st.checkbox("", key=key)
    st.session_state.density_filters = [
        dlabel for dlabel, _ in density_summary if st.session_state.get(f"density-filter-{dlabel}", False)
    ]

    with st.expander("Show per-cluster density assignments"):
        highlighted = st.session_state.get("selected_cluster")
        for r in sorted(rows, key=lambda x: int(x["cluster_id"])):
            lab = str(r["cluster_id"])
            label_text = (
                f"Cluster {lab} -> {r['density_label']} "
                f"(density {r['density_val']:.2f})"
            )
            button_type = "primary" if highlighted == lab else "secondary"
            if st.button(label_text, key=f"density-{lab}", type=button_type):
                st.session_state.selected_cluster = None if highlighted == lab else lab
                highlighted = st.session_state.selected_cluster

    if n_all >= 3:
        st.markdown(
            f"_Overall density:_ **{overall['density_label']}** "
            # f"(n/area: {overall['density_val']:.2f})"
        )
    else:
        st.markdown("_Overall density:_ *Not enough points to compute reliably.*")

    st.markdown("---")

    # ---- Distributions ----
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
            st.checkbox("", key=key)
    st.session_state.distribution_filters = [
        dist for dist, _ in dist_summary if st.session_state.get(f"distribution-filter-{dist}", False)
    ]

    with st.expander("Show per-cluster distribution assignments"):
        highlighted = st.session_state.get("selected_cluster")
        for r in sorted(rows, key=lambda x: int(x["cluster_id"])):
            lab = str(r["cluster_id"])
            label_text = f"Cluster {lab} -> {r['distribution']}"
            button_type = "primary" if highlighted == lab else "secondary"
            if st.button(label_text, key=f"dist-{lab}", type=button_type):
                st.session_state.selected_cluster = None if highlighted == lab else lab
                highlighted = st.session_state.selected_cluster

    if n_all >= 3:
        st.markdown(f"_Overall distribution:_ **{overall['distribution']}**")
    else:
        st.markdown("_Overall distribution:_ *Not enough points to compute reliably.*")

# ------------------ Plotting Section ------------------
with left:
    if not any(r["distribution"] == selected_dist for r in rows) and selected_dist != "All":
        st.warning(f"No clusters found with '{selected_dist}' distribution.")
        st.stop()

    PALETTE = [
        "#0072B2", "#E69F00", "#009E73", "#D55E00",
        "#56B4E9", "#F0E442", "#CC79A7", "#000000",
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
        "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
        "#9C755F", "#BAB0AC",
    ]

    labels_as_str = pd.Series(labels_relabeled).astype(str).values
    uniq = sorted(pd.unique(labels_as_str))
    color_map = {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(uniq)}
    highlighted = st.session_state.get("selected_cluster")
    shape_filters = set(st.session_state.get("shape_filters", []))
    density_filters = set(st.session_state.get("density_filters", []))
    distribution_filters = set(st.session_state.get("distribution_filters", []))
    filters_active = bool(shape_filters or density_filters or distribution_filters)

    def matches_active_filters(cluster_info) -> bool:
        if shape_filters and cluster_info["shape"] not in shape_filters:
            return False
        if density_filters and cluster_info["density_label"] not in density_filters:
            return False
        if distribution_filters and cluster_info["distribution"] not in distribution_filters:
            return False
        return True

    fig = go.Figure()
    for r in sorted(rows, key=lambda x: int(x["cluster_id"])):
        if selected_dist != "All" and r["distribution"] != selected_dist:
            continue
        lab = str(r["cluster_id"])
        mask = labels_as_str == lab
        pts = X2d[mask]
        if highlighted is None and filters_active:
            if matches_active_filters(r):
                marker = dict(
                    size=7,
                    opacity=1.0,
                    color=color_map[lab],
                    line=dict(width=1.5, color="#FFFFFF"),
                )
            else:
                marker = dict(size=4, opacity=0.2, color=color_map[lab])
        elif highlighted is None:
            marker = dict(size=5, opacity=0.9, color=color_map[lab])
        elif highlighted == lab:
            marker = dict(
                size=7,
                opacity=1.0,
                color=color_map[lab],
                line=dict(width=1.5, color="#FFFFFF"),
            )
        else:
            marker = dict(size=4, opacity=0.2, color=color_map[lab])
        fig.add_trace(go.Scattergl(
            x=pts[:, 0], y=pts[:, 1],
            mode="markers",
            name=f"Cluster {lab}",
            marker=marker,
        ))
    title = f"Clusters with '{selected_dist}' Distribution" if selected_dist != "All" else "All Clusters"
    fig.update_layout(
        title=title,
        xaxis_title="t-SNE-1",
        yaxis_title="t-SNE-2",
        height=700,
        legend_title="Cluster ID",
        margin=dict(t=40, r=10, b=10, l=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    if highlighted:
        st.caption(f"Highlighting cluster {highlighted}. Click again to clear.")
    elif filters_active:
        parts = []
        if shape_filters:
            parts.append("shape: " + ", ".join(sorted(s.capitalize() for s in shape_filters)))
        if density_filters:
            parts.append("density: " + ", ".join(sorted(d for d in density_filters)))
        if distribution_filters:
            parts.append("distribution: " + ", ".join(sorted(d for d in distribution_filters)))
        st.caption(f"Highlighting clusters matching {'; '.join(parts)}.")
    else:
        st.caption("Use the cluster lists on the right to highlight a group.")
