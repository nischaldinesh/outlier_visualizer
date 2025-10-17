from collections import Counter
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull

try:
    import alphashape
    from shapely.ops import unary_union
    from shapely.geometry import MultiPolygon
except Exception:
    alphashape = None
    unary_union = None
    MultiPolygon = None


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

def tsne_embed(
    X_std: np.ndarray,
    random_state: int,
    perplexity: float,
    metric: str,
) -> np.ndarray:
    import inspect
    n = X_std.shape[0]
    perp = clamp_perplexity(perplexity, n)
    if perp != perplexity:
        st.info(f"Perplexity adjusted from {perplexity} → {perp:.1f} for n={n}")

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
    if "n_iter" not in supported and hasattr(tsne, "n_iter"):
        try:
            tsne.n_iter = 1500
        except Exception:
            pass
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
        perim_a = 0.0
        for i in range(len(perim_pts)):
            a = perim_pts[i]; b = perim_pts[(i + 1) % len(perim_pts)]
            perim_a += float(np.linalg.norm(a - b))
        holes = 0.0

    hull_pts = convex_hull_poly(points)
    x = hull_pts[:, 0]; y = hull_pts[:, 1]
    area_h = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    hull_perim = 0.0
    for i in range(len(hull_pts)):
        a = hull_pts[i]; b = hull_pts[(i + 1) % len(hull_pts)]
        hull_perim += float(np.linalg.norm(a - b))

    solidity = float(area_a / area_h) if area_h > 0 else 0.0
    compactness = float(4.0 * np.pi * area_a / (perim_a ** 2)) if perim_a > 0 else 0.0
    return {
        "solidity": float(solidity),
        "compactness": float(compactness),
        "holes": float(holes),
        "area_alpha": float(area_a),
        "perim_alpha": float(perim_a),
        "area_hull": float(area_h),
        "perim_hull": float(hull_perim),
    }

def aspect_ratio_from_cov(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 1.0
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    vals = np.linalg.eigvals(cov).real
    vals = np.clip(vals, 1e-12, None)
    return float(np.sqrt(vals.max() / vals.min()))

def classify_shape(solidity: float, aspect_ratio: float) -> str:
    if solidity < 0.85:
        return "concave/irregular"
    if aspect_ratio <= 1.5:
        return "spherical"
    if aspect_ratio <= 3.0:
        return "elliptical"
    return "elongated"

def build_groups(labels_array):
    groups = {}
    for i, cid in enumerate(labels_array):
        if isinstance(cid, (np.integer, int)) and int(cid) == -1:
            continue  # skip DBSCAN noise
        groups.setdefault(str(cid), []).append(i)
    return groups



st.set_page_config(page_title="Comparative Analytical Framework", layout="wide")


st.markdown("""
<style>
main .block-container { padding-top: 0.75rem; padding-bottom: 2rem; }
[data-testid="stSidebar"] .block-container { padding-top: 0.75rem; }
h1, h2, h3 { margin-top: 0.25rem; }
</style>
""", unsafe_allow_html=True)

st.title("Comparative Analytical Framework for Outlier Detection Algorithms")


status = st.empty()
if "status_msg" not in st.session_state:
    st.session_state.status_msg = "Upload a CSV to begin."
status.info(st.session_state.status_msg)

with st.sidebar:
    st.header("Upload & Settings")
    file = st.file_uploader("Please uplaod a CSV file", type=["csv"])
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


if file is None:
    st.stop()


st.session_state.status_msg = "Processing..."
status.warning(st.session_state.status_msg)


raw = pd.read_csv(file)
if limit and limit > 0:
    raw = raw.head(limit)

X = raw.select_dtypes(include=[np.number]).copy()
keep_mask = ~X.isna().any(axis=1)
X = X.loc[keep_mask]

if X.shape[0] < 3 or X.shape[1] < 1:
    status.error("Not enough usable numeric data after cleaning.")
    st.stop()


scaler = StandardScaler()
X_std = scaler.fit_transform(X.values)


X2d = tsne_embed(
    X_std=X_std,
    random_state=int(seed),
    perplexity=float(tsne_perp),
    metric=tsne_metric,
)

eps_val = None if db_eps <= 0.0 else float(db_eps)
if eps_val is None:
    eps_val = auto_eps(X2d, k=10, q=95.0)
    # st.info(f"Auto-tuned DBSCAN eps = {eps_val:.3f}")
labels_used = DBSCAN(eps=eps_val, min_samples=int(db_min_samples)).fit_predict(X2d)


clusters = build_groups(labels_used)
if not clusters:
    status.warning("No clusters found (after excluding DBSCAN noise). Try adjusting DBSCAN or perplexity.")
    st.stop()

rows = []
for cid, idxs in clusters.items():
    pts = X2d[np.array(idxs)]
    npts = len(pts)
    if npts < 3:
        shape = "too-small"
    else:
        ar = aspect_ratio_from_cov(pts)
        metrics = shape_metrics_2d(pts, alpha=alpha)
        shape = classify_shape(metrics["solidity"], ar)
    rows.append({"cluster_id": cid, "shape": shape})


st.session_state.status_msg = "Successful."
status.success(st.session_state.status_msg)


left, right = st.columns([2, 1.2])

with left:
    PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#56B4E9", "#F0E442", "#CC79A7", "#000000",
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
    "#9C755F", "#BAB0AC",
    "#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072",
    "#80B1D3", "#FDB462", "#B3DE69", "#FCCDE5",
    "#D9D9D9", "#BC80BD", "#CCEBC5", "#FFED6F",
]


  
    all_labels = list(pd.unique(pd.Series(labels_used).astype(str)))
    uniq = [lab for lab in sorted(all_labels) if lab != "-1"]


    color_map = {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(uniq)}

    fig = go.Figure()
    for lab in uniq:
        mask = (pd.Series(labels_used).astype(str).values == lab)
        pts = X2d[mask]
        fig.add_trace(go.Scattergl(
            x=pts[:, 0], y=pts[:, 1],
            mode="markers",
            name=lab,
            marker=dict(size=5, opacity=0.9, color=color_map[lab]),
        ))

    fig.update_layout(
        title="Data Visuilization: Scatter Plot",
        xaxis_title="t-SNE-1",
        yaxis_title="t-SNE-2",
        height=700,
        legend_title="cluster id",
        margin=dict(t=40, r=10, b=10, l=10),
    )
    st.plotly_chart(fig, use_container_width=True)


with right:
    st.subheader("Data characteristics")
    st.markdown(f"- **Dataset size (after clean):** {X.shape[0]}")
    # st.markdown(f"- **# DBSCAN clusters (excl. noise):** {len(clusters)}")

    counts = Counter(r["shape"] for r in rows)
    ordered = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

    st.markdown("- **Cluster Shapes:**")
    md_lines = [f"{i+1}. **{shape.capitalize()}** ({cnt})" for i, (shape, cnt) in enumerate(ordered)]
    st.markdown("\n".join(md_lines))

  
    subcol, _ = st.columns([0.6, 0.4])
    with subcol:
        with st.expander("Show per-cluster assignments"):
            mapping_lines = [
                f"- Cluster **{r['cluster_id']}** → {r['shape']}"
                for r in sorted(rows, key=lambda x: x["cluster_id"])
            ]
            st.markdown("\n".join(mapping_lines))
