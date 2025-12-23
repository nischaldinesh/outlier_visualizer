# Outlier Visualizer

## Quickstart

```bash
# 1) Clone the repository
git clone https://github.com/nischaldinesh/outlier_visualizer.git

# 2) Enter the project folder
cd outlier_visualizer

# 3) Create a virtual environment (local, isolated Python env)
python -m venv .venv

# 4) Activate the virtual environment
source .venv/bin/activate

# 5) Install all Python dependencies listed in requirements.txt
python -m pip install -r requirements.txt

# 6) Launch the Streamlit app in your browser
streamlit run main.py
```

## Note:

All clustering, shapes, densities, and distribution summaries are computed **on the 2D t‑SNE embedding**, not the original feature space.

## Parameters Description

### t‑SNE: Perplexity

- **Means:** Effective neighborhood size (how many neighbors each point considers).
- **Increase:** Smoother, more global structure, clusters may merge.
- **Decrease:** Finer local detail, risk of fragmenting clusters.

### t‑SNE: Distance Metric (`euclidean` / `cosine` / `manhattan`)

- **Means:** How similarity is measured before embedding.
- **Note:**
  - _Euclidean_ — magnitude‑sensitive (good default after standardization).
  - _Cosine_ — direction‑based (ignore length, good for text/embeddings).
  - _Manhattan_ — sometimes more robust to outliers.

### t‑SNE: Random Seed

- **Means:** Reproducibility of the 2D layout.
- **Note:** Changing it may slightly alter layouts and downstream clusters.

### DBSCAN: `min_samples`

- **Means:** Minimum neighbors within `eps` to make a core point.
- **Increase:** Fewer but stronger clusters, more points labeled as noise.
- **Decrease:** More/smaller clusters, fewer points labeled as noise.

### DBSCAN: `eps`

- **Means:** Neighborhood radius in the **t‑SNE plane**.
- **Increase:** Neighborhoods expand, clusters may merge.
- **Decrease:** Neighborhoods shrink, more fragmentation/noise.
- **Auto (when set to 0):**
  - Compute each point’s **10‑NN distance** in 2D, then take the **95th percentile**.
  - If non‑positive, fall back to **median positive**; else **0.5**.

### α‑shape: Alpha

- **Means:** Concavity of the polygon that wraps cluster points.
- **Increase:** Smoother, hull‑like boundary (area increases, so density decreses).
- **Decrease:** More concave/tight boundary (area decreses, so density increses).

### Cluster Density (k-NN based)
**Means:** How tightly points are packed within a cluster in the 2D t-SNE space.

**Computation:**
- For each point, the distance to its *k-th nearest neighbor* is computed.
- Density is defined as the **inverse of this distance**, averaged over all points in the cluster.
- Smaller distances → higher density; larger distances → lower density.


### Density: k (Number of Neighbors)
**Means:** Controls the locality used for density estimation.

**Selection:**
- Chosen adaptively based on dataset size and dimensionality.
- Smaller datasets → smaller *k* (more local sensitivity).
- Larger or higher-dimensional datasets → larger *k* (more stable density).


### Density Labels (Sparse / Moderate / Dense)
**Means:** Qualitative grouping of clusters based on computed density values.

**Method:**
- Density values across clusters are grouped using quantile-based binning.
- If variation is low, a binary split is used for stability.

**Use:**
- Enables filtering clusters by density.
- Supports comparison of algorithm behavior across different density conditions.
