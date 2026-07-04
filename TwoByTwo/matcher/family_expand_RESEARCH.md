# Fragmentation-robust interaction-boundary methods — research synthesis (2026-07-01)

Deep-research (102 agents, 20 sources, 24/25 claims confirmed) on: *alternatives
to global PCA for defining the boundary/connectivity of a fragmented LArTPC
interaction, as the growth criterion for agglomerative "family" clustering.*

## Headline: the field already does this
Both production LArTPC chains build the interaction incrementally from
sub-clusters, NOT from one global axis — validating the family-expansion design:
- **Pandora** (rule-based): many decoupled topology-tuned algorithms that
  deliberately **over-fragment** (avoid merging energy from multiple particles)
  then **reassemble by topological association** between pairs/chains of clusters.
  [arXiv:1506.05348]
- **SPINE / lartpc_mlreco3d (GrapPA)** (learned): a hierarchical GNN chain
  charge fragments → particle instances → interactions; EM showers are seeded by
  DBSCAN then a GNN predicts the fragment adjacency, "because EM activity exhibits
  spatially detached fragments." Mean ARI 97.8% (shower) / 99.2% (interaction) /
  99.8% (primary) — on PILArNet sim, low pile-up. [arXiv:2007.01335]

## Recommended growth criterion (replaces global PCA)
Two complementary ingredients (medium-confidence synthesis of confirmed claims):
1. **Local (per-neighbourhood) PCA / structure-tensor "dimensionality" features**
   — linearity (λ1−λ2)/λ1, planarity (λ2−λ3)/λ1, scattering λ3/λ1 — the
   track-vs-shower-vs-blob discriminator. "Most directly reusable, dependency-light
   replacement for global PCA." Libs: `jakteristics`, CloudCompare. [Weinmann/KIT]
2. **Proximity-graph connectivity rule** — MST, or adaptive edge-cutting
   (CutESC on a Gabriel graph, `github.com/alperaksac/cutESC`) — to decide which
   neighbour joins and where the boundary is. [CutESC, Pattern Recognition 2019]

**Dependency-light first cut (the recommendation):** MST / single-linkage over
sub-clusters, **gated by local-PCA dimensionality agreement + a scale-aware gap
threshold**, instead of a fixed distance. Upgrade path: a GrapPA-style GNN edge
predictor over sub-clusters.

## Also useful, situational
- **alpha-shapes / concave hull** (Open3D, `alphashape`): explicit boundary
  *surface*, tunable α — a boundary descriptor, NOT a family decision.
- **ElPiGraph** (elastic principal graph) / **L1-medial skeleton**: model
  branching topology, tolerate noise. Caveat: robustness is to noise/outliers,
  and elastic/skeleton penalties can *resist bridging genuine gaps* (delta rays,
  shower onset) — the exact thing we WANT to bridge. No clean pip lib for L1-medial.

## Cautions (verified)
- "Complete graph beats MST/kNN" is for the **learned GNN** message-passing over
  ~tens of fragments — NOT for a non-learned rule over O(100–1000) points, where a
  complete graph is O(N²) and **MST/single-linkage is the right first-cut**.
- **Spectral / Laplacian-eigenmaps robustness REFUTED (0-3)** — don't adopt it on
  a robustness argument.
- alpha-shape / ElPiGraph on surface scans → extrapolation to volumetric ionization
  is reasonable but unbenchmarked.

## The open question that IS our tuning knob
"What quantitative **gap scale** separates intra-family fragmentation (delta rays,
EM-shower onset) from inter-family proximity (distinct interactions)?" → this is
exactly what the scale-aware/adaptive gap threshold must learn per family.

Full report + citations: task `we1c3ltkk` output.
