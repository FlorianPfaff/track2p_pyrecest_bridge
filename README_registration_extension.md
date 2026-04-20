# Registration-aware longitudinal tracking extension

`track2p_pyrecest_bridge.py` on `main` already provides ROI-aware pairwise costs and
PyRecEst-ready association bundles, but it expects the later session to already
be expressed in the earlier session's coordinate frame. This extension adds the
missing registration step for raw cross-session tracking.

## Added module

- `track2p_registration_extension.py`

## What it does

- estimates a transform between consecutive sessions using PyRecEst point-set
  registration over ROI centroids,
- warps ROI masks and optional mean-image/FOV data from the measurement session
  into the reference-session frame, and
- reuses the existing bridge to build ROI-aware pairwise costs and standard
  `SessionAssociationBundle` objects.

## Minimal example

```python
from track2p_pyrecest_bridge import load_track2p_subject
from track2p_registration_extension import (
    build_registered_session_pair_association_bundle,
)

sessions = load_track2p_subject("/path/to/jm039", plane_name="plane0")
registered = build_registered_session_pair_association_bundle(
    sessions[0],
    sessions[1],
    registration_model="affine",
    pairwise_cost_kwargs={
        "max_centroid_distance": 25.0,
        "roi_feature_weight": 0.25,
    },
)

pairwise_cost_matrix = registered.association_bundle.pairwise_cost_matrix
registered_plane = registered.plane_registration.registered_measurement_plane
```
