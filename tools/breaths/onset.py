"""Research-harness alias for onset features, plus the JSON-model scorer.

`onset_features` now lives in FinalPass (`finalpass.breath_edges`). The JSON
scorer below is kept so the version-1 and version-2 model files used during the
evaluation can still be reproduced exactly.
"""
from __future__ import annotations

import numpy as np

from finalpass.breath_edges import onset_features  # noqa: F401

MODEL_PATH = __import__("pathlib").Path(__file__).with_name("t_inhale_model.json")


def load_model(path=MODEL_PATH) -> dict:
    import json
    return json.loads(open(path).read())


def t_inhale_score(features: dict, gap_before_s: float, model: dict) -> float:
    """Logit that this breath opens with a mouth-release burst (a T-inhale).

    Fitted on the operator's own labels; see t_inhale_model.json for provenance,
    held-out performance and the label caveat. Higher is more T-like.
    """
    vals = []
    for f in model["features"]:
        if f == "gap_before_ms":
            v = max(gap_before_s * 1000.0, 10.0)
            vals.append(np.log10(v) if model.get("gap_before_ms_is_log10") else v)
        else:
            vals.append(features[f])
    z = (np.asarray(vals) - np.asarray(model["mean"])) / np.asarray(model["std"])
    return float(model["intercept"] + z @ np.asarray(model["weights"]))
