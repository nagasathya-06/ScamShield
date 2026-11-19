"""
LIME (Local Interpretable Model-agnostic Explanations) Module.
Explains predictions for any black-box classifier by perturbing text inputs.
"""

from typing import List, Tuple, Dict, Any, Callable
import numpy as np
from lime.lime_text import LimeTextExplainer


class ScamShieldLIMEExplainer:
    """
    Wraps LIME text explainer to provide interpretable local explanations
    with feature contributions and styled HTML representations.
    """

    def __init__(self, class_names: List[str] = None):
        if class_names is None:
            class_names = ["Safe", "Scam"]
        self.class_names = class_names
        self.explainer = LimeTextExplainer(
            class_names=self.class_names,
            bow=False,
            random_state=42
        )

    def explain(
        self,
        text: str,
        predict_proba_fn: Callable[[List[str]], np.ndarray],
        num_features: int = 8,
        num_samples: int = 300
    ) -> Dict[str, Any]:
        """
        Generates local explanation for a specific text message.
        """
        exp = self.explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_proba_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=(1,)  # Explain for Scam class
        )

        weights = exp.as_list(label=1)
        # weights is a list of (word, score): positive means pushes towards Scam, negative means pushes towards Safe
        scam_drivers = [(w, s) for w, s in weights if s > 0]
        safe_drivers = [(w, s) for w, s in weights if s < 0]

        return {
            "weights": weights,
            "scam_drivers": scam_drivers,
            "safe_drivers": safe_drivers,
            "html": exp.as_html()
        }
