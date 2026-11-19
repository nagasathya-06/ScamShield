"""
End-to-End Integration Tests for Scam Shield Pipeline.
"""

import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
from src.config import MODELS_DIR
from src.preprocessor import clean_text_for_word_model, RuleBasedFeatureExtractor
from src.explainability.rule_explainer import diagnose_threats


class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.vectorizer = joblib.load(MODELS_DIR / "hybrid_vectorizer.joblib")
        cls.nb_scratch = joblib.load(MODELS_DIR / "scratch_naive_bayes.joblib")
        cls.rf_model = joblib.load(MODELS_DIR / "random_forest.joblib")

    def test_scam_message_inference(self):
        scam_text = "Urgent: Your SBI account is blocked. Update your KYC at http://bit.ly/sbi-kyc now or power off."
        X = self.vectorizer.transform([scam_text])
        
        pred_nb = self.nb_scratch.predict(X)[0]
        prob_nb = self.nb_scratch.predict_proba(X)[0][1]
        
        self.assertEqual(pred_nb, 1)
        self.assertGreater(prob_nb, 0.70)

    def test_safe_banking_message_inference(self):
        safe_text = "Dear Customer, INR 1,200.00 debited from A/C **4921 on 15-Nov-24 to Zomato UPI Ref 4321098. Avail Bal: INR 22,400.00 - HDFC Bank"
        X = self.vectorizer.transform([safe_text])

        pred_nb = self.nb_scratch.predict(X)[0]
        prob_nb = self.nb_scratch.predict_proba(X)[0][1]

        self.assertEqual(pred_nb, 0)
        self.assertLess(prob_nb, 0.35)

    def test_rule_threat_diagnostics(self):
        threat_text = "PhonePe alert: Enter UPI PIN to claim Rs 3500 lottery cashback at bit.ly/cb-claim"
        threats = diagnose_threats(threat_text)
        self.assertGreater(len(threats), 0)
        categories = [t["category"] for t in threats]
        self.assertIn("Suspicious Link", categories)
        self.assertIn("Credential Threat", categories)


if __name__ == '__main__':
    unittest.main()
