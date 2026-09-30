import unittest
from pathlib import Path

from dga_features_extractor.features import compute_domain_features, compute_domain_features_subset, default_feature_names
from dga_features_extractor.schema import resolve_feature_names, selected_feature_names


class TestFeatures(unittest.TestCase):
    def test_feature_determinism(self):
        a = compute_domain_features("example.com")
        b = compute_domain_features("example.com")
        self.assertEqual(a, b)

    def test_expected_keys_exist(self):
        feats = compute_domain_features("a1-b2.example")
        for key in ["L_NUM_LABELS", "L_ENTROPY", "L_LC_D", "1G_MEAN", "2G_DST_KL", "3G_PEA", "N_LEN_FQDN"]:
            self.assertIn(key, feats)

    def test_schema_regression_against_reference_arff(self):
        reference = Path("domains_output.arff")
        schema_names = resolve_feature_names(reference)
        selected_names = selected_feature_names()
        self.assertEqual(schema_names, selected_names)
        self.assertEqual(len(schema_names), len(selected_names))
        self.assertGreater(len(default_feature_names()), len(selected_names))

    def test_selected_feature_count_is_stable(self):
        self.assertEqual(len(selected_feature_names()), 54)

    def test_subset_feature_computation_only_returns_requested_columns(self):
        selected = selected_feature_names()
        subset = compute_domain_features_subset("a1-b2.example", selected)
        full = compute_domain_features("a1-b2.example")

        self.assertEqual(set(subset.keys()), set(selected))
        for name in selected:
            self.assertEqual(subset[name], full[name])


if __name__ == "__main__":
    unittest.main()
