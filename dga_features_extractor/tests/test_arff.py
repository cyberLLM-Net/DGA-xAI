import csv
import unittest

from dga_features_extractor.arff import escape_arff_string, format_arff_row, generate_arff_header


class TestArff(unittest.TestCase):
    def test_header_generation(self):
        header = generate_arff_header("UDCDGA", ["F1", "F2"])
        self.assertIn("@RELATION UDCDGA", header)
        self.assertIn("@ATTRIBUTE DOMAIN STRING", header)
        self.assertIn("@ATTRIBUTE F1 NUMERIC", header)
        self.assertIn("@ATTRIBUTE F2 NUMERIC", header)
        self.assertIn("@ATTRIBUTE CLASS STRING", header)
        self.assertIn("@ATTRIBUTE LABEL {0,1}", header)
        self.assertTrue(header.strip().endswith("@DATA"))

    def test_arff_string_escape(self):
        escaped = escape_arff_string("bad'domain\\name.com")
        self.assertEqual(escaped, "'bad\\'domain\\\\name.com'")

    def test_row_format_and_separate_label(self):
        row = format_arff_row("example.com", [1.2, 3.4], "kraken", 1)
        fields = next(csv.reader([row.strip()], delimiter=",", quotechar="'", escapechar="\\"))
        self.assertEqual(fields[0], "example.com")
        self.assertEqual(fields[1], "1.2")
        self.assertEqual(fields[2], "3.4")
        self.assertEqual(fields[3], "kraken")
        self.assertEqual(fields[4], "1")


if __name__ == "__main__":
    unittest.main()
