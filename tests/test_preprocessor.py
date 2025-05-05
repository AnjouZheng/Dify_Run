import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = TextPreprocessor()

    def test_remove_special_characters(self):
        input_text = "Hello! How are you? #spam @email"
        expected_output = "hello how are you spam email"
        result = self.preprocessor.clean_text(input_text)
        self.assertEqual(result, expected_output)

    def test_lowercase_conversion(self):
        input_text = "SPAM EMAIL Test"
        expected_output = "spam email test"
        result = self.preprocessor.clean_text(input_text)
        self.assertEqual(result, expected_output)

    def test_remove_stopwords(self):
        input_text = "this is a sample spam email with some stopwords"
        expected_output = "sample spam email stopwords"
        result = self.preprocessor.remove_stopwords(input_text)
        self.assertEqual(result, expected_output)

    def test_tokenization(self):
        input_text = "spam detection machine learning"
        expected_tokens = ["spam", "detection", "machine", "learning"]
        result = self.preprocessor.tokenize(input_text)
        self.assertEqual(result, expected_tokens)

    def test_empty_input(self):
        input_text = ""
        expected_output = ""
        result = self.preprocessor.clean_text(input_text)
        self.assertEqual(result, expected_output)

    def test_numeric_text(self):
        input_text = "123 456 spam email 789"
        expected_output = "spam email"
        result = self.preprocessor.clean_text(input_text)
        self.assertEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main()