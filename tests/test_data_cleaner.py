import pandas as pd
import pandas.testing as pdt
import unittest

from src.data_cleaner import DataCleaner


def make_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [" Alice ", "Bob", None, " Carol  "],
            "age": [25, None, 35, 120],
            "city": ["SCL", "LPZ", "SCL", "LPZ"],
        }
    )


class TestDataCleaner(unittest.TestCase):
    def test_example_trim_strings_with_pandas_testing(self):
        df = pd.DataFrame({"name": ["  Alice  ", "  Bob  ", "Carol"], "age": [25, 30, 35]})
        cleaner = DataCleaner()

        result = cleaner.trim_strings(df, ["name"])

        expected = pd.DataFrame({"name": ["Alice", "Bob", "Carol"], "age": [25, 30, 35]})
        pdt.assert_frame_equal(result, expected)

    def test_example_drop_invalid_rows_with_pandas_testing(self):
        df = pd.DataFrame(
            {
                "name": ["Alice", None, "Bob"],
                "age": [25, 30, None],
                "city": ["SCL", "LPZ", "SCL"],
            }
        )
        cleaner = DataCleaner()

        result = cleaner.drop_invalid_rows(df, ["name"])

        expected_name_series = pd.Series(["Alice", "Bob"], index=[0, 2], name="name")
        pdt.assert_series_equal(result["name"], expected_name_series, check_names=True)

    def test_drop_invalid_rows_removes_rows_with_missing_values(self):
        cleaner = DataCleaner()
        df = make_sample_df()

        result = cleaner.drop_invalid_rows(df, ["name", "age"])

        self.assertEqual(result[["name", "age"]].isna().sum().sum(), 0)
        self.assertLess(len(result), len(df))

    def test_drop_invalid_rows_raises_keyerror_for_unknown_column(self):
        cleaner = DataCleaner()
        df = make_sample_df()

        with self.assertRaises(KeyError):
            cleaner.drop_invalid_rows(df, ["does_not_exist"])

    def test_trim_strings_strips_whitespace_without_changing_other_columns(self):
        cleaner = DataCleaner()
        df = pd.DataFrame(
            {
                "name": [" Alice ", "Bob", " Carol  "],
                "age": [25, 30, 35],
                "city": ["SCL", "LPZ", "SCL"],
            }
        )
        original_city = df["city"].copy()

        result = cleaner.trim_strings(df, ["name"])

        self.assertEqual(df.loc[0, "name"], " Alice ")
        self.assertEqual(df.loc[2, "name"], " Carol  ")
        self.assertEqual(result.loc[0, "name"], "Alice")
        self.assertEqual(result.loc[2, "name"], "Carol")
        pdt.assert_series_equal(result["city"], original_city)

    def test_trim_strings_raises_typeerror_for_non_string_column(self):
        cleaner = DataCleaner()
        df = make_sample_df()

        with self.assertRaises(TypeError):
            cleaner.trim_strings(df, ["age"])

    def test_remove_outliers_iqr_removes_extreme_values(self):
        cleaner = DataCleaner()
        df = pd.DataFrame({"age": [10, 11, 12, 13, 100], "city": ["SCL", "SCL", "LPZ", "LPZ", "SCL"]})

        result = cleaner.remove_outliers_iqr(df, "age", factor=1.5)

        self.assertNotIn(100, result["age"].values)
        self.assertIn(12, result["age"].values)

    def test_remove_outliers_iqr_raises_keyerror_for_missing_column(self):
        cleaner = DataCleaner()
        df = make_sample_df()

        with self.assertRaises(KeyError):
            cleaner.remove_outliers_iqr(df, "salary")

    def test_remove_outliers_iqr_raises_typeerror_for_non_numeric_column(self):
        cleaner = DataCleaner()
        df = make_sample_df()

        with self.assertRaises(TypeError):
            cleaner.remove_outliers_iqr(df, "city")


if __name__ == "__main__":
    unittest.main()

