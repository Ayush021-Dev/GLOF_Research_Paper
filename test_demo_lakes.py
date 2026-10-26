"""Integration checks for the offline dataset-lake demo."""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent


class DemoLakesTest(unittest.TestCase):
    def test_all_dataset_lakes_can_be_analysed(self):
        catalog = json.loads((ROOT / "demo_lakes/catalog.json").read_text())
        self.assertGreater(len(catalog), 5)
        self.assertEqual(len({lake["master_lake_id"] for lake in catalog}), len(catalog))
        features = pd.read_csv(ROOT / "master_lake_dataset/lake_features.csv")
        app = AppTest.from_file(str(ROOT / "dem_glof_app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        scores, areas = [], []
        for index, lake in enumerate(catalog[:5]):
            source = features[features.observation_id == lake["observation_id"]].iloc[0]
            self.assertEqual(source.master_lake_id, lake["master_lake_id"])
            with rasterio.open(ROOT / lake["mask_path"]) as mask:
                self.assertIsNotNone(mask.crs)
                self.assertEqual(set(np.unique(mask.read(1))), {0, 1})
            app.multiselect(key="demo_lakes").set_value([index]).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.metric), 0)  # Previous lake's result must not remain.
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            score = float(app.metric[0].value.split(" / ")[0])
            area = float(app.metric[2].value.split()[0])
            self.assertTrue(0 <= score <= 100)
            self.assertAlmostEqual(area, source.area_derived_km2, delta=source.area_derived_km2 * .06)
            self.assertTrue(any(lake["master_lake_id"] in heading.value for heading in app.subheader))
            metrics = app.dataframe[1].value
            self.assertTrue(np.isfinite(float(metrics.loc["lake_elev_mean"].iloc[0])))
            scores.append(score)
            areas.append(area)
        self.assertGreater(len(set(scores)), 1)
        self.assertEqual(len(set(areas)), 5)
        app.multiselect(key="demo_lakes").set_value([0, 1]).run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        summary = app.dataframe[0].value
        self.assertEqual(len(summary), 2)
        app.selectbox(key="detail_lake").select(catalog[1]['master_lake_id']).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe[0].value), 2)
        self.assertAlmostEqual(float(app.metric[0].value.split(' / ')[0]), float(summary.iloc[1].Score), delta=.1)
        app.checkbox(key="all_lakes").check().run()
        app.button[0].click().run(timeout=120)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertFalse(app.warning)
        self.assertEqual(len(app.dataframe[0].value), len(catalog))
        self.assertTrue(np.isfinite(app.dataframe[0].value[['Score', 'Area_km2', 'Elevation_m']]).all().all())
        app.checkbox(key="all_lakes").uncheck().run()
        app.multiselect(key="demo_lakes").set_value([]).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        app.radio(key="lake_input").set_value("Upload a lake mask").run()
        self.assertFalse(app.exception)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.error[0].value, "Upload a lake mask first.")


if __name__ == "__main__":
    unittest.main()
