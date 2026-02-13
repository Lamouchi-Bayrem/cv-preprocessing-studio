import numpy as np
from core.pipeline import PreprocessingPipeline

def test_pipeline_runs():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    pipeline = PreprocessingPipeline()
    out = pipeline.run(img)
    assert out.shape == (100, 100)
