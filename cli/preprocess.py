import cv2
import argparse
from core.pipeline import PreprocessingPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    img = cv2.imread(args.input)
    pipeline = PreprocessingPipeline()
    out = pipeline.run(img)
    cv2.imwrite(args.output, out * 255)

if __name__ == "__main__":
    main()
