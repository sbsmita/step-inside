"""Download the ONNX models this project runs through OpenCV's DNN module (all Apache-2.0)."""
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = {
    # OpenCV Model Zoo
    "object_detection_yolox_2022nov.onnx":
        "https://huggingface.co/opencv/object_detection_yolox/resolve/main/object_detection_yolox_2022nov.onnx",
    "face_detection_yunet_2023mar.onnx":
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    "text_detection_en_ppocrv3_2023may.onnx":
        "https://huggingface.co/opencv/text_detection_ppocr/resolve/main/text_detection_en_ppocrv3_2023may.onnx",
    "text_recognition_CRNN_CH_2021sep.onnx":
        "https://huggingface.co/opencv/text_recognition_crnn/resolve/main/text_recognition_CRNN_CH_2021sep.onnx",
    # Depth Anything V2 Small (Apache-2.0; the larger variants are non-commercial and not used)
    "depth_anything_v2_small.onnx":
        "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx",
}


def main() -> None:
    for name, url in MODELS.items():
        dest = HERE / name
        if dest.exists() and dest.stat().st_size > 100_000:
            print(f"ok   {name}")
            continue
        print(f"get  {name}")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
