# PPOCR Inference Service

A simple and experimental OCR service powered by PaddleOCR. <br>
This is not ready for production, use it with caution.

### Quick start

You need Windows 10 or 11 and [VCRedist Runtimes](https://www.techpowerup.com/download/visual-c-redistributable-runtime-package-all-in-one/) installed. <br>

1. [Download](https://github.com/matt-m-o/PPOCR-Inference-Service/releases) the latest build.
2. [Download](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/models_list_en.md) detection, angle classification, recognition models and recognition labels for the languages you need.
3. Extract and place models in the "./models" directory.
4. Define configuration presets for each language in the "./presets/language_presets" directory.

Example: english_v4.json
```json
{
  "name": "english_v4",
  "language_code": "en",
  "detection_model_dir": "en_PP-OCRv3_det_infer",
  "classification_model_dir" : "ch_ppocr_mobile_v2.0_cls_infer",
  "recognition_model_dir" : "en_PP-OCRv4_rec_infer",
  "recognition_label_file_dir" : "dict_en.txt"
}
```
5. Define the application configuration preset in the "./presets" directory.

Example: default.json
```json
{
  "name": "default",
  "language_presets": {
      "ja": "japanese_v4",
      "en": "english_v4"
  },
  "language_code": "ja",
  "initialize_all_language_presets": true,
  "inference_backend" : "Open_VINO",
  "cpu_threads": 16,
  "port": 12345,
  "max_image_width": 1920
}
```
** Currently "language_code" and "initialize_all_language_presets" do not take effect. <br>
** "inference_backend" can take any of the following values: Paddle_CPU, Open_VINO, ONNX_CPU, Paddle_Lite.

6. Run "ppocr_infer_service_grpc.exe"

7. Import the protos/ocr_service.proto from the source code into your programming language of preference or Postman.
