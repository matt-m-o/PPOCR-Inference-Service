#ifndef PPOCR_INFER_H
#define PPOCR_INFER_H

// #include "settings.hpp"
#include <iostream>
#include <fastdeploy/vision.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "../../includes/cpp-base64-2.rc.08/base64.cpp"

#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

int const cls_batch_size = 1;
int const rec_batch_size = 6;

struct ContextResolution {
  int width;
  int height;
};

struct InferResult {
  fastdeploy::vision::OCRResult ocr_result;
  ContextResolution context_resolution;
};


InferResult infer( const cv::Mat& image, fastdeploy::pipeline::PPOCRv4& ocr_pipeline  ) {

    InferResult infer_result;

    // auto im = cv::imread(image_file);
    // auto im_bak = im.clone();

    fastdeploy::vision::OCRResult result;
    if (!ocr_pipeline.Predict(image, &result)) {
      std::cerr << "Failed to predict." << std::endl;
      return infer_result;
    }

    // std::cout << result.Str() << std::endl;

    // auto vis_im = fastdeploy::vision::VisOcr(im_bak, result);
    // cv::imwrite("vis_result.jpg", vis_im);
    // std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
    
    ContextResolution context_resolution;
    context_resolution.width = image.cols;
    context_resolution.height = image.rows;

    infer_result.ocr_result = result;
    infer_result.context_resolution = context_resolution;

    return infer_result;
}

InferResult inferBase64( const std::string& base64EncodedImage, fastdeploy::pipeline::PPOCRv4& ocr_pipeline ) {

    InferResult result;

    std::string decoded_data = base64_decode(base64EncodedImage);

    std::string dec_jpg =  base64_decode(base64EncodedImage);
    std::vector<uchar> data(dec_jpg.begin(), dec_jpg.end());
    cv::Mat image = cv::imdecode(cv::Mat(data), 1);
    

    if (!image.empty()) {
        // Image loaded successfully
        // cv::imshow("Loaded Image", image);
        // cv::waitKey(0);
        result = infer( image, ocr_pipeline );
    } else {
        std::cerr << "Failed to load the image." << std::endl;
    }

    return result;
}


struct Models {
    fastdeploy::vision::ocr::DBDetector detectionModel;
    fastdeploy::vision::ocr::Classifier classificationModel;
    fastdeploy::vision::ocr::Recognizer recognitionModel;
};

Models getModels(
    const std::string &det_model_dir,
    const std::string &cls_model_dir,
    const std::string &rec_model_dir,
    const std::string &rec_label_file,  
    const fastdeploy::RuntimeOption &option    
) {
    auto det_model_file = det_model_dir + sep + "inference.pdmodel";
    auto det_params_file = det_model_dir + sep + "inference.pdiparams";

    auto cls_model_file = cls_model_dir + sep + "inference.pdmodel";
    auto cls_params_file = cls_model_dir + sep + "inference.pdiparams";

    auto rec_model_file = rec_model_dir + sep + "inference.pdmodel";
    auto rec_params_file = rec_model_dir + sep + "inference.pdiparams";

    auto det_option = option;
    auto cls_option = option;
    auto rec_option = option;


    // The cls and rec model can inference a batch of images now.
    // User could initialize the inference batch size and set them after create
    // PP-OCR model.
    // int cls_batch_size = 1;
    // int rec_batch_size = 6;

    
    // If use TRT backend, the dynamic shape will be set as follow.
    // We recommend that users set the length and height of the detection model to
    // a multiple of 32.
    // We also recommend that users set the Trt input shape as follow.
    // det_option.SetTrtInputShape("x", {1, 3, 64, 64}, {1, 3, 640, 640},
    //                             {1, 3, 960, 960});
    // cls_option.SetTrtInputShape("x", {1, 3, 48, 10}, {cls_batch_size, 3, 48, 320},
    //                             {cls_batch_size, 3, 48, 1024});
    // rec_option.SetTrtInputShape("x", {1, 3, 48, 10}, {rec_batch_size, 3, 48, 320},
    //                             {rec_batch_size, 3, 48, 2304});


    auto det_model = fastdeploy::vision::ocr::DBDetector(
        det_model_file, det_params_file, det_option);
    auto cls_model = fastdeploy::vision::ocr::Classifier(
        cls_model_file, cls_params_file, cls_option);
    auto rec_model = fastdeploy::vision::ocr::Recognizer(
        rec_model_file, rec_params_file, rec_label_file, rec_option);


    assert(det_model.Initialized());
    assert(cls_model.Initialized());
    assert(rec_model.Initialized());


    det_model.GetPreprocessor().SetMaxSideLen(960);
    det_model.GetPostprocessor().SetDetDBThresh(0.3);
    det_model.GetPostprocessor().SetDetDBBoxThresh(0.6);
    det_model.GetPostprocessor().SetDetDBUnclipRatio(1.5);
    det_model.GetPostprocessor().SetDetDBScoreMode("slow");
    det_model.GetPostprocessor().SetUseDilation(0);
    cls_model.GetPostprocessor().SetClsThresh(0.9);

    Models models;
    models.detectionModel = det_model;
    models.classificationModel = cls_model;
    models.recognitionModel = rec_model;

    return models;
}

fastdeploy::pipeline::PPOCRv4 initPipeline(
    fastdeploy::vision::ocr::DBDetector &det_model, //const std::string &det_model_dir,
    fastdeploy::vision::ocr::Classifier &cls_model, // const std::string &cls_model_dir,
    fastdeploy::vision::ocr::Recognizer &rec_model // const std::string &rec_model_dir,
    // Settings& settings
) {

    // The classification model is optional, so the PP-OCR can also be connected
    // in series as follows
    // auto ppocr_v3 = fastdeploy::pipeline::PPOCRv3(&det_model, &rec_model);
    // auto ppocr_v4 =
    //     fastdeploy::pipeline::PPOCRv4(&det_model, &cls_model, &rec_model);
    auto pipeline = fastdeploy::pipeline::PPOCRv4( &det_model, &cls_model, &rec_model );
    // auto pipeline = fastdeploy::pipeline::PPOCRv4(
    //     &settings.models.detectionModel,
    //     &settings.models.classificationModel,
    //     &settings.models.recognitionModel
    // );

    
    // Set inference batch size for cls model and rec model, the value could be -1
    // and 1 to positive infinity.
    // When inference batch size is set to -1, it means that the inference batch
    // size
    // of the cls and rec models will be the same as the number of boxes detected
    // by the det model.
    pipeline.SetClsBatchSize(cls_batch_size);
    pipeline.SetRecBatchSize(rec_batch_size);

    if (!pipeline.Initialized()) {
        std::cerr << "Failed to initialize PP-OCR." << std::endl;
        // return;
    }

    return pipeline;
}


nlohmann::json ocrResultToJson( const InferResult& infer_result ) {

  auto ocrResult = infer_result.ocr_result;
  auto context_resolution = infer_result.context_resolution;

  std::array<std::string, 4> const boxVerticesIndices = { "top_left", "top_right", "bottom_right", "bottom_left" };

  nlohmann::json resultsJson;
  resultsJson["id"] = "";
  resultsJson["results"] = json::array();
  resultsJson["context_resolution"] = { {"width", context_resolution.width}, {"height", context_resolution.height} };

  int itemIdx = 0;
  for ( const std::string& text : ocrResult.text ) {

    nlohmann::json itemJson;
    itemJson["text"] = text;
    itemJson["box"] = json();
    itemJson["score"] = ocrResult.cls_scores[itemIdx];

    auto const box = ocrResult.boxes[ itemIdx ];
    int boxVertexIdx = 0;
    for ( int bAxisIdx = 0; bAxisIdx < 8; ++bAxisIdx ) {

      const int x = box[ bAxisIdx ];
      const int y = box[ bAxisIdx+1 ];

      const std::string boxVertexName = boxVerticesIndices[ boxVertexIdx ];
      itemJson["box"][boxVertexName] = { {"x", x}, { "y", y } };

      boxVertexIdx++; // [ 1 ... 4 ]
      bAxisIdx++; // [ 1 ... 8]
    }


    resultsJson["results"].push_back(itemJson);
    itemIdx++;
  }

  return resultsJson;
}



#endif // PPOCR_INFER
