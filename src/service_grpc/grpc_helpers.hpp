#ifndef GRPC_HELPERS_HPP
#define GRPC_HELPERS_HPP

#include "../hpp/inference_manager.hpp"
#include "ocr_service.grpc.pb.h"

using ocr_service::RecognizeDefaultResponse;
using ocr_service::RecognizeDefaultResponse;
using ocr_service::DetectResponse;

void ocrResultGRPCHelper(
    const InferenceResult& inference_result,
    RecognizeDefaultResponse* response
) {

    auto context_resolution = response->mutable_context_resolution();
    context_resolution->set_width( inference_result.context_resolution.width );
    context_resolution->set_height( inference_result.context_resolution.height );
    
    int item_idx = 0;
    for ( const std::string& text : inference_result.ocr_result.text ) {

        auto new_result = response->add_results();
        new_result->set_recognition_score( inference_result.ocr_result.rec_scores[ item_idx ] );
        new_result->set_classification_score( inference_result.ocr_result.cls_scores[ item_idx ] ); // Text direction
        new_result->set_classification_label( inference_result.ocr_result.cls_labels[ item_idx ] ); // Text direction

        auto new_box = new_result->mutable_box();

        auto const box = inference_result.ocr_result.boxes[ item_idx ];
        int box_vertex_idx = 0;
        for ( int b_axis_idx = 0; b_axis_idx < 8; ++b_axis_idx ) {
        
            const int x = box[ b_axis_idx ];
            const int y = box[ b_axis_idx + 1 ];

            ocr_service::Vertex* vertex;

            if ( box_vertex_idx == 0 ) {
                vertex = new_box->mutable_top_left();
            }
            else if ( box_vertex_idx == 1 ) {
                vertex = new_box->mutable_top_right();
            }
            else if ( box_vertex_idx == 2 ) {
                vertex = new_box->mutable_bottom_right();
            }
            else if ( box_vertex_idx == 3 ) {
                vertex = new_box->mutable_bottom_left();
            }

            vertex->set_x(x);
            vertex->set_y(y);

            box_vertex_idx++; // [ 1 ... 4 ]
            b_axis_idx++; // [ 1 ... 8]
        }

        auto text_line = new_result->add_text_lines();
        text_line->set_content(text);
        text_line->mutable_box()->CopyFrom(*new_box);

        item_idx++;
    }
}

void detectionResultGRPCHelper(
    const DetectionResult& detection_result,
    DetectResponse* response
) {

    auto context_resolution = response->mutable_context_resolution();
    context_resolution->set_width( detection_result.context_resolution.width );
    context_resolution->set_height( detection_result.context_resolution.height );
    
    int item_idx = 0;
    for ( const auto _box : detection_result.ocr_result.boxes ) {

        std::cout << item_idx;

        auto new_result = response->add_results();

        auto new_box = new_result->mutable_box();

        auto const box = detection_result.ocr_result.boxes[ item_idx ];
        int box_vertex_idx = 0;
        for ( int b_axis_idx = 0; b_axis_idx < 8; ++b_axis_idx ) {

        
            const int x = box[ b_axis_idx ];
            const int y = box[ b_axis_idx + 1 ];

            ocr_service::Vertex* vertex;

            if ( box_vertex_idx == 0 ) {
                vertex = new_box->mutable_top_left();
            }
            else if ( box_vertex_idx == 1 ) {
                vertex = new_box->mutable_top_right();
            }
            else if ( box_vertex_idx == 2 ) {
                vertex = new_box->mutable_bottom_right();
            }
            else if ( box_vertex_idx == 3 ) {
                vertex = new_box->mutable_bottom_left();
            }

            vertex->set_x(x);
            vertex->set_y(y);

            box_vertex_idx++; // [ 1 ... 4 ]
            b_axis_idx++; // [ 1 ... 8]
        }

        auto text_line = new_result->add_text_lines();
        text_line->mutable_box()->CopyFrom(*new_box);

        item_idx++;
    }
}

#endif