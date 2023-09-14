#ifndef GRPC_HELPERS_HPP
#define GRPC_HELPERS_HPP

#include "../hpp/inference_manager.hpp"
#include "ppocr_service.grpc.pb.h"

using ppocr_service::RecognizeGenericResponse;

void inferenceResultGRPCHelper(
    const InferenceResult& inference_result,
    RecognizeGenericResponse* response
) {

    auto context_resolution = response->mutable_context_resolution();
    context_resolution->set_width( inference_result.context_resolution.width );
    context_resolution->set_height( inference_result.context_resolution.height );
    
    int item_idx = 0;
    for ( const std::string& text : inference_result.ocr_result.text ) {

        auto new_result = response->add_results();
        new_result->set_text(text);
        new_result->set_score( inference_result.ocr_result.cls_scores[ item_idx ] );

        auto new_box = new_result->mutable_box();

        auto const box = inference_result.ocr_result.boxes[ item_idx ];
        int box_vertex_idx = 0;
        for ( int b_axis_idx = 0; b_axis_idx < 8; ++b_axis_idx ) {
        
            const int x = box[ b_axis_idx ];
            const int y = box[ b_axis_idx + 1 ];

            ppocr_service::Vertex* vertex;

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

        item_idx++;
    }
}

#endif