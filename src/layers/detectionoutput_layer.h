#pragma once

#include "../feather_simple_generated.h"
#include "layer.h"

namespace feather
{
class DetectionOutputLayer : public Layer
{
public:
    DetectionOutputLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
        : Layer(layer_param, rt_param)
    {
        const DetectionOutputParameter *detectoutput = layer_param->detection_output_param();
        num_classes = detectoutput->num_classes();
        share_location = detectoutput->share_location();
        background_label_id = detectoutput->background_label_id();
        nms_threshold = detectoutput->nms_threshold();
        top_k = detectoutput->top_k();
        code_type = detectoutput->code_type();
        keep_top_k = detectoutput->keep_top_k();
        confidence_threshold = detectoutput->confidence_threshold();
#ifdef PRT_PARAM
        printf("num_classes: %d, share_location: %d, background_label_id: %d, nms_threshold: %f, top_k: %d, code_type: %d, keep_top_k: %d, confidence_threshold: %f\n",
               num_classes, share_location, background_label_id, nms_threshold, top_k, code_type, keep_top_k, confidence_threshold);
#endif
    }

    struct BBoxRect
    {
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        int label;
    };

    static inline float intersection_area(const BBoxRect& a, const BBoxRect& b)
    {
        if (a.xmin > b.xmax || a.xmax < b.xmin || a.ymin > b.ymax || a.ymax < b.ymin)
        {
            // no intersection
            return 0.f;
        }

        float inter_width = std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin);
        float inter_height = std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin);

        return inter_width * inter_height;
    }

    template <typename T>
    static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores, int left, int right)
    {
        int i = left;
        int j = right;
        float p = scores[(left + right) / 2];

        while (i <= j)
        {
            while (scores[i] > p)
                i++;

            while (scores[j] < p)
                j--;

            if (i <= j)
            {
                // swap
                std::swap(datas[i], datas[j]);
                std::swap(scores[i], scores[j]);

                i++;
                j--;
            }
        }

        if (left < j)
            qsort_descent_inplace(datas, scores, left, j);

        if (i < right)
            qsort_descent_inplace(datas, scores, i, right);
    }

    template <typename T>
    static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores)
    {
        if (datas.empty() || scores.empty())
            return;

        qsort_descent_inplace(datas, scores, 0, scores.size() - 1);
    }

    static void nms_sorted_bboxes(const std::vector<BBoxRect>& bboxes, std::vector<int>& picked, float nms_threshold)
    {
        picked.clear();

        const int n = bboxes.size();

        std::vector<float> areas(n);
        for (int i = 0; i < n; i++)
        {
            const BBoxRect& r = bboxes[i];

            float width = r.xmax - r.xmin;
            float height = r.ymax - r.ymin;

            areas[i] = width * height;
        }

        for (int i = 0; i < n; i++)
        {
            const BBoxRect& a = bboxes[i];

            int keep = 1;
            for (int j = 0; j < (int)picked.size(); j++)
            {
                const BBoxRect& b = bboxes[picked[j]];

                // intersection over union
                float inter_area = intersection_area(a, b);
                float union_area = areas[i] + areas[picked[j]] - inter_area;
                //			   float IoU = inter_area / union_area
                if (inter_area / union_area > nms_threshold)
                    keep = 0;
            }

            if (keep)
                picked.push_back(i);
        }
    }

    int Forward()
    {
        const float* location_ptr = _bottom_blobs[_bottom[0]]->data(); /* regrerssion */
        const float* priorbox_ptr = _bottom_blobs[_bottom[2]]->data(); /* priorbox */
        const float* variance_ptr = priorbox_ptr + _bottom_blobs[_bottom[2]]->height(); /* priorbox variance */

#if 0//def PRINT_TOP
        const float* confidence = _bottom_blobs[_bottom[1]]->data();
        printf("\n\n[%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f]\n"
               "[%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f]\n"
               "[%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f]\n"
               "[%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f]\n",
               location_ptr[0], location_ptr[1], location_ptr[2], location_ptr[3], location_ptr[4], location_ptr[5], location_ptr[6], location_ptr[7],
               priorbox_ptr[0], priorbox_ptr[1], priorbox_ptr[2], priorbox_ptr[3], priorbox_ptr[4], priorbox_ptr[5], priorbox_ptr[6], priorbox_ptr[7],
               variance_ptr[0], variance_ptr[1], variance_ptr[2], variance_ptr[3], variance_ptr[4], variance_ptr[5], variance_ptr[6], variance_ptr[7],
               confidence[0],   confidence[1],   confidence[2],   confidence[3],   confidence[4],   confidence[5],   confidence[6],   confidence[7]);

        printf("priorbox: %d, %d\n", _bottom_blobs[_bottom[2]]->height(), _bottom_blobs[_bottom[2]]->height()/4);
#endif

        //printf("\n\n\n\n\nnum_prior: %d\n\n\n", num_prior);
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < num_prior; i++)
        {
            const float* loc = location_ptr + i * 4;
            const float* pb  = priorbox_ptr + i * 4;
            const float* var = variance_ptr + i * 4;

            float* bbox = bboxes + i*4;

            // CENTER_SIZE
            float pb_w = pb[2] - pb[0];
            float pb_h = pb[3] - pb[1];
            float pb_cx = (pb[0] + pb[2]) * 0.5f;
            float pb_cy = (pb[1] + pb[3]) * 0.5f;

            float bbox_cx = var[0] * loc[0] * pb_w + pb_cx;
            float bbox_cy = var[1] * loc[1] * pb_h + pb_cy;
            float bbox_w = exp(var[2] * loc[2]) * pb_w;
            float bbox_h = exp(var[3] * loc[3]) * pb_h;

            bbox[0] = bbox_cx - bbox_w * 0.5f;
            bbox[1] = bbox_cy - bbox_h * 0.5f;
            bbox[2] = bbox_cx + bbox_w * 0.5f;
            bbox[3] = bbox_cy + bbox_h * 0.5f;
        }

        // sort and nms for each class
        std::vector< std::vector<BBoxRect> > all_class_bbox_rects;
        std::vector< std::vector<float> > all_class_bbox_scores;
        all_class_bbox_rects.resize(num_classes);
        all_class_bbox_scores.resize(num_classes);

        // start from 1 to ignore background class
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 1; i < num_classes; i++)
        {
            // filter by confidence_threshold
            std::vector<BBoxRect> class_bbox_rects;
            std::vector<float> class_bbox_scores;

            for (int j = 0; j < num_prior; j++)
            {
                float score = _bottom_blobs[_bottom[1]]->data()[j * num_classes + i];
                if (score > confidence_threshold)
                {
                    const float* bbox = bboxes + j*4;
                    BBoxRect c = { bbox[0], bbox[1], bbox[2], bbox[3], i };
                    class_bbox_rects.push_back(c);
                    class_bbox_scores.push_back(score);
                }
            }

            // sort inplace
            qsort_descent_inplace(class_bbox_rects, class_bbox_scores);

            // keep nms_top_k
            if ((int)class_bbox_rects.size() > top_k)
            {
                class_bbox_rects.resize(top_k);
                class_bbox_scores.resize(top_k);
            }

            // apply nms
            std::vector<int> picked;
            nms_sorted_bboxes(class_bbox_rects, picked, nms_threshold);

            // select
            for (int j = 0; j < (int)picked.size(); j++)
            {
                int z = picked[j];
                all_class_bbox_rects[i].push_back(class_bbox_rects[z]);
                all_class_bbox_scores[i].push_back(class_bbox_scores[z]);
            }
        }

        // gather all class
        std::vector<BBoxRect> bbox_rects;
        std::vector<float> bbox_scores;

        for (int i = 1; i < num_classes; i++)
        {
            const std::vector<BBoxRect>& class_bbox_rects = all_class_bbox_rects[i];
            const std::vector<float>& class_bbox_scores = all_class_bbox_scores[i];

            bbox_rects.insert(bbox_rects.end(), class_bbox_rects.begin(), class_bbox_rects.end());
            bbox_scores.insert(bbox_scores.end(), class_bbox_scores.begin(), class_bbox_scores.end());
        }

        // global sort inplace
        qsort_descent_inplace(bbox_rects, bbox_scores);

        // keep_top_k
        if ((int)bbox_rects.size() > keep_top_k)
        {
            bbox_rects.resize(keep_top_k);
            bbox_scores.resize(keep_top_k);
        }

        // fill result
        int num_detected = bbox_rects.size();
        //printf("num_detected: %d\n", num_detected);
        float* top_blob = _top_blobs[_top[0]]->data();
        top_blob[0] = num_detected*1.0f;
        top_blob++;
        for (int i = 0; i < num_detected; i++)
        {
            const BBoxRect& r = bbox_rects[i];
            float score = bbox_scores[i];
            float* outptr = top_blob + i*6;
            //printf("%d %f\n", r.label, score);
            outptr[0] = r.label;
            outptr[1] = score;
            outptr[2] = r.xmin;
            outptr[3] = r.ymin;
            outptr[4] = r.xmax;
            outptr[5] = r.ymax;
        }

        Layer::Forward();
        return 0;
    }

    int GenerateTopBlobs()
    {
        _top_blobs[_top[0]] = new Blob<float>(1, 1, 1, 1 + 6*keep_top_k);
        _top_blobs[_top[0]]->_name = "Top";
        return 0;
    }

    int Init()
    {
        num_prior = _bottom_blobs[_bottom[2]]->height() / 4;
        MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&bboxes,  sizeof(float) * 4 * num_prior));
        return 0;
    }
private:
    uint32_t num_classes;
    bool share_location;
    uint32_t background_label_id;
    float nms_threshold;
    int top_k;
    feather::DetectionOutputParameter_::CodeType code_type;
    int keep_top_k;
    float confidence_threshold;
    float *bboxes;
    int num_prior;
};
};
