#pragma once

#include "../feather_simple_generated.h"
#include "layer.h"

namespace feather
{
class PriorBoxLayer : public Layer
{
public:
    PriorBoxLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
        : Layer(layer_param, rt_param)
    {
        uint32_t i =0;
        const PriorBoxParameter *priorbox = layer_param->priorbox_param();
//#define PRT_PARAM
        for(i = 0; i < VectorLength(priorbox->min_size()); ++i)
        {
            min_size.push_back(priorbox->min_size()->Get(i));
#ifdef PRT_PARAM
            if (0 == i)
                printf(" min_size: %f", min_size[i]);
            else
                printf(" %f", min_size[i]);
#endif
        }
        for(i = 0; i < VectorLength(priorbox->max_size()); ++i)
        {
            max_size.push_back(priorbox->max_size()->Get(i));
#ifdef PRT_PARAM
            if (0 == i)
                printf(" max_size: %f", max_size[i]);
            else
                printf(" %f", max_size[i]);
#endif
        }
        for(i = 0; i < VectorLength(priorbox->aspect_ratio()); ++i)
        {
            aspect_ratio.push_back(priorbox->aspect_ratio()->Get(i));
#ifdef PRT_PARAM
            if (0 == i)
                printf(" aspect_ratio: %f", aspect_ratio[i]);
            else
                printf(" %f", aspect_ratio[i]);
#endif
        }
        for(i = 0; i < VectorLength(priorbox->variance()); ++i)
        {
            variance.push_back(priorbox->variance()->Get(i));
#ifdef PRT_PARAM
            if (0 == i)
                printf(" variance: %f", variance[i]);
            else
                printf(" %f", variance[i]);
#endif
        }
        flip = priorbox->flip();
        clip = priorbox->clip();
        img_w = priorbox->img_w();
        img_h = priorbox->img_h();
        step_w = (float)priorbox->step_w();
        step_h = (float)priorbox->step_h();
        offset = priorbox->offset();
#ifdef PRT_PARAM
        printf(" flip : %d, clip: %d, img_w: %d, img_h: %d step_w: %d, step_h: %d, offset: %f\n",
               flip, clip, img_w, img_h, step_w, step_h, offset);
#endif
    }

    ~PriorBoxLayer()
    {
        min_size.clear();
        max_size.clear();
        aspect_ratio.clear();
        variance.clear();
    }

    int GenerateTopBlobs()
    {
        auto first_blob = _bottom_blobs[_bottom[0]];
        h = first_blob->height();
        w = first_blob->width();

        assert(2 == _bottom.size());
        auto second_blob = _bottom_blobs[_bottom[1]];
        if (img_w == 0)
            img_w = second_blob->width();
        if (img_h == 0)
            img_h = second_blob->height();
        if (step_w == 0)
            step_w = (float)img_w / w;
        if (step_h == 0)
            step_h = (float)img_h / h;

        num_prior = min_size.size() * aspect_ratio.size() + min_size.size();
        if (max_size.size() != 0)
            num_prior += min_size.size();
        if (flip)
            num_prior += min_size.size() * aspect_ratio.size();

        _top_blobs[_top[0]] = new Blob<float>(1, 2, 4 * w * h * num_prior, 1);
        _top_blobs[_top[0]]->_name = "Top";

        //printf("%s %s feature map size [%d, %d], img size [%d %d] step size [%d %d] %d, %d, %d, %d\n", _bottom[0].c_str(), _bottom[1].c_str(), w, h, img_w, img_h, step_w, step_h, min_size.size(), aspect_ratio.size(), max_size.size(), num_prior);
        return 0;
    }

    int Init()
    {
        //printf("[priorbox] %d %d %d %f %f %f\n", w, h, num_prior, step_w, step_h, offset);
        for (int i = 0; i < h; i++)
        {
            float* box = _top_blobs[_top[0]]->data() + i * w * num_prior * 4;
            float center_x = offset * step_w;
            float center_y = offset * step_h + i * step_h;

            for (int j = 0; j < w; j++)
            {
                float box_w, box_h;
                for (int k = 0; k < min_size.size(); k++)
                {
                    float min_size_l = min_size[k];

                    // min size box
                    box_w = box_h = min_size_l;

                    box[0] = (center_x - box_w * 0.5f) / img_w;
                    box[1] = (center_y - box_h * 0.5f) / img_h;
                    box[2] = (center_x + box_w * 0.5f) / img_w;
                    box[3] = (center_y + box_h * 0.5f) / img_h;

                    box += 4;

                    if (max_size.size() > 0)
                    {
                        float max_size_l = max_size[k];

                        // max size box
                        box_w = box_h = sqrt(min_size_l * max_size_l);

                        box[0] = (center_x - box_w * 0.5f) / img_w;
                        box[1] = (center_y - box_h * 0.5f) / img_h;
                        box[2] = (center_x + box_w * 0.5f) / img_w;
                        box[3] = (center_y + box_h * 0.5f) / img_h;

                        box += 4;
                    }

                    // all aspect_ratios
                    for (int p = 0; p < aspect_ratio.size(); p++)
                    {
                        float ar = aspect_ratio[p];

                        box_w = min_size_l * sqrt(ar);
                        box_h = min_size_l / sqrt(ar);

                        box[0] = (center_x - box_w * 0.5f) / img_w;
                        box[1] = (center_y - box_h * 0.5f) / img_h;
                        box[2] = (center_x + box_w * 0.5f) / img_w;
                        box[3] = (center_y + box_h * 0.5f) / img_h;

                        box += 4;

                        if (flip)
                        {
                            box[0] = (center_x - box_h * 0.5f) / img_w;
                            box[1] = (center_y - box_w * 0.5f) / img_h;
                            box[2] = (center_x + box_h * 0.5f) / img_w;
                            box[3] = (center_y + box_w * 0.5f) / img_h;

                            box += 4;
                        }
                    }
                }

                center_x += step_w;
            }

            center_y += step_h;
        }

        assert(1 == _top_blobs[_top[0]]->width());
        if (clip)
        {
            float* box = _top_blobs[_top[0]]->data();
            for (int i = 0; i < _top_blobs[_top[0]]->height(); i++)
                box[i] = std::min(std::max(box[i], 0.f), 1.f);
        }

        // channel 1 variance
        float* var = _top_blobs[_top[0]]->data() + _top_blobs[_top[0]]->height();
        for (int i = 0; i <  _top_blobs[_top[0]]->height() / 4; i++)
        {
            var[0] = variance[0];
            var[1] = variance[1];
            var[2] = variance[2];
            var[3] = variance[3];
            var += 4;
        }

        return 0;
    }

private:
    std::vector<float> min_size;
    std::vector<float> max_size;
    std::vector<float> aspect_ratio;
    bool flip;
    bool clip;
    std::vector<float> variance;
    uint32_t img_w;
    uint32_t img_h;
    float step_w;
    float step_h;
    float offset;
    int w;
    int h;
    int num_prior;
};
};
