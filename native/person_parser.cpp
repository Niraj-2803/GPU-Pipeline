#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cmath>

extern "C" bool NvDsInferParseYoloPerson(
    const std::vector<NvDsInferLayerInfo>& layers,
    const NvDsInferNetworkInfo& network,
    const NvDsInferParseDetectionParams& params,
    std::vector<NvDsInferParseObjectInfo>& objects) {
    if(layers.size()!=1 || layers[0].inferDims.numDims!=2 || layers[0].inferDims.d[1]!=6) return false;
    auto data=static_cast<const float*>(layers[0].buffer);
    for(int i=0;i<layers[0].inferDims.d[0];i++) {
        auto b=data+i*6;
        // Argmax would select class zero exactly when its score equals the maximum.
        if(!std::isfinite(b[4]) || b[4]<params.perClassPreclusterThreshold[0] || b[4]<b[5]) continue;
        float x1=std::clamp(b[0],0.0f,float(network.width)),y1=std::clamp(b[1],0.0f,float(network.height));
        float x2=std::clamp(b[2],0.0f,float(network.width)),y2=std::clamp(b[3],0.0f,float(network.height));
        if(x2-x1<1 || y2-y1<1)continue;
        NvDsInferParseObjectInfo obj{};
        obj.classId=0;obj.left=x1;obj.top=y1;obj.width=x2-x1;obj.height=y2-y1;obj.detectionConfidence=b[4];objects.push_back(obj);
    }
    return true;
}
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloPerson);
