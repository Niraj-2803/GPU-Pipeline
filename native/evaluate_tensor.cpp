#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>

struct Log: nvinfer1::ILogger {
  void log(Severity severity,const char* text) noexcept override {
    if(severity<=Severity::kWARNING)std::cerr<<text<<'\n';
  }
};
void check(cudaError_t result){if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));}
int main(int argc,char** argv){
  if(argc!=4){std::cerr<<"evaluate_tensor ENGINE INPUT_FLOAT32 OUTPUT_FLOAT32\n";return 2;}
  try {
    int device=std::getenv("CAMEX_GPU_ID")?std::atoi(std::getenv("CAMEX_GPU_ID")):0;
    check(cudaSetDevice(device));cudaDeviceProp props;check(cudaGetDeviceProperties(&props,device));
    std::cout<<"CUDA device "<<device<<": "<<props.name<<"\n";
    Log logger;std::ifstream file(argv[1],std::ios::binary);
    std::vector<char> plan((std::istreambuf_iterator<char>(file)),{});
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(plan.data(),plan.size()));
    if(!engine)throw std::runtime_error("Could not load engine");
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    context->setInputShape("input",nvinfer1::Dims4(1,3,640,640));
    const size_t in_size=3*640*640, out_size=8400*6;
    std::vector<float> input(in_size), output(out_size);
    void *dev_in=nullptr,*dev_out=nullptr;cudaStream_t stream;
    check(cudaMalloc(&dev_in,in_size*sizeof(float)));check(cudaMalloc(&dev_out,out_size*sizeof(float)));
    check(cudaStreamCreate(&stream));context->setTensorAddress("input",dev_in);context->setTensorAddress("output",dev_out);
    std::ifstream source(argv[2],std::ios::binary);std::ofstream target(argv[3],std::ios::binary);
    int count=0;
    while(source.read(reinterpret_cast<char*>(input.data()),in_size*sizeof(float))){
      check(cudaMemcpyAsync(dev_in,input.data(),in_size*sizeof(float),cudaMemcpyHostToDevice,stream));
      if(!context->enqueueV3(stream))throw std::runtime_error("Inference failed");
      check(cudaMemcpyAsync(output.data(),dev_out,out_size*sizeof(float),cudaMemcpyDeviceToHost,stream));
      check(cudaStreamSynchronize(stream));target.write(reinterpret_cast<char*>(output.data()),out_size*sizeof(float));count++;
    }
    cudaStreamDestroy(stream);cudaFree(dev_in);cudaFree(dev_out);
    std::cout<<"Evaluated "<<count<<" tensors\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
