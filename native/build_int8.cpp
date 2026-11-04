#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using namespace nvinfer1;
struct Log:ILogger{void log(Severity s,const char* m)noexcept override{if(s<=Severity::kINFO)std::cout<<m<<std::endl;}} logger;
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Calibration:IInt8MinMaxCalibrator{
  std::string folder,cache;int views,position=0;bool failed=false;void* device=nullptr;
  std::vector<float> host=std::vector<float>(8*3*640*640);
  Calibration(std::string f,std::string c,int v):folder(f),cache(c),views(v){check(cudaMalloc(&device,host.size()*4));}
  ~Calibration(){cudaFree(device);}
  int getBatchSize()const noexcept override{return 8;}
  bool getBatch(void* bindings[],const char*[],int)noexcept override{
    if(position>=views*10)return false;
    try{
      constexpr int pixels=640*640;
      for(int b=0;b<8;b++){
        int sample=(position+b)%(views*10),camera=sample%views+1,index=sample/views;
        char name[32];snprintf(name,sizeof(name),"/C%02d.rgb",camera);
        std::ifstream f(folder+name,std::ios::binary);f.seekg(index*pixels*3);
        std::vector<unsigned char> rgb(pixels*3);f.read(reinterpret_cast<char*>(rgb.data()),rgb.size());
        if(!f)throw std::runtime_error("Missing calibration pixels");
        for(int p=0;p<pixels;p++)for(int c=0;c<3;c++)host[b*3*pixels+c*pixels+p]=rgb[p*3+c]/255.f;
      }
      check(cudaMemcpy(device,host.data(),host.size()*4,cudaMemcpyHostToDevice));bindings[0]=device;
      position+=8;std::cout<<"CALIBRATE "<<position<<"/"<<views*10<<std::endl;return true;
    }catch(const std::exception& e){failed=true;std::cerr<<e.what()<<std::endl;return false;}
  }
  const void* readCalibrationCache(size_t& n)noexcept override{n=0;return nullptr;}
  void writeCalibrationCache(const void* p,size_t n)noexcept override{
    std::ofstream f(cache,std::ios::binary);f.write(static_cast<const char*>(p),n);
  }
};
int main(int argc,char**argv){try{
  if(argc!=6)throw std::runtime_error("build_int8 MODEL CALIBRATION_FOLDER ENGINE CUDA_DEVICE VIEWS");
  if(std::stoi(argv[5])<1)throw std::runtime_error("Calibration requires at least one camera view");
  check(cudaSetDevice(std::stoi(argv[4])));
  std::unique_ptr<IBuilder> b(createInferBuilder(logger));
  std::unique_ptr<INetworkDefinition> n(b->createNetworkV2(1U<<unsigned(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*n,logger));
  if(!parser->parseFromFile(argv[1],int(ILogger::Severity::kWARNING)))return 2;
  std::unique_ptr<IBuilderConfig> c(b->createBuilderConfig());
  c->setFlag(BuilderFlag::kINT8);c->setFlag(BuilderFlag::kFP16);c->setFlag(BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
  int head=0;
  for(int i=0;i<n->getNbLayers();i++){
    auto l=n->getLayer(i);auto t=l->getType();
    if(std::string(l->getName()).find("model.22")!=std::string::npos && l->getNbOutputs()>0 && l->getOutput(0)->getType()==DataType::kFLOAT &&
       (t==LayerType::kCONVOLUTION || t==LayerType::kACTIVATION || t==LayerType::kELEMENTWISE || t==LayerType::kSOFTMAX)){
      l->setPrecision(DataType::kHALF);head++;
      for(int j=0;j<l->getNbOutputs();j++)if(l->getOutput(j)->getType()==DataType::kFLOAT)l->setOutputType(j,DataType::kHALF);
    }
  }
  if(head==0)throw std::runtime_error("No detection-head layers matched; refusing an unintended precision strategy");
  c->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,2ULL<<30);
  c->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
  auto profile=b->createOptimizationProfile();
  profile->setDimensions("input",OptProfileSelector::kMIN,Dims4(1,3,640,640));
  profile->setDimensions("input",OptProfileSelector::kOPT,Dims4(8,3,640,640));
  profile->setDimensions("input",OptProfileSelector::kMAX,Dims4(16,3,640,640));c->addOptimizationProfile(profile);
  auto cp=b->createOptimizationProfile();
  for(auto s:{OptProfileSelector::kMIN,OptProfileSelector::kOPT,OptProfileSelector::kMAX})cp->setDimensions("input",s,Dims4(8,3,640,640));
  c->setCalibrationProfile(cp);
  Calibration calibration(argv[2],std::string(argv[3])+".calibration.cache",std::stoi(argv[5]));c->setInt8Calibrator(&calibration);
  std::unique_ptr<IHostMemory> plan(b->buildSerializedNetwork(*n,*c));
  if(!plan || calibration.failed || calibration.position<calibration.views*10)return 3;
  std::ofstream out(argv[3],std::ios::binary);out.write(static_cast<const char*>(plan->data()),plan->size());
  std::cout<<"INT8_COMPLETE head_layers="<<head<<" bytes="<<plan->size()<<std::endl;
  return out?0:4;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
