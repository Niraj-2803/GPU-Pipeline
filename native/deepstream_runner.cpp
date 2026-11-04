#include <gst/gst.h>
#include <glib-unix.h>
#include "gstnvdsmeta.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <memory>
#include <map>
#include <set>
#include <sstream>
#include <sys/resource.h>
#include <vector>
#include <deque>
#include <thread>
#include <stdexcept>

// Metadata leaves the probe through a bounded queue so a slow consumer cannot
// block the GPU pipeline. Dropped metadata is reported separately from frames.
struct MetadataWriter {
  std::mutex mutex; std::condition_variable ready; std::deque<std::string> queue;
  std::ofstream output; std::thread worker; bool closing=false;
  std::atomic<unsigned long> dropped{0};
  explicit MetadataWriter(const std::string& path):output(path) {
    if(!output)throw std::runtime_error("Cannot open metadata output");
    worker=std::thread([this]{
      std::unique_lock<std::mutex> lock(mutex);
      while(!closing || !queue.empty()) {
        ready.wait_for(lock,std::chrono::milliseconds(200),[this]{return closing||!queue.empty();});
        std::deque<std::string> batch; batch.swap(queue);lock.unlock();
        for(const auto& line:batch)output<<line<<'\n';output.flush();lock.lock();
      }
    });
  }
  void push(std::string line) {
    std::lock_guard<std::mutex> lock(mutex);
    if(queue.size()>=2048){dropped++;return;}
    queue.push_back(std::move(line));ready.notify_one();
  }
  ~MetadataWriter(){ {std::lock_guard<std::mutex> lock(mutex);closing=true;} ready.notify_one();worker.join(); }
};

struct Camera {
  guint64 last_pts=GST_CLOCK_TIME_NONE,last_detect_pts=GST_CLOCK_TIME_NONE;
  unsigned long frames=0, reported=0, persons=0, duplicates=0, sequence=0;
  unsigned long invalid_latency=0;
  unsigned long detected_frames=0,reported_detected=0,track_observations=0,unassigned=0;
  unsigned int current_tracks=0;
  std::map<guint64,unsigned int> display_ids;
  std::set<guint64> measured_track_ids;
  bool seen=false; double last_seen=0, fps=0;
  GstPad* timeline=nullptr; std::vector<double> latency;
  size_t reported_latency=0;
  std::map<guint64,double> ingested;
  long long last_sample_bucket=-1;
  double last_processed_wall=0;
  unsigned long tracking_epoch=0;
  std::map<guint64,std::pair<double,double>> stage_times;
};
struct State {
  GstElement* pipeline=nullptr; GMainLoop* loop=nullptr;
  std::mutex mutex; std::vector<Camera> cameras;
  int count=0,batch=8,cols=1,rows=1,tile_w=960,tile_h=540,input_w=1920,input_h=1080;
  int last_bucket=-1; unsigned long mosaic_frames=0,reported_mosaic=0;
  double launched=0,start=0,seconds=10,warmup=3,reported_at=0,finish_at=0,measured_seconds=0;
  double warmup_reported=0;
  bool output=false,tracking=false,error=false,finishing=false,interrupted=false,eos=false,prerolling=false;
  std::string folder,uri; int drop=3;
  int target_fps=10;
  int preview_fps=10,last_preview_bucket=-1;
  bool profile=false;
  double last_interval_p95=-1;
  std::vector<double> mux_age,infer_transit,tracker_transit,processing_transit,batch_sizes;
  std::atomic<double> sampling_start{0};
  std::atomic<bool> sampling_stop{false};
  std::unique_ptr<std::atomic_ulong[]> paced_inputs,sampled_out;
  std::unique_ptr<std::atomic_bool[]> decoder_ready;
  bool sync_start=false,start_failed=false;
  int decoded_sources=0;
  guint64 playback_epoch=GST_CLOCK_TIME_NONE;
  double startup_seconds=0;
  std::mutex startup_mutex;
  std::condition_variable startup_ready;
  std::vector<std::string> source_uris,source_labels;
  std::ofstream intervals;
  std::vector<std::string> track_samples;
  std::unique_ptr<MetadataWriter> metadata;
  std::string run_id="run";
};
static double now_s(){return g_get_monotonic_time()/1e6;}
static int selected_gpu=0;
struct SampleContext {State* state;int camera;};
struct SourceContext {State* state;int camera;GstElement* pace;bool arrived=false;};
static GstPadProbeReturn source_start(GstPad*,GstPadProbeInfo* info,gpointer data){
  auto& context=*static_cast<SourceContext*>(data);auto& s=*context.state;
  auto buffer=GST_PAD_PROBE_INFO_BUFFER(info);
  if(!buffer || !s.sync_start)return GST_PAD_PROBE_OK;
  {
    std::unique_lock<std::mutex> lock(s.startup_mutex);
    if(!context.arrived){context.arrived=true;s.decoded_sources++;}
    if(s.decoded_sources==s.count && !GST_CLOCK_TIME_IS_VALID(s.playback_epoch)){
      auto clock=gst_element_get_clock(s.pipeline);
      if(clock){
        s.playback_epoch=gst_clock_get_time(clock)-gst_element_get_base_time(s.pipeline)+100*GST_MSECOND;
        gst_object_unref(clock);s.startup_seconds=now_s()-s.launched;
        std::cout<<"SOURCE START all "<<s.count<<" decoders ready after "<<s.startup_seconds<<"s; common file-playback epoch="<<s.playback_epoch/1e9<<"s"<<std::endl;
      }else s.start_failed=true;
      s.startup_ready.notify_all();
    }
    if(!s.startup_ready.wait_for(lock,std::chrono::seconds(40),[&]{return GST_CLOCK_TIME_IS_VALID(s.playback_epoch)||s.start_failed;})){
      s.start_failed=true;s.startup_ready.notify_all();
    }
    if(s.start_failed)return GST_PAD_PROBE_DROP;
  }
  // One immutable offset assigns a common playback start to stored files.
  // Keep source spacing and every decoded frame; never reset the clock mid-run.
  buffer=gst_buffer_make_writable(buffer);GST_PAD_PROBE_INFO_DATA(info)=buffer;
  if(GST_CLOCK_TIME_IS_VALID(GST_BUFFER_PTS(buffer)))GST_BUFFER_PTS(buffer)+=s.playback_epoch;
  if(GST_CLOCK_TIME_IS_VALID(GST_BUFFER_DTS(buffer)))GST_BUFFER_DTS(buffer)+=s.playback_epoch;
  return GST_PAD_PROBE_OK;
}
static GstPadProbeReturn sample_frame(GstPad*,GstPadProbeInfo* info,gpointer data){
  auto& context=*static_cast<SampleContext*>(data);auto& s=*context.state;
  s.decoder_ready[context.camera].store(true,std::memory_order_relaxed);
  auto buffer=GST_PAD_PROBE_INFO_BUFFER(info);const auto pts=GST_BUFFER_PTS(buffer);
  const double wall=now_s();bool discard=false;
  {std::lock_guard<std::mutex> lock(s.mutex);auto& times=s.cameras[context.camera].ingested;
   times[pts]=wall;while(times.size()>128)times.erase(times.begin());}
  if(GST_CLOCK_TIME_IS_VALID(pts)) {
    auto& camera=s.cameras[context.camera];
    long long bucket=static_cast<long long>((static_cast<long double>(pts)/GST_SECOND)*s.target_fps+0.00001);
    discard=bucket<=camera.last_sample_bucket;
    if(!discard)camera.last_sample_bucket=bucket;
  }
  double start=s.sampling_start.load(std::memory_order_relaxed);
  if(start && wall>=start && wall<start+s.seconds && !s.sampling_stop.load(std::memory_order_relaxed)) {
    s.paced_inputs[context.camera].fetch_add(1,std::memory_order_relaxed);
    if(discard)s.sampled_out[context.camera].fetch_add(1,std::memory_order_relaxed);
  }
  return discard?GST_PAD_PROBE_DROP:GST_PAD_PROBE_OK;
}
static double percentile(std::vector<double> v,double q){
  if(v.empty())return -1;std::sort(v.begin(),v.end());return v[std::min(v.size()-1,size_t((v.size()-1)*q))];
}
static GstElement* make(const char* factory){
  auto e=gst_element_factory_make(factory,nullptr);
  if(!e){std::cerr<<"ERROR missing GStreamer element "<<factory<<std::endl;std::exit(2);}
  if(g_object_class_find_property(G_OBJECT_GET_CLASS(e),"gpu-id"))g_object_set(e,"gpu-id",selected_gpu,nullptr);
  return e;
}
static NvOSD_ColorParams track_color(unsigned int number){
  double h=std::fmod((number-1)*137.507764,360.0)/60.0;
  double c=.82,x=c*(1-std::abs(std::fmod(h,2.0)-1)),m=.18;
  double r=0,g=0,b=0;
  if(h<1){r=c;g=x;}else if(h<2){r=x;g=c;}else if(h<3){g=c;b=x;}
  else if(h<4){g=x;b=c;}else if(h<5){r=x;b=c;}else{r=c;b=x;}
  return {r+m,g+m,b+m,1.0};
}
static GstPadProbeReturn mux_profile(GstPad*,GstPadProbeInfo* info,gpointer data){
  auto& s=*static_cast<State*>(data);
  auto meta=gst_buffer_get_nvds_batch_meta(GST_PAD_PROBE_INFO_BUFFER(info));
  if(!meta)return GST_PAD_PROBE_OK;
  double wall=now_s();auto clock=gst_element_get_clock(s.pipeline);
  guint64 running=clock?gst_clock_get_time(clock)-gst_element_get_base_time(s.pipeline):0;
  if(clock)gst_object_unref(clock);
  std::lock_guard<std::mutex> lock(s.mutex);
  bool measuring=s.start && wall>=s.start && wall<s.start+s.seconds && !s.finishing;
  if(measuring)s.batch_sizes.push_back(meta->num_frames_in_batch);
  for(auto l=meta->frame_meta_list;l;l=l->next){
    auto f=static_cast<NvDsFrameMeta*>(l->data);if(f->source_id>=s.cameras.size())continue;
    auto& c=s.cameras[f->source_id];c.stage_times[f->buf_pts]={wall,0};
    if(c.stage_times.size()>64)c.stage_times.erase(c.stage_times.begin());
    if(measuring){
      auto event=gst_pad_get_sticky_event(c.timeline,GST_EVENT_SEGMENT,0);
      if(event){const GstSegment* segment=nullptr;gst_event_parse_segment(event,&segment);
        auto pts=gst_segment_to_running_time(segment,GST_FORMAT_TIME,f->buf_pts);
        if(GST_CLOCK_TIME_IS_VALID(pts) && running>=pts)s.mux_age.push_back((running-pts)/1e6);
        gst_event_unref(event);}
    }
  }
  return GST_PAD_PROBE_OK;
}
static GstPadProbeReturn detections(GstPad*,GstPadProbeInfo* info,gpointer data){
  auto& s=*static_cast<State*>(data);auto meta=gst_buffer_get_nvds_batch_meta(GST_PAD_PROBE_INFO_BUFFER(info));
  if(!meta)return GST_PAD_PROBE_OK;
  double wall=now_s();std::lock_guard<std::mutex> lock(s.mutex);
  for(auto l=meta->frame_meta_list;l;l=l->next){
    auto f=static_cast<NvDsFrameMeta*>(l->data);if(f->source_id>=s.cameras.size())continue;
    auto& c=s.cameras[f->source_id];
    bool measuring=s.start && wall>=s.start && wall<s.start+s.seconds && !s.finishing && f->bInferDone && c.last_detect_pts!=f->buf_pts;
    c.last_detect_pts=f->buf_pts;if(measuring)c.detected_frames++;
    if(s.profile){auto found=c.stage_times.find(f->buf_pts);if(found!=c.stage_times.end()){
      found->second.second=wall;
      if(measuring)s.infer_transit.push_back((wall-found->second.first)*1000);
    }}
    // Track people only. Keep this same filter in detection-only comparisons.
    for(auto o=f->obj_meta_list;o;){
      auto next=o->next;auto obj=static_cast<NvDsObjectMeta*>(o->data);
      if(obj->class_id!=0)nvds_remove_obj_meta_from_frame(f,obj);
      else if(measuring)c.persons++;
      o=next;
    }
  }
  return GST_PAD_PROBE_OK;
}
static void pad_added(GstElement*,GstPad* pad,gpointer target){
  auto& context=*static_cast<SourceContext*>(target);
  auto caps=gst_pad_get_current_caps(pad);if(!caps)caps=gst_pad_query_caps(pad,nullptr);
  bool video=caps && g_str_has_prefix(gst_structure_get_name(gst_caps_get_structure(caps,0)),"video/");
  if(caps)gst_caps_unref(caps);if(!video)return;
  if(context.state->sync_start)gst_pad_add_probe(pad,GST_PAD_PROBE_TYPE_BUFFER,source_start,&context,nullptr);
  auto sink=gst_element_get_static_pad(context.pace,"sink");
  if(!gst_pad_is_linked(sink) && gst_pad_link(pad,sink)!=GST_PAD_LINK_OK)std::cerr<<"ERROR source link failed"<<std::endl;
  gst_object_unref(sink);
}
static GstPadProbeReturn inference(GstPad*,GstPadProbeInfo* info,gpointer data){
  auto& s=*static_cast<State*>(data);auto buffer=GST_PAD_PROBE_INFO_BUFFER(info);
  auto meta=gst_buffer_get_nvds_batch_meta(buffer);if(!meta)return GST_PAD_PROBE_OK;
  const double wall=now_s();auto clock=gst_element_get_clock(s.pipeline);
  guint64 running=clock?gst_clock_get_time(clock)-gst_element_get_base_time(s.pipeline):0;
  if(clock)gst_object_unref(clock);
  std::lock_guard<std::mutex> lock(s.mutex);
  for(auto l=meta->frame_meta_list;l;l=l->next){
    auto f=static_cast<NvDsFrameMeta*>(l->data);if(f->source_id>=s.cameras.size())continue;
    auto& c=s.cameras[f->source_id];c.seen=true;c.last_seen=wall;c.sequence=f->frame_num;
    if(c.last_processed_wall && wall-c.last_processed_wall>2.0)c.tracking_epoch++;
    c.last_processed_wall=wall;
    if(!s.prerolling && !s.start && std::all_of(s.cameras.begin(),s.cameras.end(),[](const Camera& c){return c.seen;})){
      s.start=wall+s.warmup;s.reported_at=s.start;
      s.sampling_start.store(s.start,std::memory_order_relaxed);
      std::cout<<"READY all "<<s.count<<" sources producing output; warming up for "<<s.warmup<<" seconds"<<std::endl;
    }
    bool measuring=s.start && wall>=s.start && wall<s.start+s.seconds && !s.finishing;
    bool unique=f->buf_pts!=c.last_pts;
    if(s.profile){auto found=c.stage_times.find(f->buf_pts);if(found!=c.stage_times.end()){
      if(measuring && unique){
        s.processing_transit.push_back((wall-found->second.first)*1000);
        if(found->second.second)s.tracker_transit.push_back((wall-found->second.second)*1000);
      }
      c.stage_times.erase(found);
    }}
    if(measuring && !unique)c.duplicates++;
    c.last_pts=f->buf_pts;
    if(measuring && unique && f->bInferDone){
      c.frames++;
      guint64 presentation=GST_CLOCK_TIME_NONE;
      auto event=gst_pad_get_sticky_event(c.timeline,GST_EVENT_SEGMENT,0);
      if(event){const GstSegment* segment=nullptr;gst_event_parse_segment(event,&segment);
        presentation=gst_segment_to_running_time(segment,GST_FORMAT_TIME,f->buf_pts);gst_event_unref(event);}
      if(GST_CLOCK_TIME_IS_VALID(presentation) && running>=presentation)c.latency.push_back((running-presentation)/1e6);
      else c.invalid_latency++;
    }
    c.current_tracks=0;
    std::ostringstream sample;
    bool capture=s.tracking && s.output && measuring && unique && f->source_id==0;
    if(capture)sample<<"{\"source_id\":1,\"source_camera\":\""<<s.source_labels[0]<<"\",\"sequence\":"<<c.sequence<<",\"elapsed\":"<<wall-s.start<<",\"tracks\":[";
    bool sample_comma=false;
    std::ostringstream event;bool event_comma=false;
    double processing_ms=-1;
    auto arrival=c.ingested.find(f->buf_pts);
    if(arrival!=c.ingested.end()){processing_ms=(wall-arrival->second)*1000;c.ingested.erase(arrival);}
    const bool publish=s.metadata && measuring && unique && f->bInferDone;
    if(publish)event<<std::fixed<<std::setprecision(6)<<"{\"schema_version\":\"1.0\",\"type\":\"frame\",\"run_id\":\""<<s.run_id
      <<"\",\"backend\":\"deepstream\",\"camera_id\":\""<<s.source_labels[f->source_id]<<"\",\"frame_id\":"<<c.sequence
      <<",\"pts_ns\":"<<f->buf_pts<<",\"source_pts_ns\":"<<(s.sync_start && f->buf_pts>=s.playback_epoch?f->buf_pts-s.playback_epoch:f->buf_pts)
      <<",\"timestamp_domain\":\""<<(s.sync_start?"media":"pipeline")<<"\",\"monotonic_s\":"<<wall
      <<",\"elapsed_s\":"<<wall-s.start<<",\"width\":"<<s.input_w<<",\"height\":"<<s.input_h
      <<",\"tracking_epoch\":"<<c.tracking_epoch<<",\"processing_ms\":"<<processing_ms<<",\"tracker\":\""<<(s.tracking?"nvdcf":"none")<<"\",\"objects\":[";
    for(auto o=f->obj_meta_list;o;o=o->next){
      auto obj=static_cast<NvDsObjectMeta*>(o->data);
      bool assigned=s.tracking && obj->object_id!=UNTRACKED_OBJECT_ID;
      unsigned int display_id=0;
      if(assigned){
        auto found=c.display_ids.emplace(obj->object_id,c.display_ids.size()+1);
        display_id=found.first->second;c.current_tracks++;
        if(measuring && unique){c.track_observations++;c.measured_track_ids.insert(obj->object_id);}
      }else if(s.tracking && measuring && unique)c.unassigned++;
      if(publish){
        if(event_comma)event<<",";event_comma=true;
        auto r=obj->rect_params;
        const auto x1=std::clamp(double(r.left),0.0,double(s.input_w));
        const auto y1=std::clamp(double(r.top),0.0,double(s.input_h));
        const auto x2=std::clamp(double(r.left+r.width),x1,double(s.input_w));
        const auto y2=std::clamp(double(r.top+r.height),y1,double(s.input_h));
        event<<"{\"class_id\":0,\"label\":\"person\",\"confidence\":"<<(std::isfinite(obj->confidence)?obj->confidence:-1)
          <<",\"track_id\":"<<(assigned?"\""+std::to_string(obj->object_id)+"\"":"null")
          <<",\"bbox_xyxy\":["<<x1<<","<<y1<<","<<x2<<","<<y2<<"]}";
      }
      auto color=assigned?track_color(display_id):(s.tracking?NvOSD_ColorParams{.6,.6,.6,1}:NvOSD_ColorParams{.1,1,.25,1});
      if(s.output){
        obj->rect_params.border_width=3;
        obj->rect_params.border_color=color;
        g_free(obj->text_params.display_text);
        obj->text_params.display_text=s.tracking?(assigned?g_strdup_printf("T%02u",display_id):g_strdup("pending")):g_strdup("");
        obj->text_params.x_offset=std::max(0,int(obj->rect_params.left));
        obj->text_params.y_offset=std::max(0,int(obj->rect_params.top));
        obj->text_params.font_params.font_name=const_cast<char*>("Sans");
        obj->text_params.font_params.font_size=std::max(9,std::min(12,s.tile_w/40));
        obj->text_params.font_params.font_color={1,1,1,1};
        obj->text_params.set_bg_clr=1;obj->text_params.text_bg_clr={0,0,0,.8};
      }
      if(capture && assigned){
        auto r=obj->rect_params;
        if(sample_comma)sample<<",";sample_comma=true;
        sample<<"{\"tracker_id\":\""<<obj->object_id<<"\",\"display_id\":"<<display_id<<",\"box\":["<<r.left<<","<<r.top<<","<<r.width<<","<<r.height
              <<"],\"rgb\":["<<color.red<<","<<color.green<<","<<color.blue<<"],\"tracker_confidence\":"<<obj->tracker_confidence<<"}";
      }
    }
    if(capture){sample<<"]}";s.track_samples.push_back(sample.str());}
    if(publish){event<<"]}";s.metadata->push(event.str());}
  }
  return GST_PAD_PROBE_OK;
}
static GstPadProbeReturn preview_gate(GstPad*,GstPadProbeInfo*,gpointer data){
  auto& s=*static_cast<State*>(data);const double wall=now_s();std::lock_guard<std::mutex> lock(s.mutex);
  // Initialize the tiler and converters during warm-up before reducing preview work.
  if(!s.start || wall<s.start)return GST_PAD_PROBE_OK;
  if(wall>=s.start+s.seconds || s.finishing)return GST_PAD_PROBE_DROP;
  int bucket=int((wall-s.start)*s.preview_fps);
  if(bucket<=s.last_preview_bucket)return GST_PAD_PROBE_DROP;
  s.last_preview_bucket=bucket;return GST_PAD_PROBE_OK;
}
static GstPadProbeReturn mosaic(GstPad*,GstPadProbeInfo* info,gpointer data){
  auto& s=*static_cast<State*>(data);const double wall=now_s();std::lock_guard<std::mutex> lock(s.mutex);
  if(!s.start || wall<s.start || wall>=s.start+s.seconds || s.finishing)return GST_PAD_PROBE_DROP;
  int bucket=int((wall-s.start)*s.preview_fps);
  if(bucket<=s.last_bucket)return GST_PAD_PROBE_DROP;
  s.last_bucket=bucket;
  auto buffer=GST_PAD_PROBE_INFO_BUFFER(info);
  buffer=gst_buffer_make_writable(buffer);GST_PAD_PROBE_INFO_DATA(info)=buffer;
  GST_BUFFER_PTS(buffer)=guint64(bucket)*GST_SECOND/s.preview_fps;
  GST_BUFFER_DTS(buffer)=GST_CLOCK_TIME_NONE;GST_BUFFER_DURATION(buffer)=GST_SECOND/s.preview_fps;
  auto meta=gst_buffer_get_nvds_batch_meta(buffer);
  if(meta && meta->frame_meta_list){
    auto frame=static_cast<NvDsFrameMeta*>(meta->frame_meta_list->data);
    for(int offset=0;offset<s.count;offset+=MAX_ELEMENTS_IN_DISPLAY_META){
      auto display=nvds_acquire_display_meta_from_pool(meta);
      for(int i=offset;i<std::min(s.count,offset+MAX_ELEMENTS_IN_DISPLAY_META);i++){
        auto& c=s.cameras[i];auto& t=display->text_params[display->num_labels++];
        bool stale=!c.seen || wall-c.last_seen>.5;
        t.display_text=s.tracking?g_strdup_printf("S%03d %s %.1ffps %ut%s",i+1,s.source_labels[i].c_str(),c.fps,c.current_tracks,stale?" STALE":""):
                                   g_strdup_printf("S%03d %s %.1ffps DET%s",i+1,s.source_labels[i].c_str(),c.fps,stale?" STALE":"");
        t.x_offset=(i%s.cols)*s.tile_w+6;t.y_offset=(i/s.cols)*s.tile_h+6;
        t.font_params.font_name=const_cast<char*>("Sans");t.font_params.font_size=std::max(11,std::min(20,s.tile_w/22));
        t.font_params.font_color=stale?NvOSD_ColorParams{1,.25,.2,1}:NvOSD_ColorParams{1,1,1,1};
        t.set_bg_clr=1;t.text_bg_clr={0,0,0,.85};
      }
      nvds_add_display_meta_to_frame(frame,display);
    }
  }
  s.mosaic_frames++;
  return GST_PAD_PROBE_OK;
}
static void report(State& s,double end){
  double duration=end-s.reported_at;if(duration<=0)return;
  double total=0,detector_total=0,low=1e9,high=0;int active=0,tracks=0;std::vector<double> latency,interval_latency;
  s.intervals<<"{\"elapsed\":"<<end-s.start<<",\"interval_seconds\":"<<duration<<",\"fps\":[";
  for(int i=0;i<s.count;i++){
    auto& c=s.cameras[i];c.fps=(c.frames-c.reported)/duration;c.reported=c.frames;
    detector_total+=(c.detected_frames-c.reported_detected)/duration;c.reported_detected=c.detected_frames;tracks+=c.current_tracks;
    total+=c.fps;low=std::min(low,c.fps);high=std::max(high,c.fps);active+=c.fps>0;
    latency.insert(latency.end(),c.latency.begin(),c.latency.end());
    interval_latency.insert(interval_latency.end(),c.latency.begin()+c.reported_latency,c.latency.end());
    c.reported_latency=c.latency.size();
    if(i)s.intervals<<",";s.intervals<<c.fps;
  }
  s.last_interval_p95=percentile(interval_latency,.95);
  s.intervals<<"],\"interval_p95_ms\":"<<s.last_interval_p95<<",\"detector_total_fps\":"<<detector_total<<",\"current_tracks\":"<<tracks<<",\"mosaic_frames\":"<<s.mosaic_frames<<"}\n";s.intervals.flush();
  std::cout<<std::fixed<<std::setprecision(1)<<"FPS "<<std::setw(4)<<end-s.start<<"/"<<s.seconds<<"s | "<<(s.tracking?"det+track=":"detection=")<<total
           <<" | per stream min/avg/max="<<low<<"/"<<total/s.count<<"/"<<high<<" | active="<<active<<"/"<<s.count
           <<" | cumulative P95="<<percentile(latency,.95)<<"ms | interval P95="<<s.last_interval_p95<<"ms";
  if(s.tracking)std::cout<<" | detector="<<detector_total<<" FPS | tracks="<<tracks;
  if(s.output)std::cout<<" | mosaic="<<(s.mosaic_frames-s.reported_mosaic)/duration<<" FPS";
  std::cout<<std::endl;s.reported_mosaic=s.mosaic_frames;s.reported_at=end;
}
static gboolean bus(GstBus*,GstMessage* message,gpointer data){
  auto& s=*static_cast<State*>(data);
  if(GST_MESSAGE_TYPE(message)==GST_MESSAGE_ERROR){
    GError* error=nullptr;gchar* debug=nullptr;gst_message_parse_error(message,&error,&debug);
    std::cerr<<"ERROR "<<error->message<<" "<<(debug?debug:"")<<std::endl;
    g_error_free(error);g_free(debug);s.error=true;g_main_loop_quit(s.loop);
  }else if(GST_MESSAGE_TYPE(message)==GST_MESSAGE_EOS){
    s.eos=true;if(!s.finishing){s.error=true;std::cerr<<"ERROR unexpected early EOS"<<std::endl;}
    g_main_loop_quit(s.loop);
  }return TRUE;
}
static gboolean interrupted(gpointer data){static_cast<State*>(data)->interrupted=true;return G_SOURCE_CONTINUE;}
static gboolean tick(gpointer data){
  auto& s=*static_cast<State*>(data);bool finish=false;const double wall=now_s();
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    if(s.start && wall<s.start && wall-s.warmup_reported>=5) {
      std::cout<<"WARMUP remaining="<<std::ceil(s.start-wall)<<"s"<<std::endl;s.warmup_reported=wall;
    }
    if(!s.finishing && s.start && wall>=s.start){
      double end=std::min(wall,s.start+s.seconds);
      if(end-s.reported_at>=1 || end>=s.start+s.seconds)report(s,end);
    }
    if(!s.finishing && ((s.start && wall>=s.start+s.seconds) || s.interrupted)){
      s.finishing=true;s.sampling_stop.store(true,std::memory_order_relaxed);s.finish_at=wall;s.measured_seconds=s.start?std::clamp(wall-s.start,0.0,s.seconds):0;finish=true;
    }
    if((!s.start && wall-s.launched>45) || (s.finishing && wall-s.finish_at>10)){
      s.error=true;std::cerr<<"ERROR startup or output-finalization timeout"<<std::endl;g_main_loop_quit(s.loop);return G_SOURCE_REMOVE;
    }
  }
  if(finish){
    std::cout<<"FINISH measurement stopped; finalizing output"<<std::endl;
    if(s.output)gst_element_send_event(s.pipeline,gst_event_new_eos());else g_main_loop_quit(s.loop);
  }
  return G_SOURCE_CONTINUE;
}
int main(int argc,char** argv){
  // Each decoder needs several descriptors. Change this process's soft limit
  // only; retain the existing hard limit and all host-wide configuration.
  struct rlimit files;
  if(getrlimit(RLIMIT_NOFILE,&files)==0 && files.rlim_cur<8192){
    files.rlim_cur=std::min(rlim_t(8192),files.rlim_max);
    if(setrlimit(RLIMIT_NOFILE,&files)!=0){std::cerr<<"ERROR cannot raise benchmark file limit"<<std::endl;return 2;}
  }
  gst_init(&argc,&argv);
  if(std::getenv("CAMEX_GPU_ID"))selected_gpu=std::stoi(std::getenv("CAMEX_GPU_ID"));
  if(argc!=8 && argc!=10 && argc!=11 && argc!=12 && argc!=13){std::cerr<<"interactive N CONFIG FOLDER SOURCE_URI DROP OUTPUT SECONDS [WIDTH HEIGHT [TRACK [SOURCES_TSV [WARMUP]]]]"<<std::endl;return 2;}
  State s;s.count=std::stoi(argv[1]);s.folder=argv[3];s.uri=argv[4];s.drop=std::stoi(argv[5]);s.output=std::stoi(argv[6]);s.seconds=std::stod(argv[7]);
  if(std::getenv("CAMEX_RUN_ID"))s.run_id=std::getenv("CAMEX_RUN_ID");
  for(char c:s.run_id)if(!g_ascii_isalnum(c)&&c!='-'&&c!='_')return 2;
  s.metadata=std::make_unique<MetadataWriter>(s.folder+"/metadata.jsonl");
  s.profile=std::getenv("CAMEX_PROFILE_STAGES") && std::string(std::getenv("CAMEX_PROFILE_STAGES"))=="1";
  s.sync_start=std::getenv("CAMEX_SYNC_START") && std::string(std::getenv("CAMEX_SYNC_START"))=="1";
  if(std::getenv("CAMEX_MOSAIC_FPS"))s.preview_fps=std::stoi(std::getenv("CAMEX_MOSAIC_FPS"));
  if(s.preview_fps!=5 && s.preview_fps!=10)return 2;
  if(std::getenv("CAMEX_ANALYSIS_FPS"))s.target_fps=std::stoi(std::getenv("CAMEX_ANALYSIS_FPS"));
  if(s.target_fps<1 || s.target_fps>30)return 2;
  if(std::getenv("CAMEX_INFER_BATCH")) s.batch=std::stoi(std::getenv("CAMEX_INFER_BATCH"));
  if(argc>=10){s.input_w=std::stoi(argv[8]);s.input_h=std::stoi(argv[9]);}
  if(argc>=11)s.tracking=std::stoi(argv[10])!=0;
  if(argc==13)s.warmup=std::stod(argv[12]);
  if(s.warmup<0 || s.warmup>90)return 2;
  if(!((s.input_w==1920 && s.input_h==1080) || (s.input_w==2560 && s.input_h==1440)))return 2;
  if(s.count<1 || s.count>400 || s.seconds<1 || s.seconds>86400 || s.drop<1 || s.drop>30)return 2;
  if(argc>=12){
    std::ifstream sources(argv[11]);std::string line;
    while(std::getline(sources,line)){
      auto split=line.find('\t');
      if(split==std::string::npos || (line.rfind("file:///sources/",0)!=0 && line.rfind("rtsp://",0)!=0 && line.rfind("rtsps://",0)!=0)){std::cerr<<"ERROR invalid source manifest"<<std::endl;return 2;}
      auto tag=line.substr(split+1);
      if(tag.empty() || tag.size()>64)return 2;
      for(char c:tag)if(!g_ascii_isalnum(c)&&c!='-'&&c!='_')return 2;
      s.source_uris.push_back(line.substr(0,split));s.source_labels.push_back(tag);
    }
    if(s.source_uris.size()!=size_t(s.count)){std::cerr<<"ERROR source manifest count mismatch"<<std::endl;return 2;}
  }else{s.source_uris.assign(s.count,s.uri);s.source_labels.assign(s.count,"C18");}
  s.cameras.resize(s.count);s.intervals.open(s.folder+"/per_second.jsonl");if(!s.intervals)return 2;
  s.paced_inputs.reset(new std::atomic_ulong[s.count]);s.sampled_out.reset(new std::atomic_ulong[s.count]);
  s.decoder_ready.reset(new std::atomic_bool[s.count]);
  for(int i=0;i<s.count;i++){s.paced_inputs[i].store(0);s.sampled_out[i].store(0);s.decoder_ready[i].store(false);}
  s.pipeline=gst_pipeline_new("camex-interactive");s.loop=g_main_loop_new(nullptr,FALSE);
  auto mux=make("nvstreammux"),infer=make("nvinfer");
  // NvTracker sizes its stream/target capacity from the upstream mux batch.
  // NvInfer independently splits these buffers into TensorRT batches of eight.
  int mux_w=std::getenv("CAMEX_MUX_W")?std::stoi(std::getenv("CAMEX_MUX_W")):s.input_w;
  int mux_h=std::getenv("CAMEX_MUX_H")?std::stoi(std::getenv("CAMEX_MUX_H")):s.input_h;
  int mux_timeout=std::getenv("CAMEX_MUX_TIMEOUT")?std::stoi(std::getenv("CAMEX_MUX_TIMEOUT")):10000;
  g_object_set(mux,"batch-size",s.count,"width",mux_w,"height",mux_h,"live-source",TRUE,"batched-push-timeout",mux_timeout,"sync-inputs",FALSE,nullptr);
  g_object_set(infer,"config-file-path",argv[2],"batch-size",s.batch,nullptr);
  gst_bin_add_many(GST_BIN(s.pipeline),mux,infer,nullptr);if(!gst_element_link(mux,infer))return 3;
  if(s.profile){auto pad=gst_element_get_static_pad(mux,"src");gst_pad_add_probe(pad,GST_PAD_PROBE_TYPE_BUFFER,mux_profile,&s,nullptr);gst_object_unref(pad);}
  GstElement* processed=infer;
  auto detection_pad=gst_element_get_static_pad(infer,"src");gst_pad_add_probe(detection_pad,GST_PAD_PROBE_TYPE_BUFFER,detections,&s,nullptr);gst_object_unref(detection_pad);
  if(s.tracking){
    auto tracker=make("nvtracker");
    g_object_set(tracker,"tracker-width",640,"tracker-height",384,"gpu-id",selected_gpu,"display-tracking-id",FALSE,
      "ll-lib-file","/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
      "ll-config-file",std::getenv("CAMEX_TRACKER_CONFIG")?std::getenv("CAMEX_TRACKER_CONFIG"):"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml",nullptr);
    int groups=std::getenv("CAMEX_TRACKER_GROUPS")?std::stoi(std::getenv("CAMEX_TRACKER_GROUPS")):1;
    if(groups>1) {
      if(!g_object_class_find_property(G_OBJECT_GET_CLASS(tracker),"sub-batches"))return 3;
      std::ostringstream batches;
      for(int group=0;group<groups;group++) {
        if(group)batches<<";";
        for(int i=group;i<s.count;i+=groups){if(i!=group)batches<<",";batches<<i;}
      }
      g_object_set(tracker,"sub-batches",batches.str().c_str(),nullptr);
    }
    gst_bin_add(GST_BIN(s.pipeline),tracker);if(!gst_element_link(infer,tracker))return 3;processed=tracker;
  }
  if(!s.output){
    auto sink=make("fakesink");g_object_set(sink,"sync",FALSE,"async",FALSE,"enable-last-sample",FALSE,nullptr);
    gst_bin_add(GST_BIN(s.pipeline),sink);if(!gst_element_link(processed,sink))return 3;
  }else{
    s.cols=int(std::ceil(std::sqrt(s.count)));s.rows=(s.count+s.cols-1)/s.cols;
    s.tile_w=std::min(960,(3840/s.cols/32)*32);s.tile_h=s.tile_w*9/16;
    auto tiler=make("nvmultistreamtiler"),convert=make("nvvideoconvert"),rgba=make("capsfilter"),osd=make("nvdsosd"),convert2=make("nvvideoconvert"),nv12=make("capsfilter");
    auto encoder=make("nvv4l2h264enc"),parser=make("h264parse"),mp4=make("qtmux"),sink=make("filesink");
    g_object_set(tiler,"rows",s.rows,"columns",s.cols,"width",s.cols*s.tile_w,"height",s.rows*s.tile_h,nullptr);
    auto caps=gst_caps_from_string("video/x-raw(memory:NVMM),format=RGBA");g_object_set(rgba,"caps",caps,nullptr);gst_caps_unref(caps);
    caps=gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12");g_object_set(nv12,"caps",caps,nullptr);gst_caps_unref(caps);
    g_object_set(osd,"process-mode",1,"display-text",TRUE,nullptr);
    g_object_set(encoder,"bitrate",12000000,"iframeinterval",10,"preset-id",1,"tuning-info-id",2,nullptr);
    g_object_set(sink,"location",(s.folder+"/mosaic.partial.mp4").c_str(),"sync",FALSE,"async",FALSE,nullptr);
    gst_bin_add_many(GST_BIN(s.pipeline),tiler,convert,rgba,osd,convert2,nv12,encoder,parser,mp4,sink,nullptr);
    if(!gst_element_link_many(processed,tiler,convert,rgba,osd,convert2,nv12,encoder,parser,mp4,sink,nullptr))return 3;
    if(s.preview_fps<10){auto gate=gst_element_get_static_pad(tiler,"sink");gst_pad_add_probe(gate,GST_PAD_PROBE_TYPE_BUFFER,preview_gate,&s,nullptr);gst_object_unref(gate);}
    auto pad=gst_element_get_static_pad(tiler,"src");gst_pad_add_probe(pad,GST_PAD_PROBE_TYPE_BUFFER,mosaic,&s,nullptr);gst_object_unref(pad);
    std::cout<<"VIDEO "<<s.cols*s.tile_w<<"x"<<s.rows*s.tile_h<<" grid="<<s.cols<<"x"<<s.rows<<" target="<<s.preview_fps<<" mosaic FPS; AI target="<<s.target_fps<<" FPS; "<<(s.tracking?"person track IDs and stable colours":"person detections only")<<std::endl;
  }
  for(int i=0;i<s.count;i++){
    auto source=make("nvurisrcbin"),pace=make("identity"),queue=make("queue");
    int surfaces=std::getenv("CAMEX_DECODE_SURFACES")?std::stoi(std::getenv("CAMEX_DECODE_SURFACES")):2;
    const bool rtsp=s.source_uris[i].rfind("rtsp",0)==0;
    const bool loop_files=std::getenv("CAMEX_LOOP_FILES") && std::string(std::getenv("CAMEX_LOOP_FILES"))=="1";
    g_object_set(source,"uri",s.source_uris[i].c_str(),"file-loop",!rtsp && loop_files,"disable-audio",TRUE,"drop-frame-interval",s.drop,"num-extra-surfaces",surfaces,nullptr);
    if(rtsp){
      if(g_object_class_find_property(G_OBJECT_GET_CLASS(source),"select-rtp-protocol"))g_object_set(source,"select-rtp-protocol",4,nullptr);
      if(g_object_class_find_property(G_OBJECT_GET_CLASS(source),"latency"))g_object_set(source,"latency",100,nullptr);
      if(g_object_class_find_property(G_OBJECT_GET_CLASS(source),"rtsp-reconnect-interval"))g_object_set(source,"rtsp-reconnect-interval",5,nullptr);
      if(g_object_class_find_property(G_OBJECT_GET_CLASS(source),"rtsp-reconnect-attempts"))g_object_set(source,"rtsp-reconnect-attempts",-1,nullptr);
    }
    g_object_set(pace,"sync",TRUE,nullptr);g_object_set(queue,"max-size-buffers",2,"max-size-bytes",0,"max-size-time",guint64(0),"leaky",2,nullptr);
    gst_bin_add_many(GST_BIN(s.pipeline),source,pace,queue,nullptr);s.cameras[i].timeline=gst_element_get_static_pad(pace,"src");
    gst_pad_add_probe(s.cameras[i].timeline,GST_PAD_PROBE_TYPE_BUFFER,sample_frame,new SampleContext{&s,i},[](gpointer data){delete static_cast<SampleContext*>(data);});
    auto context=new SourceContext{&s,i,pace};
    g_signal_connect_data(source,"pad-added",G_CALLBACK(pad_added),context,[](gpointer data,GClosure*){delete static_cast<SourceContext*>(data);},GConnectFlags(0));
    if(!gst_element_link(pace,queue))return 3;
    auto out=gst_element_get_static_pad(queue,"src"),in=gst_element_request_pad_simple(mux,("sink_"+std::to_string(i)).c_str());
    if(gst_pad_link(out,in)!=GST_PAD_LINK_OK)return 3;gst_object_unref(out);gst_object_unref(in);
  }
  auto pad=gst_element_get_static_pad(processed,"src");gst_pad_add_probe(pad,GST_PAD_PROBE_TYPE_BUFFER,inference,&s,nullptr);gst_object_unref(pad);
  auto b=gst_pipeline_get_bus(GST_PIPELINE(s.pipeline));gst_bus_add_watch(b,bus,&s);gst_object_unref(b);
  g_timeout_add(20,tick,&s);g_unix_signal_add(SIGINT,interrupted,&s);g_unix_signal_add(SIGTERM,interrupted,&s);s.launched=now_s();
  std::cout<<"START "<<s.count<<" independent decoders | input="<<s.input_w<<"x"<<s.input_h<<" | mux="<<mux_w<<"x"<<mux_h<<" | analysis=640x640 | detector batch="<<s.batch<<" | stream capacity="<<s.count<<" | tracker="<<(s.tracking?"NvDCF 640x384":"off")<<" | output="<<(s.output?"yes":"no")<<" | timed window="<<s.seconds<<"s"<<std::endl;
  if(std::getenv("CAMEX_PREROLL") && std::string(std::getenv("CAMEX_PREROLL"))=="1") {
    s.prerolling=true;
    if(gst_element_set_state(s.pipeline,GST_STATE_PAUSED)==GST_STATE_CHANGE_FAILURE)return 3;
    GstState actual,pending;auto status=gst_element_get_state(s.pipeline,&actual,&pending,20*GST_SECOND);
    std::cout<<"PREROLL state="<<actual<<" pending="<<pending<<" status="<<status<<std::endl;
    if(status==GST_STATE_CHANGE_FAILURE || status==GST_STATE_CHANGE_ASYNC)return 3;
    // async=false sinks can report PAUSED before source decoders have loaded.
    // Wait for a decoded first buffer from every source before starting the clock.
    double deadline=now_s()+40;int ready=0;
    do {
      ready=0;for(int i=0;i<s.count;i++)ready+=s.decoder_ready[i].load(std::memory_order_relaxed);
      if(ready==s.count)break;
      g_usleep(10000);
    }while(now_s()<deadline);
    std::cout<<"PREROLL decoded sources="<<ready<<"/"<<s.count<<std::endl;
    if(ready!=s.count){gst_element_set_state(s.pipeline,GST_STATE_NULL);return 3;}
    {std::lock_guard<std::mutex> lock(s.mutex);s.prerolling=false;}
    s.launched=now_s();
  }
  if(gst_element_set_state(s.pipeline,GST_STATE_PLAYING)==GST_STATE_CHANGE_FAILURE)return 3;
  g_main_loop_run(s.loop);gst_element_set_state(s.pipeline,GST_STATE_NULL);
  if(!s.measured_seconds && s.start)s.measured_seconds=std::clamp(now_s()-s.start,0.0,s.seconds);
  double total=0,detector_total=0,low=1e9;std::vector<double> latency;std::ofstream result(s.folder+"/result.json");
  result<<std::fixed<<std::setprecision(6)<<"{\"cameras\":"<<s.count<<",\"mux_width\":"<<mux_w<<",\"mux_height\":"<<mux_h<<",\"warmup_seconds\":"<<s.warmup<<",\"measurement_start_monotonic\":"<<s.start<<",\"seconds\":"<<s.measured_seconds<<",\"output\":"<<(s.output?"true":"false")
        <<",\"stream_capacity\":"<<s.count<<",\"analysis_fps\":"<<s.target_fps<<",\"detector_batch\":"<<s.batch<<",\"tracking_enabled\":"<<(s.tracking?"true":"false")<<",\"error\":"<<(s.error?"true":"false")<<",\"interrupted\":"<<(s.interrupted?"true":"false")<<",\"mosaic_frames\":"<<s.mosaic_frames
        <<",\"mosaic_target_fps\":"<<s.preview_fps<<",\"mosaic_width\":"<<s.cols*s.tile_w<<",\"mosaic_height\":"<<s.rows*s.tile_h<<",\"per_camera\":[";
  for(int i=0;i<s.count;i++){
    auto& c=s.cameras[i];double fps=s.measured_seconds>0?c.frames/s.measured_seconds:0;low=std::min(low,fps);total+=fps;latency.insert(latency.end(),c.latency.begin(),c.latency.end());
    double detector_fps=s.measured_seconds>0?c.detected_frames/s.measured_seconds:0;detector_total+=detector_fps;
    if(i)result<<",";result<<"{\"id\":"<<i+1<<",\"frames\":"<<c.frames<<",\"fps\":"<<fps<<",\"detector_frames\":"<<c.detected_frames<<",\"detector_fps\":"<<detector_fps
       <<",\"person_detections\":"<<c.persons<<",\"track_observations\":"<<c.track_observations<<",\"unique_track_ids\":"<<c.measured_track_ids.size()
       <<",\"unassigned_observations\":"<<c.unassigned<<",\"duplicates\":"<<c.duplicates<<",\"invalid_latency\":"<<c.invalid_latency<<",\"latency_samples\":"<<c.latency.size()<<",\"paced_input_frames\":"<<s.paced_inputs[i].load()<<",\"sampled_out_frames\":"<<s.sampled_out[i].load()<<",\"p95_ms\":"<<percentile(c.latency,.95)<<"}";
  }
  result<<"],\"metadata_dropped\":"<<s.metadata->dropped.load()<<",\"synchronized_file_start\":"<<(s.sync_start?"true":"false")<<",\"decoder_startup_seconds\":"<<s.startup_seconds<<",\"playback_epoch_ns\":"<<(GST_CLOCK_TIME_IS_VALID(s.playback_epoch)?s.playback_epoch:0)<<",\"total_fps\":"<<total<<",\"detector_total_fps\":"<<detector_total<<",\"minimum_fps\":"<<low<<",\"p95_ms\":"<<percentile(latency,.95)<<"}\n";
  result.close();s.intervals.close();
  if(s.profile){
    std::ofstream profile(s.folder+"/stage_profile.json");
    profile<<"{\"note\":\"Wall-clock transit includes queues and scheduling; not isolated GPU kernel time\",\"stages\":{";
    std::vector<std::pair<std::string,std::vector<double>*>> groups={{"mux_age_ms",&s.mux_age},{"inference_transit_ms",&s.infer_transit},{"tracker_transit_ms",&s.tracker_transit},{"processing_transit_ms",&s.processing_transit},{"mux_batch_frames",&s.batch_sizes}};
    for(size_t i=0;i<groups.size();i++){
      auto& v=*groups[i].second;double sum=0;for(auto d:v)sum+=d;if(i)profile<<",";
      profile<<"\""<<groups[i].first<<"\":{\"samples\":"<<v.size()<<",\"mean\":"<<(v.empty()?-1:sum/v.size())<<",\"p50\":"<<percentile(v,.5)<<",\"p95\":"<<percentile(v,.95)<<"}";
    }
    profile<<"}}\n";
  }
  if(s.tracking && s.output){std::ofstream samples(s.folder+"/track_samples_C001.jsonl");for(auto& line:s.track_samples)samples<<line<<"\n";}
  if(s.output && !s.error && s.eos && s.mosaic_frames>0)std::rename((s.folder+"/mosaic.partial.mp4").c_str(),(s.folder+"/mosaic.mp4").c_str());
  std::cout<<"RESULT total="<<total<<" FPS | minimum="<<low<<" FPS | P95="<<percentile(latency,.95)<<"ms | mosaic frames="<<s.mosaic_frames<<std::endl;
  for(auto& c:s.cameras)gst_object_unref(c.timeline);gst_object_unref(s.pipeline);g_main_loop_unref(s.loop);
  return s.error?7:(s.interrupted?130:0);
}
