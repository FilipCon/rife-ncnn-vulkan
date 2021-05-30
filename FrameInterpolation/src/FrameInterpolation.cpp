#include "FrameInterpolation.h"
#include <iostream>
#include <stdexcept>

// ncnn
#include "cpu.h"
#include "gpu.h"

void TaskQueue::put(const Task& v) {
    lock.lock();
    while (tasks.size() >= 8) condition.wait(lock); // TODO control queue size
    tasks.push(v);
    lock.unlock();
    condition.signal();
}

void TaskQueue::get(Task& v) {
    lock.lock();
    while (tasks.empty()) condition.wait(lock);
    v = tasks.front();
    tasks.pop();
    lock.unlock();
    condition.signal();
}

Interpolation::Interpolation(std::string model, int ttaMode, int uhdMode) {
    timesteps.push_back(0.5); // TODO set timesteps

    bool isRifeV2 = false;
    if (model.find(PATHSTR("rife-v2")) != std::string::npos) isRifeV2 = true;

    createGPUInstance(gpuid, jobsProc, gpuCount, totalJobsProc);

    // create rife
    rife.resize(gpuCount);
    for (int i = 0; i < gpuCount; ++i) {
        int numThreads = gpuid[i] == -1 ? jobsProc[i] : 1;
        rife[i] = new RIFE(gpuid[i], ttaMode, uhdMode, numThreads, isRifeV2);
        rife[i]->load(model);
    }

    // start processing threads
    startParallerProc();
}

void Interpolation::createGPUInstance(std::vector<int>& gpuid,
                                      std::vector<int>& jobs_proc,
                                      int& use_gpu_count,
                                      int& total_jobs_proc) {
    ncnn::create_gpu_instance();
    gpuid.push_back(ncnn::get_default_gpu_index());
    use_gpu_count = (int) gpuid.size(); // NOTE
    jobs_proc.resize(use_gpu_count, 2);
    int cpu_count = std::max(1, ncnn::get_cpu_count());
    int gpu_count = ncnn::get_gpu_count();
    for (int i = 0; i < use_gpu_count; ++i) {
        if (gpuid[i] < -1 || gpuid[i] >= gpu_count) {
            ncnn::destroy_gpu_instance();
            throw std::runtime_error("invalid gpu device\n");
        }
    }

    for (int i = 0; i < use_gpu_count; ++i) {
        if (gpuid[i] == -1) {
            jobs_proc[i] = std::min(jobs_proc[i], cpu_count);
            total_jobs_proc += 1;
        } else {
            int gpu_queue_count =
                    ncnn::get_gpu_info(gpuid[i]).compute_queue_count();
            jobs_proc[i] = std::min(jobs_proc[i], gpu_queue_count);
            total_jobs_proc += jobs_proc[i];
        }
    }
}

void Interpolation::startParallerProc() {
    ptp.resize(gpuCount);
    for (int i = 0; i < gpuCount; ++i) {
        ptp[i].rife = rife[i];
        ptp[i].s = this;
    }

    procThreads.resize(totalJobsProc);
    int id = 0;
    for (int i = 0; i < gpuCount; ++i) {
        if (gpuid[i] == -1) {
            procThreads[id++] = new ncnn::Thread(proc, (void*) &ptp[i]);
        } else {
            for (int j = 0; j < jobsProc[i]; ++j) {
                procThreads[id++] = new ncnn::Thread(proc, (void*) &ptp[i]);
            }
        }
    }
}

Interpolation::~Interpolation() {
    // send termination tasks
    Task end;
    end.id = -233;
    for (int i = 0; i < totalJobsProc; ++i) { toProcQueue.put(end); }

    // delete threads
    for (int i = 0; i < totalJobsProc; ++i) {
        procThreads[i]->join();
        delete procThreads[i];
    }

    // delete rifes
    for (int i = 0; i < gpuCount; ++i) { delete rife[i]; }
    rife.clear();

    ncnn::destroy_gpu_instance();
}

void Interpolation::load(void* image0, void* image1, int w, int h) {
    Task v;
    v.id = 0;
    v.timestep = timesteps[0];
    v.in0image = ncnn::Mat(w, h, image0, (size_t) 3, 3);
    v.in1image = ncnn::Mat(w, h, image1, (size_t) 3, 3);
    v.outimage = ncnn::Mat(w, h, image0, (size_t) 3, 3); // NOTE create copy
    toProcQueue.put(v);
}

void Interpolation::save(void* image) {
    Task v;
    toSaveQueue.get(v);
    image = v.outimage.data;
}

void Interpolation::interpolate(void* image0, void* image1, void* output, int w,
                                int h) {
    load(image0, image1, w, h);
    save(output);
}

void* proc(void* args) {
    const ProcThreadParams* ptp = (const ProcThreadParams*) args;
    const RIFE* rife = ptp->rife;
    Interpolation* s = ptp->s;

    while (true) {
        Task v;
        s->toProcQueue.get(v);
        if (v.id == -233) break;

        rife->process(v.in0image, v.in1image, v.timestep, v.outimage);
        s->toSaveQueue.put(v);
    }
    return nullptr;
}
