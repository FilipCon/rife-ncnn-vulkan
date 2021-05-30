#pragma once
#include <queue>
#include <vector>

/* ncnn */
#include "filesystem_utils.h"
#include "rife.h"

class Task {
 public:
    int id;
    float timestep;

    ncnn::Mat in0image;
    ncnn::Mat in1image;
    ncnn::Mat outimage;
};

class TaskQueue {
 public:
    TaskQueue() {}
    void put(const Task& v);
    void get(Task& v);

 private:
    ncnn::Mutex lock;
    ncnn::ConditionVariable condition;
    std::queue<Task> tasks;
};

class Interpolation;

struct ProcThreadParams {
    const RIFE* rife;
    Interpolation* s;
};

class Interpolation {
 public:
    Interpolation(std::string model, int ttaMode, int uhdMode);
    ~Interpolation();

    void interpolate(void* image0, void* image1, void* output, int w, int h);
    void load(void* image0, void* image1, int w, int h);
    void save(void* image);
    void createGPUInstance(std::vector<int>& gpuid, std::vector<int>& jobs_proc,
                           int& use_gpu_count, int& total_jobs_proc);
    void startParallerProc();

    TaskQueue toProcQueue;
    TaskQueue toSaveQueue;

    std::vector<int> gpuid;
    std::vector<int> jobsProc;
    int gpuCount = 0;
    int totalJobsProc = 0;

    std::vector<RIFE*> rife;
    std::vector<ProcThreadParams> ptp;
    std::vector<ncnn::Thread*> procThreads;

    std::vector<float> timesteps;
};

void* proc(void* args);
