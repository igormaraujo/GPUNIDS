#ifndef GPU_INSPECTION_H
#define GPU_INSPECTION_H

#include "../util/Packet.h"
#include "inspection.h"
#include <vector>
#include <iostream>
#include <string>
#include <queue>
#include <fstream>
#include <cuda_runtime.h>
#include "../util/helper_cuda.h"

#define MAXC_ 127

class GPUInspection : public Inspection
{
    private:
        int maxs_;
        int* out_;
        int* f_;
        int* g_;

        char* h_package_;
        int* h_sizePackage_;
        int* h_beginPackage_;

        int* d_data_int_;
        char* d_data_char_;

        int limitBuffer_;
        int sizeBuffer_;

        int limitDelay_;

        std::vector<cudaStream_t> stream_;
        std::vector<uint> thread_;
        std::vector<std::shared_ptr<Packet> >buffer_;
        
        bool* processing;
        struct timeval* initProc;

    public:
        GPUInspection();
        GPUInspection(const Inspection &) = delete;
        GPUInspection& operator = (const Inspection &) = delete;
        ~GPUInspection();

        virtual int exec(std::shared_ptr<Packet> pkt);

        int buildMatchingMachine(std::string rules, int bufferSize, int limitDelay, int nThreads);
        int getIndex(uint pid);
};

#endif
