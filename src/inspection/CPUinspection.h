#ifndef CPU_INSPECTION_H
#define CPU_INSPECTION_H

#include "../util/Packet.h"
#include "inspection.h"
#include <vector>
#include <iostream>
#include <string>
#include <queue>
#include <fstream>

class CPUInspection : public Inspection
{
    private:
        int maxs_;
        const int MAXC_ = 127;
        int* out_;
        int* f_;
        int* g_;
    public:
        CPUInspection();
        CPUInspection(const Inspection &) = delete;
        CPUInspection& operator = (const Inspection &) = delete;
        ~CPUInspection();

        virtual int exec(std::shared_ptr<Packet> pkt);

        int buildMatchingMachine(std::string rules);
        int findNextState(int currentState, char nextInput);
};

#endif
