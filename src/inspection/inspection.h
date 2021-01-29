#ifndef INSPECTION_H
#define INSPECTION_H

#include <memory>
#include <iostream>
#include "../util/Packet.h"
#include "../ThreadPool/Mutex.h"

class Inspection
{
    private:
        Mutex m_inspection_mutex_;
        statistics_t *stats_;
    public:
        Inspection();
        Inspection(const Inspection &) = delete;
        Inspection& operator = (const Inspection &) = delete;
        ~Inspection();

        virtual int exec(std::shared_ptr<Packet> pkt) = 0;
        void lock();
        void unlock();
        statistics_t* getStats();
};

#endif
