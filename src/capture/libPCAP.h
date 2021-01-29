#ifndef LibPCAP_H
#define LibPCAP_H

#include <string>
#include <pcap.h>
#include <iostream>
#include <iomanip>
#include <csignal>
#include <unistd.h>
#include <cstdlib>
#include <functional>
#include "../ThreadPool/ThreadPool.h"
#include "../util/Packet.h"
#include "../inspection/inspection.h"

class LibPCAP
{
    private:
        int cntTimeout_;
        int timeout_;
        int numPackets_;
        int snapLen_;
        double data_;
        pcap_t* handle_;
        std::string interface_;
        std::string filter_;
        ThreadPool pool_;
        Inspection* inspection_;
        int memBuffer_;

        void static static_gotPacket(u_char*, const struct pcap_pkthdr*, const u_char*);
        void gotPacket(u_char*, const struct pcap_pkthdr*, const u_char*);
        void static static_timeoutHandle(int sig);
        void timeoutHandle();
        static LibPCAP *instance_;
    public:
        LibPCAP() = delete;
        LibPCAP(const LibPCAP &) = delete;
        LibPCAP& operator = (const LibPCAP &) = delete;
        ~LibPCAP();

        LibPCAP(Inspection* inspection, int nThreads = 1, int numPackets = -1, int timeout = 10, int snapLen = 65535);

        void onlineCapture(std::string interface , std::string filter);
        void offlineCapture(std::string file, std::string filter);        
        
};
#endif
