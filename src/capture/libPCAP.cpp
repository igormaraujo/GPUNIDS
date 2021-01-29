#include "libPCAP.h"

using namespace std;

LibPCAP* LibPCAP::instance_ = nullptr;

LibPCAP::LibPCAP(Inspection* inspection, int nThreads, int numPackets, int timeout, int snapLen)
{
    inspection_ = inspection;
    data_ = 0.0;
    numPackets_ = numPackets;
    cntTimeout_ = 0;
    timeout_ = timeout;
    snapLen_ = snapLen;
    instance_ = this;
    pool_ = ThreadPool(nThreads);
    pool_.initialize_threadpool();
    memBuffer_ = 0;
}

LibPCAP::~LibPCAP()
{
    pool_.destroy_threadpool();
}

void LibPCAP::gotPacket(u_char *args, const struct pcap_pkthdr *header, const u_char *packet)
{   
    cntTimeout_ = 0;
    std::shared_ptr<Packet> pkt(new Packet(header, packet));
#ifdef VERBOSE
    gettimeofday(&(pkt.get()->virtualTime), nullptr);
#endif
    Task* t = new Task(inspection_, pkt);
   memBuffer_ += pkt.get()->header_->caplen;
   if(pool_.add_task(t) == 0){
      inspection_->getStats()->cntLoss++;
      delete t;
   }
}

void LibPCAP::static_gotPacket(u_char *args, const struct pcap_pkthdr *header, const u_char *packet)
{
    instance_->gotPacket(args, header, packet);
}

void LibPCAP::timeoutHandle()
{
    cntTimeout_++;
    if(cntTimeout_ == timeout_)
    {
        pcap_breakloop(handle_);
    }
    signal(SIGALRM, this->static_timeoutHandle);
    alarm(1);
}

void LibPCAP::static_timeoutHandle(int sig)
{
    instance_->timeoutHandle();
}

void LibPCAP::onlineCapture(string interface = "", string filter = "")
{
    #ifdef VERBOSE
        cout << "Online Capture" << endl;
    #endif
    char errbuf[PCAP_ERRBUF_SIZE];
    if(interface == ""){
        pcap_if_t* dev = nullptr;
        pcap_findalldevs(&dev, errbuf);
        if(dev == nullptr)
        {
            cerr << "Couldn't find default device: " << errbuf << endl;
            exit (EXIT_FAILURE);
        }
        interface = string(dev->name);
        pcap_freealldevs(dev);
    }

    struct bpf_program fp;     /* compiled filter program (expression) */
    bpf_u_int32 mask;          /* subnet mask */
    bpf_u_int32 net;           /* ip */

     /* get network number and mask associated with capture device */
    if (pcap_lookupnet(interface.c_str(), &net, &mask, errbuf) == -1)
    {
        cerr << "Couldn't get netmask for device " << interface << ": " << errbuf;
        net = 0;
        mask = 0;
    }
    #ifdef VERBOSE
        /* print capture info */
        cout << "Device: " << interface << endl;
        cout << "Number of packets: " << numPackets_ << endl;
        cout << "Filter expression: " << filter << endl;
    #endif

    handle_ = pcap_open_live(interface.c_str(), snapLen_, 1, 1000, errbuf);
    if (handle_ == nullptr)
    {
        cerr << "Couldn't open device " << interface << ": " <<  errbuf << endl;
        exit(EXIT_FAILURE);
    }
    
    /* make sure we're capturing on an Ethernet device [2] */
    if (pcap_datalink(handle_) != DLT_EN10MB)
    {
        cerr << interface << " is not an Ethernet" << endl;
        exit(EXIT_FAILURE);
    }

    /* compile the filter expression */
    if (pcap_compile(handle_, &fp, filter.c_str(), 0, net) == -1)
    {
        cerr <<  "Couldn't parse filter " << filter << ": " << pcap_geterr(handle_);
        exit(EXIT_FAILURE);
    }

    /* apply the compiled filter */
    if (pcap_setfilter(handle_, &fp) == -1)
    {
        cerr << "Couldn't install filter " << filter << ": " <<  pcap_geterr(handle_);
        exit(EXIT_FAILURE);
    }

    signal(SIGALRM, this->static_timeoutHandle);
    alarm(1);

    /* now we can set our callback function */
    pcap_loop(handle_, numPackets_, this->static_gotPacket, nullptr);
    while(pool_.hasTasks() || pool_.isBusy());

    /* cleanup */
    pcap_freecode(&fp);
    pcap_close(handle_);

    #ifdef VERBOSE
        statistics_t *stats = inspection_->getStats();
	cout << fixed;
        cout << "Capture complete." << endl;
        cout << "Packets Captured: " << stats->cntPackets << " pkts" << endl;
        cout << "Data Captured: " << setprecision(2) <<  stats->sumSizePacket << " GB" << endl;
        cout << "Avg Packet Size: " << stats->avgSizePacket << " B" << endl;
        cout << "Total Time Captured: " << stats->cntTime << " s" << endl;
        cout << "Avg Throughput: " << stats->avgRate  / 1024.0 << " Mbps" << endl;
        cout << "Avg Waiting Time: " << stats->sumWaitingTime / stats->cntPackets  << " ms" << endl; 
        cout << "Avg Buffer Time: " << stats->sumBufferTime / stats->cntPackets   << " ms" << endl; 
        cout << "Avg Transfer Time: " << stats->sumTransferTime / stats->cntPackets  << " ms" << endl; 
        cout << "Avg Processing Time: " << stats->sumProcTime / stats->cntPackets  << " ms" << endl; 
        cout << "Avg Buffer Size: " << stats->avgBufferSize  << " pkts" << endl;
        cout << "Packet Loss: " << stats->cntLoss << " pkts" << endl;
	cout << "Packet Drop out: " << stats->cntLoss / (float) (stats->cntLoss + stats->cntPackets) * 100.0 << " %" << endl;
        cout << "############################" << endl;
    #endif

}

void LibPCAP::offlineCapture(std::string file, std::string filter)
{
    #ifdef VERBOSE
        cout << "Offline Capture" << endl;
    #endif
    char errbuf[PCAP_ERRBUF_SIZE];
    #ifdef VERBOSE
        /* print capture info */
        cout << "File: " << file << endl;
        cout << "Number of packets: " << numPackets_ << endl;
        cout << "Filter expression: " << filter << endl;
    #endif

    handle_ = pcap_open_offline(file.c_str(), errbuf);
    if (handle_ == nullptr)
    {
        cerr << "Couldn't open file " << file << ": " <<  errbuf << endl;
        exit(EXIT_FAILURE);
    }
    
    struct bpf_program fp;     /* compiled filter program (expression) */
    /* compile the filter expression */
    if (pcap_compile(handle_, &fp, filter.c_str(), 0, 0) == -1)
    {
        cerr <<  "Couldn't parse filter " << filter << ": " << pcap_geterr(handle_);
        exit(EXIT_FAILURE);
    }

    /* apply the compiled filter */
    if (pcap_setfilter(handle_, &fp) == -1)
    {
        cerr << "Couldn't install filter " << filter << ": " <<  pcap_geterr(handle_);
        exit(EXIT_FAILURE);
    }

    signal(SIGALRM, this->static_timeoutHandle);
    alarm(1);

    /* now we can set our callback function */
    timeval lastPkt, diffTime;
    lastPkt.tv_sec = 0;
    while(cntTimeout_ < timeout_)
    {
        struct pcap_pkthdr pktHeader;
        const u_char* packet = pcap_next(handle_, &pktHeader);
        if(packet == NULL)
        {
            break;
        }
        if(lastPkt.tv_sec == 0)
        {
            lastPkt = pktHeader.ts;    
        }
        timersub(&(pktHeader.ts), &lastPkt, &diffTime);
        struct timespec delay;
        delay.tv_sec = diffTime.tv_sec;
        delay.tv_nsec = diffTime.tv_usec * 1000 + 1;
        nanosleep(&delay, NULL);
        gotPacket(NULL, &pktHeader, packet);
        lastPkt = pktHeader.ts;
        if (numPackets_ > 0 && inspection_->getStats()->cntPackets > numPackets_)
        {
            break;
        }

    }
    while(pool_.hasTasks() || pool_.isBusy());

    /* cleanup */
    pcap_freecode(&fp);
    pcap_close(handle_);

    #ifdef VERBOSE
        statistics_t *stats = inspection_->getStats();
	cout << fixed;
        cout << "Capture complete." << endl;
        cout << "Packets Captured: " << stats->cntPackets << " pkts" << endl;
        cout << "Data Captured: " << setprecision(2) <<  stats->sumSizePacket << " GB" << endl;
        cout << "Avg Packet Size: " << stats->avgSizePacket << " B" << endl;
        cout << "Total Time Captured: " << stats->cntTime << " s" << endl;
        cout << "Avg Throughput: " << stats->avgRate /1024.0  << " Mbps" << endl;
        cout << "Avg Waiting Time: " << stats->sumWaitingTime / stats->cntPackets  << " ms" << endl; 
        cout << "Avg Buffer Time: " << stats->sumBufferTime / stats->cntPackets   << " ms" << endl; 
        cout << "Avg Transfer Time: " << stats->sumTransferTime / stats->cntPackets  << " ms" << endl; 
        cout << "Avg Processing Time: " << stats->sumProcTime / stats->cntPackets  << " ms" << endl; 
        cout << "Avg Buffer Size: " << stats->avgBufferSize  << " pkts" << endl; 
        cout << "Packet Loss: " << stats->cntLoss << " pkts" << endl;
	cout << "Packet Drop out: " << stats->cntLoss / (float)(stats->cntLoss + stats->cntPackets) * 100.0 << " %" << endl;
        cout << "############################" << endl;
    #endif
}
