#include "Packet.h"

Packet::Packet(const struct pcap_pkthdr* pkthdr, const u_char* packet)
{
    header_ = new struct pcap_pkthdr();
    header_->ts = pkthdr->ts;
    header_->caplen = pkthdr->caplen;
    header_->len = pkthdr->len;
    
    if(header_->caplen > 0)
    {
        packet_ = new u_char[header_->caplen + 1];
        memcpy(packet_, packet, header_->caplen);
        packet_[header_->caplen] = '\0';
    } else 
    {
        packet_ = nullptr;
    }
    
    size_payload_ = 0;
    ethernet_ = nullptr;
    ip_ = nullptr;
    tcp_ = nullptr;

}

Packet::~Packet()
{
    delete header_;
    if(packet_ != nullptr)
        delete[] packet_;
}

void Packet::init()
{
    if(packet_ == nullptr)
        return;
    /* define ethernet header */
    ethernet_ = (struct sniff_ethernet *)(packet_);
    /* define/compute ip header offset */
    ip_ = (struct sniff_ip *)(packet_ + SIZE_ETHERNET);
    int size_ip = IP_HL(ip_) * 4;
    if (size_ip < 20)
    {
        std::cout << "   * Invalid IP header length: " << size_ip << " bytes" << std::endl;
        return;
    }

    if(ip_->ip_p != IPPROTO_TCP)
    {
        std::cout << "Protocol isn't TCP" << std::endl;
        return;
    }

    /* define/compute tcp header offset */
    tcp_ = (struct sniff_tcp *)(packet_ + SIZE_ETHERNET + size_ip);
    int size_tcp = TH_OFF(tcp_) * 4;
    if (size_tcp < 20)
    {
        std::cout << "   * Invalid TCP header length: "  << size_tcp << " bytes" << std::endl;
        return;
    }
	
    // define/compute tcp payload (segment) offset 
    payload_ = (u_char *)(packet_ + SIZE_ETHERNET + size_ip + size_tcp);

    // compute tcp payload (segment) size 
    size_payload_ = ntohs(ip_->ip_len) - (size_ip + size_tcp);
    
    for (int i = 0; i < size_payload_; i++)
    {
        if (!isprint(payload_[i]))
            payload_[i] = '.';
    }

}

void Packet::computeStatistics(statistics_t *stats)
{
    stats->cntPackets++;
    struct sniff_ip* ip_ = (struct sniff_ip *)(this->packet_ + SIZE_ETHERNET);
    int size_ip = IP_HL(ip_) * 4;
    struct sniff_tcp* tcp_ = (struct sniff_tcp *)(this->packet_ + SIZE_ETHERNET + size_ip);
    int size_tcp = TH_OFF(tcp_) * 4;
    double size_payload = ntohs(ip_->ip_len) - (size_ip + size_tcp);
    stats->sumSizePacket += (size_payload /1024.0 /1024.0 /1024.0);
    stats->avgSizePacket = ( (stats->cntPackets - 1.0) /  stats->cntPackets * stats->avgSizePacket + size_payload / stats->cntPackets );
    if(this->header_->ts.tv_sec > stats->lastTime)
    {
        if(stats->lastTime != 0)
        {
           stats->avgRate = (stats->cntTime * stats->avgRate + stats->sumSizeRate) / ++stats->cntTime;
	   std::cout << " [ " << stats->cntTime - 1 << " - " << stats->cntTime << " ] " << stats->sumSizeRate / 1024.0 << " Mbps  \t"<< stats->sumSizeRate / 8 / 1024.0 << " MB" << std::endl;
        }
        stats->lastTime = this->header_->ts.tv_sec;
        stats->sumSizeRate = (size_payload * 8.0 / 1024.0);
    }
    else 
    {
       stats->sumSizeRate += (size_payload * 8.0 / 1024.0);
    }
}

std::ostream& operator<<(std::ostream& os, const Packet& pkt)
{
    os << "###################################" << std::endl;
    os << "Packet: " << std::endl;
    os << "Timestamp: " << pkt.header_->ts.tv_sec << "." << pkt.header_->ts.tv_usec << std::endl;
    if(pkt.packet_ != nullptr)
    {
        os << "From: " << inet_ntoa(pkt.ip_->ip_src) << ":" << ntohs(pkt.tcp_->th_sport) << std::endl;
        os << "To: " << inet_ntoa(pkt.ip_->ip_dst) << ":" << ntohs(pkt.tcp_->th_dport) << std::endl;
        os << "Protocol: ";
        switch (pkt.ip_->ip_p)
        {
           case IPPROTO_TCP:
               os << "TCP";
               break;
           case IPPROTO_UDP:
               os << "UDP";
               break;
           case IPPROTO_ICMP:
               os << "ICMP";
               break;
           case IPPROTO_IP:
               os << "IP";
               break;
           default:
               os << "UNKNOWN";
               break;
         }
        os <<  std::endl;
        os << "Payload (" << pkt.size_payload_ << " bytes): ";
        if(pkt.size_payload_ > 0) 
            os << pkt.payload_;
        os << std::endl;
    }
    os << "###################################";
    return os;
}
