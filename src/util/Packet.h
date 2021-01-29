#ifndef PACKET_H
#define PACKET_H

#include <pcap.h>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <arpa/inet.h>

/* ethernet headers are always exactly 14 bytes */
#define SIZE_ETHERNET 14

/* Ethernet addresses are 6 bytes */
#define ETHER_ADDR_LEN 6

/* Ethernet header */
struct sniff_ethernet
{
    u_char ether_dhost[ETHER_ADDR_LEN]; /* destination host address */
    u_char ether_shost[ETHER_ADDR_LEN]; /* source host address */
    u_short ether_type;                 /* IP? ARP? RARP? etc */
};

/* IP header */
struct sniff_ip
{
    u_char ip_vhl;                 /* version << 4 | header length >> 2 */
    u_char ip_tos;                 /* type of service */
    u_short ip_len;                /* total length */
    u_short ip_id;                 /* identification */
    u_short ip_off;                /* fragment offset field */
#define IP_RF 0x8000               /* reserved fragment flag */
#define IP_DF 0x4000               /* dont fragment flag */
#define IP_MF 0x2000               /* more fragments flag */
#define IP_OFFMASK 0x1fff          /* mask for fragmenting bits */
    u_char ip_ttl;                 /* time to live */
    u_char ip_p;                   /* protocol */
    u_short ip_sum;                /* checksum */
    struct in_addr ip_src, ip_dst; /* source and dest address */
};

#define IP_HL(ip) (((ip)->ip_vhl) & 0x0f)
#define IP_V(ip) (((ip)->ip_vhl) >> 4)

/* TCP header */
typedef u_int tcp_seq;

struct sniff_tcp
{
    u_short th_sport; /* source port */
    u_short th_dport; /* destination port */
    tcp_seq th_seq;   /* sequence number */
    tcp_seq th_ack;   /* acknowledgement number */
    u_char th_offx2;  /* data offset, rsvd */
#define TH_OFF(th) (((th)->th_offx2 & 0xf0) >> 4)
    u_char th_flags;
#define TH_FIN 0x01
#define TH_SYN 0x02
#define TH_RST 0x04
#define TH_PUSH 0x08
#define TH_ACK 0x10
#define TH_URG 0x20
#define TH_ECE 0x40
#define TH_CWR 0x80
#define TH_FLAGS (TH_FIN | TH_SYN | TH_RST | TH_ACK | TH_URG | TH_ECE | TH_CWR)
    u_short th_win; /* window */
    u_short th_sum; /* checksum */
    u_short th_urp; /* urgent pointer */
};

struct statistics_t
{
    public:
        int cntPackets;
        double sumSizePacket;
	    double avgSizePacket;
        double avgRate;
        int cntTime;
        int lastTime;
        double sumSizeRate;
        int cntRate;
        double sumWaitingTime;
        double sumBufferTime;
        double sumTransferTime;
        double sumProcTime;
        double avgBufferSize;
        int cntBuffer;
        int cntLoss;

        statistics_t()
        {
            cntPackets = 0;
            sumSizePacket = 0.0;
	        avgSizePacket = 0.0;
            avgRate = 0.0;
            lastTime = 0;
            sumSizeRate = 0;
            cntTime = 0;
            cntRate = 0;
            sumWaitingTime = 0.0;
            sumBufferTime = 0.0;
            sumTransferTime = 0.0;
            sumProcTime = 0.0;
            avgBufferSize = 0.0;
            cntBuffer = 0;
            cntLoss = 0;
        }

};

class Packet
{
    private:
        u_char *packet_;
        /* declare pointers to packet headers */
        struct sniff_ethernet *ethernet_; /* The ethernet header [1] */
        struct sniff_ip *ip_;             /* The IP header */
        struct sniff_tcp *tcp_;           /* The TCP header */


    public:
        struct pcap_pkthdr *header_;
        u_char *payload_;                   /* Packet payload */
        int size_payload_;
        struct timeval virtualTime;
        Packet() = delete;
        Packet(const Packet&) = delete;
//        Packet& operator=(const Packet&) = delete;
        ~Packet();
        
        friend std::ostream& operator<<(std::ostream& os, const Packet& pkt);
        Packet(const struct pcap_pkthdr *header, const u_char *packet);
        void init();
        void computeStatistics(statistics_t* stats);

};

#endif
