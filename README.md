# A GPU-assisted NFV framework for intrusion detection system.

Authors: Igor Araujo¹, Carlos Natalino², Diego Cardoso¹

¹ Institute of Technology, Federal University of Pará, Belém, Pará, Brazil.

² Department of Electrical Engineering, Chalmers University of Technology, Gothenburg, Sweden.

Paper available at ScienceDirect : https://doi.org/10.1016/j.comcom.2021.01.024.

**Abstract:** The network function virtualization (NFV) paradigm advocates the replace-ment of specific-purpose hardware supporting packet processing by general-purpose ones, reducing costs and bringing more flexibility and agility to thenetwork operation.  However, this shift can degrade the network performance due to the non-optimal packet processing capabilities of the general-purpose hardware.  Meanwhile, graphics processing units (GPUs) have been deployed in many data centers (DCs) due to their broad use in e.g.  machine learn-ing (ML).  These GPUs can be leveraged to accelerate the packet processingcapability of virtual network functions (vNFs), but the delay introduced can be an issue for some applications.  Our work proposes a framework for packet processing acceleration using GPUs to support vNF execution.  We validatethe proposed framework with a case study,  analyzing the benefits of using GPU to support the execution of an intrusion detection system (IDS) as a vNF and evaluating the traffic intensities where using our framework brings the most benefits.  Results show that the throughput of the system increases from 50 Mbps to 1 Gbps by employing our framework,  while reducing the central process unit (CPU) resource usage by almost 40%.  The results indicate that GPUs are a good candidate for accelerating packet processing in vNFs.

## Content of this document

1. <a href="#installation">Installation</a>
2. <a href="#usage">Usage</a>
3. <a href="#citing-the-work">Citing the work</a>


## Installation

#### Requiriments

- [CUDA Toolkit Develop](https://developer.nvidia.com/Cuda-downloads)
- [libpcap](https://www.tcpdump.org/)
- [Cmake](https://cmake.org/)

You can install the GPUNIDS with:

```bash
git clone https://github.com/igormaraujo/GPUNIDS.git
cd GPUNIDS
mkdir build
cd build
cmake -D VERBOSE=1 ..
make
``` 
The executable named "GPUNIDS" will be generated at build folder

## Usage

>    GPUNIDS [Options]                                           
>                                                                
>Options:                                                        
>                                                                
> -o <file> = Offline capture a PCAP file                        
> -r <file> = Rules file                                         
> -b <num>  = Buffer size (B) in bytes to transfer between CPU-GPU   
> -d <num>  = Buffer delay (ms) limit.                                
> -f <text> = Filter capture of libPCAP. See https://www.tcpdump.org/manpages/pcap-filter.7.html   
> -n <num>  = Number of packets to be captured. -1 will capture at infinite loop. 
> -t <num>  = Number of thread of Thread Pool                    
> -i <text> = Network Interface Card (NIC)                       
> -p <char> = Processing hardware. P - GPU | S - CPU             
> -l <num>  = timeout in seconds to finalize a real time capture after the last packet received
> -h        = print help                                         

Example:

```bash
./GPUNIDS -i eth0 -r rules.csv -d 500 -b 4194304 -f "tcp port 5201 and ip src 10.0.0.1" -p P -l 10 -t 4
```

## Citing the work


> I. Araujo, C. Natalino and D. Cardoso, A GPU-assisted NFV framework for intrusion detection system, Computer Communications (2021), https://doi.org/10.1016/j.comcom.2021.01.024 


Bibtex entry:

~~~~
@article{ARAUJO2021,
title = "A GPU-assisted NFV framework for intrusion detection system",
journal = "Computer Communications",
year = "2021",
issn = "0140-3664",
doi = "https://doi.org/10.1016/j.comcom.2021.01.024",
url = "http://www.sciencedirect.com/science/article/pii/S0140366421000451",
author = "Igor Araujo and Carlos Natalino and Diego Cardoso"
}
~~~~
