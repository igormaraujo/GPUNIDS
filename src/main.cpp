/*********************************************************************************

    MIT License

    Copyright (c) 2018 Igor Meireles de Araujo

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

**********************************************************************************/


/**
 * @file main.cpp
 * @brief The main program function
 * 
 * @details This file contains the main() function and read the arguments 
 *
 * @author Igor Meireles de Araújo
 * @date 6 Sep 2018
 * @bug No know bugs.
 * @copyright MIT License 
 */

/* -- Includes --*/
#include <iostream> // for std::cout
#include <cstdlib>  // for exit, EXIT_FAILURE, atoi
#include <cstring>  // for strlen
#include <string>   // for std::string
#include <thread>   // for std::thread::hardware_concurrency

#include "capture/libPCAP.h"
#include "inspection/CPUinspection.h"
#include "inspection/GPUinspection.h"

using namespace std;


/**
 * enum Processing
 * @brief Hardware Processing
 */
enum class Processing {
    CPU, ///< Processing in CPU
    GPU  ///< Processing in GPU
};

/**
 * @struct Parameters
 * @brief Struct to save the arguments passed to program by command line.
 */
struct Parameters
{
    string file = "";    ///< Input PCAP file to offline capture. Default: ""
    int buffer  = 2048;  ///< Buffer size in bytes to transfer between CPU-GPU. Default: 2048
    string filter = "";  ///< Filter capture to libPCAP. See https://www.tcpdump.org/manpages/pcap-filter.7.html
    int nPackets = -1;   ///< Number of packets to capture. Zero and negative values mean capture until a error occur. Default: -1
    int threads = (thread::hardware_concurrency() == 0) ? 2 : thread::hardware_concurrency(); ///< Number to thread to be used by program. Default value is std::thread::hardware_concurrecy, in case returned 0 is used 2 threads. See https://en.cppreference.com/w/cpp/thread/thread/hardware_concurrency.
    string interface = ""; ///< The Network Interface Card (NIC). Default value is chose one NIC available randomly.
    Processing proc = Processing::CPU; ///< Processing Hradware. Default value is Processing::CPU.
    int timeout = 0; ///< timeout in seconds to finalize the capture.
    string rules = "";
    int delay = 50; ///< Buffer limit delay. Default Value is 50 ms.
};

/**
 * @fn void printHelp()
 * @brief Print in terminal the help usage of program.
 */
void printHelp()
{
    cout << "                                                                " << endl;
    cout << "                     GPUNIDS - v. 0.0.2                         " << endl;
    cout << "   Graphic Processing Unit Network Intrusion Detection System   " << endl;
    cout << "                                                                " << endl;
    cout << "usage:                                                          " << endl;
    cout << "                                                                " << endl;
    cout << "    GPUNIDS [Options]                                           " << endl;
    cout << "                                                                " << endl;
    cout << "Options:                                                        " << endl;
    cout << "                                                                " << endl;
    cout << " -o <file> = Offline capture a PCAP file                        " << endl;
    cout << " -r <file> = Rules file                                         " << endl;
    cout << " -b <num>  = Buffer size in bytes to transfer between CPU-GPU   " << endl;
    cout << " -d <num>  = Buffer delay limit.                                " << endl;
    cout << " -f <text> = Filter capture of libPCAP.                         " << endl;
    cout << "      See https://www.tcpdump.org/manpages/pcap-filter.7.html   " << endl;
    cout << " -n <num>  = Number of packets to be captured. -1 will capture  " << endl;
    cout << "             until a error occur.                               " << endl;
    cout << " -t <num>  = Number of thread of Thread Pool                    " << endl;
    cout << " -i <text> = Network Interface Card (NIC)                       " << endl;
    cout << " -p <char> = Processing hardware. P - GPU | S - CPU             " << endl;
    cout << " -l <num>  = timeout in seconds to finalize a real time capture " << endl;
    cout << " -h        = print help                                         " << endl;
    cout << "                                                                " << endl;
    exit(EXIT_FAILURE);
}

/**
 * @fn Parameters* readParameters(const int argc, char* const argv[])
 * @brief read the arguments passed to program by command line.
 *
 * @param argc is a integer. The number of arguments passed by command line.
 * @param argv[] is a array of pointers of character. List of arguments passed by command line.
 *
 * @return Pointer Parameters struct.
 */
Parameters* readParameters(const int argc, char* const argv[])
{
    cout << argv[0];
    char tmp;
    Parameters* params = new Parameters();
    for ( int i = 1; i < argc; i++)
    {
        cout << " " << argv[i];
        if(argv[i][1] != 'h'){
            cout << " " << argv[i + 1];
        }

        if(strlen(argv[i]) != 2 || argv[i][0] != '-')
        {   
            printHelp();    
        }
        try
        {
            switch(argv[i][1])
            {

                case 'b':
                    params->buffer = atoi(argv[++i]);
                    break;

                case 'd':
                    params->delay = atoi(argv[++i]);
                    break;

                case 'f':
                    params->filter = string(argv[++i]);
                    break;

                case 'h':
                    printHelp();
                    break;

                case 'i':
                    params->interface = string(argv[++i]);
                    break;

                case 'l':
                    params->timeout = atoi(argv[++i]);
                    break;

                case 'n':
                    params->nPackets = atoi(argv[++i]);
                    break;

                case 'o':
                    params->file = string(argv[++i]);
                    break;

                case 'p':
                    tmp = argv[++i][0];
                    if(tmp == 'P')
                    {
                        params->proc = Processing::GPU;
                    }
                    else if (tmp == 'S')
                    {
                        params->proc = Processing::CPU;
                    }
                    else 
                    {
                        printHelp();
                    }
                    break;
                
                case 'r':
                    params->rules = string(argv[++i]);
                    break;

                case 't':
                    params->threads = atoi(argv[++i]);
                    break;

                default:
                    printHelp();
            }
        }
        catch (int e)
        {
            cout << "An exception ocurred. Exception Nr. " << e << endl;
            printHelp();
        }
    }
    cout << endl;
    return params;
}

/**
 * @fn int main(int argc, char *argv[])
 * @brief main function
 * @param argc is a integer.The number of arguments passed into program from the command line.
 * @param argv is a array of pointer of characters. A array with the arguments passed into program from the command line.
 * @return Should return 0, mean that program executed successfully
 *
 */
int main(int argc, char *argv[])
{
    /*
     * Read parameters passed to program by command line.
     */
    Parameters *params = readParameters(argc, argv);

    switch(params->proc)
    {
        case Processing::CPU:
            {
                CPUInspection* inspection = new CPUInspection();
                inspection->buildMatchingMachine(params->rules);
            
                LibPCAP pcap(&inspection[0], params->threads - 1, params->nPackets, params->timeout);
            
                if(params->file != "")
                {    
                    pcap.offlineCapture(params->file, params->filter);
                } else {
                	pcap.onlineCapture(params->interface, params->filter);
                }
                /*
                 * Free memory allocated 
                 */
                delete params;
                delete inspection;

            }
            break;

        
        case Processing::GPU:
            {
                GPUInspection* inspection = new GPUInspection();
                inspection->buildMatchingMachine(params->rules, params->buffer, params->delay, 16);
            
                LibPCAP pcap(&inspection[0], params->threads - 1, params->nPackets, params->timeout);
            
                if(params->file != "")
                {    
                    pcap.offlineCapture(params->file, params->filter);
                } else {
                	pcap.onlineCapture(params->interface, params->filter);
                }
                /*
                 * Free memory allocated 
                 */
                delete params;
                delete inspection;

            }
            break;

        default:
            
            cout << "Processing hardware incorrect " << endl;
            printHelp();
    }

    return 0;
}
