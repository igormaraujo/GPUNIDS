#include"GPUinspection.h"

__device__ int findNextState(int currentState, char nextInput, int *d_g, int *d_f)
{

    int answer = currentState;
	int ch = nextInput;

	// If goto is not defined, use failure function
	while (d_g[answer * MAXC_ + ch] == -1)
		answer = d_f[answer];

	return d_g[answer * MAXC_ + ch];
}

__global__ void kernel_exec(int *d_g, int *d_f, int *d_out, char *d_packet, int *d_sizePacket, int *d_beginPacket, int *d_nPacket)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < d_nPacket[0]){

		// Initialize current state
		int currentState = 0;
		int sizePacket = d_sizePacket[id];
		int beginPacket = d_beginPacket[id];
		for (int i = 0; i < sizePacket; i++)
		{
			currentState = findNextState(currentState, d_packet[beginPacket + i], d_g, d_f);

			// If match not found, move to next state
			if (d_out[currentState] != 0)
			{
				// rule found
			}
		}
	}
}


GPUInspection::GPUInspection()
{
    maxs_ = 0;
    out_ = nullptr;
    f_ = nullptr;
    g_ = nullptr;
    h_package_ = nullptr;
    h_sizePackage_ = nullptr;
    h_beginPackage_ = nullptr;
    thread_.clear();
    sizeBuffer_ = 0;
    buffer_.clear();

}

GPUInspection::~GPUInspection()
{
    delete[] out_;
    delete[] f_;
    delete[] g_;
    delete[] h_package_;
    delete[] h_sizePackage_;
    delete[] h_beginPackage_;
}

int GPUInspection::exec(std::shared_ptr<Packet> pkt)
{
    #ifdef VERBOSE
        struct timeval start, end, diff;
        gettimeofday(&end, nullptr);
        timersub(&end, &(pkt.get()->virtualTime), &diff);
        this->lock();
        uint tid = (uint) pthread_self();
        int id = getIndex(tid);
        pkt->computeStatistics(this->getStats());
        this->getStats()->sumWaitingTime += diff.tv_sec * 1000.0 + diff.tv_usec / 1000.0;
        this->unlock();

        gettimeofday(&(pkt.get()->virtualTime), nullptr);
    #endif

    pkt.get()->init();


    char* l_packet;
    int* l_sizePacket;
    int* l_beginPacket;
    int l_nPacket;

    bool proc = false;

    this->lock();
    buffer_.push_back(pkt);
    sizeBuffer_ += pkt.get()->size_payload_;

    struct timeval currentTime, bufferTime;
    gettimeofday(&currentTime, nullptr);
    timersub(&currentTime, &(buffer_[0].get()->virtualTime), &bufferTime);
    double bt = bufferTime.tv_sec * 1000.0 + bufferTime.tv_usec / 1000.0;
    
    for(int i = 0; i < 16; i++){
        if(cudaStreamQuery(stream_[i]) == cudaSuccess && processing[i] == true){
            processing[i] = false;
            #ifdef VERBOSE
                if( gettimeofday(&end, nullptr) != 0)
                {
                    std::cerr << "Fail to get current time" << std::endl;
                    exit(-1);
                }
                timersub(&end, &(initProc[i]), &diff);
                this->getStats()->sumProcTime += diff.tv_sec * 1000.0 + diff.tv_usec / 1000.0;
            #endif
        }
    }

    if(sizeBuffer_ >= limitBuffer_ || bt > limitDelay_)
    {
        #ifdef VERBOSE
            this->getStats()->cntBuffer += 1;
            this->getStats()->avgBufferSize = (this->getStats()->cntBuffer-1.0f) /this->getStats()->cntBuffer * this->getStats()->avgBufferSize + ((buffer_.size() -1.0f) / this->getStats()->cntBuffer );

            if( gettimeofday(&end, nullptr) != 0)
            {
                std::cerr << "Fail to get current time" << std::endl;
                exit(-1);
            }
            if( gettimeofday(&start, nullptr) != 0)
            {
                std::cerr << "Fail to get current time" << std::endl;
                exit(-1);
            }
        #endif
        l_packet = new char[limitBuffer_ + 1];
        l_sizePacket = new int[buffer_.size() - 1];
        l_beginPacket = new int[buffer_.size() - 1];
        l_nPacket = buffer_.size() - 1;
        
        

        for(int i = 0, index = 0; i < buffer_.size() - 1; i++)
        {
            l_sizePacket[i] = (buffer_[i].get()->size_payload_ < limitBuffer_) ? buffer_[i].get()->size_payload_ : limitBuffer_;
            l_beginPacket[i] = index;
            strncpy(l_packet + index,reinterpret_cast<const char*>(buffer_[i].get()->payload_),l_sizePacket[i]);
            index += buffer_[i].get()->size_payload_;
            
            timersub(&end, &(buffer_[i].get()->virtualTime), &diff);
            this->getStats()->sumBufferTime += diff.tv_sec * 1000.0 + diff.tv_usec / 1000.0;
        }
        buffer_.erase(buffer_.begin(), buffer_.end() - 1);
        sizeBuffer_ = pkt.get()->size_payload_;
        proc = true;
    }
    this->unlock();

    if(proc)
    {

        bool busy = true;
        while (busy){
            for(int i = 0 ; i  < 16 ; i++){
                if(cudaStreamQuery(stream_[i]) == cudaSuccess){
                    
                    id = i;
                    //cout << "Cuda Stream " << id << " avaliable" << endl;
                    busy = false;
                    break;
                }
            }
        }
            
        int nThreads = 16;//thread_.size();
        int nBufferSize = limitBuffer_;
        int* d_nPacket = d_data_int_ 
            + 1         // nBufferSize
            + 1 * id;   // nPacket[nThreads]
        int* d_g = d_data_int_
            + 1         // nBufferSize
            + nThreads;   // nPacket[nThreads]
        int* d_f = d_g
            + maxs_ * MAXC_;     // d_g[maxs_][MAXC_]
        int* d_out = d_f
            + maxs_;
        int* d_sizePacket = d_out
            + maxs_             // d_out[maxs_]
            + id * nBufferSize;  // d_sizePacket[nTrheads][nBufferSize]
        int* d_beginPacket = d_out
            + maxs_                     // d_out[maxs_]
            + nThreads * nBufferSize    // d_sizePacket[nTrheads][nBufferSize]
            + id * nBufferSize;          // d_beginPacket[nTrheads][nBufferSize]

        char* d_packet = d_data_char_
            + id * nBufferSize;
        
        checkCudaErrors(cudaMemcpyAsync(d_nPacket, &l_nPacket,sizeof(int), cudaMemcpyHostToDevice, stream_[id]));
        checkCudaErrors(cudaMemcpyAsync(d_sizePacket, l_sizePacket, l_nPacket * sizeof(int), cudaMemcpyHostToDevice, stream_[id]));
        checkCudaErrors(cudaMemcpyAsync(d_beginPacket, l_beginPacket, l_nPacket * sizeof(int), cudaMemcpyHostToDevice, stream_[id]));
        checkCudaErrors(cudaMemcpyAsync(d_packet, l_packet, nBufferSize * sizeof(char), cudaMemcpyHostToDevice, stream_[id]));

        //cudaStreamSynchronize(stream_[id]);
        #ifdef VERBOSE        
        if( gettimeofday(&end, nullptr) != 0)
            {
                std::cerr << "Fail to get current time" << std::endl;
                exit(-1);
            }
            timersub(&end, &start, &diff);
            this->getStats()->sumTransferTime += diff.tv_sec * 1000.0 + diff.tv_usec / 1000.0;
            if( gettimeofday(&(initProc[id]), nullptr) != 0)
            {
                std::cerr << "Fail to get current time" << std::endl;
                exit(-1);
            }
        #endif
        processing[id] = true;
        kernel_exec<<<l_nPacket / 128 + 1 ,128, 0, stream_[id]>>>(d_g, d_f, d_out, d_packet, d_sizePacket, d_beginPacket, d_nPacket);
        //cudaStreamSynchronize(stream_[id]);

        delete[] l_packet;
        delete[] l_sizePacket;
        delete[] l_beginPacket;
	
    }
    

    #ifdef DEBUG
        this->lock();
        std::cout << *(pkt.get()) << std::endl;
        this->unlock();
    #endif
     return pkt.get()->header_->caplen;
}

int GPUInspection::buildMatchingMachine(std::string rules, int bufferSize, int limitDelay, int nThreads)
{

    std::vector<std::string> words;
    std::string line;
    ifstream file(rules.c_str(), std::ifstream::in);
    limitBuffer_ = bufferSize;
    limitDelay_ = limitDelay;

    while(getline(file,line))
    {
        words.push_back(line);    
        maxs_ += line.size();
    }

    //TODO alloc memory
    out_ = new int[maxs_];
    f_ = new int[maxs_];
    g_ = new int[maxs_ * MAXC_];
    h_package_ = new char[nThreads * bufferSize];
	h_sizePackage_ = new int[nThreads * bufferSize];
	h_beginPackage_ = new int[nThreads * bufferSize];

    processing = new bool[nThreads];
    for(int i = 0; i < nThreads; i++){
        processing[i] = false;
    }
    initProc = new struct timeval[nThreads];

    
    // Initialize all values in output function as 0.
    memset(out_, 0, sizeof(int) * maxs_);

    // Initialize all values in goto function as -1.
    memset(g_, -1, sizeof(int) * maxs_ * MAXC_);

    // Initially, we just have the 0 state
    int states = 1;

    // Construct values for goto function, i.e., fill g[][]
    // This is same as building a Trie for arr[]
    for (int i = 0; i < words.size(); ++i)
    {
    	const string word(words[i]);
        int currentState = 0;

        // Insert all characters of current word in arr[]
        for (int j = 0; j < word.size(); ++j)
        {
            int ch = word[j];

            // Allocate a new node (create a new state) if a
            // node for ch doesn't exist.
            if (g_[currentState * MAXC_ + ch] == -1)
                g_[currentState * MAXC_ + ch] = states++;

            currentState = g_[currentState * MAXC_ + ch];
        }

        // Add current word in output function
        out_[currentState] |= (1 << i);
    }

    // For all characters which don't have an edge from
    // root (or state 0) in Trie, add a goto edge to state
    // 0 itself
    for (int ch = 0; ch < MAXC_; ++ch)
        if (g_[0 * MAXC_ + ch] == -1)
            g_[0 * MAXC_ + ch] = 0;

    // Now, let's build the failure function

    // Initialize values in fail function
    memset(f_, -1, sizeof(int) * maxs_);

    // Failure function is computed in breadth first order
    // using a queue
    std::queue<int> q;

     // Iterate over every possible input
    for (int ch = 0; ch < MAXC_; ++ch)
    {
        // All nodes of depth 1 have failure function value
        // as 0. For example, in above diagram we move to 0
        // from states 1 and 3.
        if (g_[0 * MAXC_ + ch] != 0)
        {
            f_[g_[0 * MAXC_ + ch]] = 0;
            q.push(g_[0 * MAXC_ + ch]);
        }
    }

    // Now queue has states 1 and 3
    while (q.size())
    {
        // Remove the front state from queue
        int state = q.front();
        q.pop();

        // For the removed state, find failure function for
        // all those characters for which goto function is
        // not defined.
        for (int ch = 0; ch <= MAXC_; ++ch)
        {
            // If goto function is defined for character 'ch'
            // and 'state'
            if (g_[state * MAXC_ + ch] != -1)
            {
                // Find failure state of removed state
                int failure = f_[state];

                // Find the deepest node labeled by proper
                // suffix of string from root to current
                // state.
                while (g_[failure * MAXC_ + ch] == -1)
                      failure = f_[failure];

                failure = g_[failure * MAXC_ + ch];
                f_[g_[state * MAXC_ + ch]] = failure;

                // Merge output values
                out_[g_[state * MAXC_ + ch]] |= out_[failure];

                // Insert the next level node (of Trie) in Queue
                q.push(g_[state * MAXC_ + ch]);
            }
        }
    }
    //TODO GPU    
    

    //Alloc and copy data to GPU's Memory

    // d_data_int_ Allocation Patter
    // nBufferSize|nPackage[nThreads]|d_g[maxs_][MAXC_]|d_f[maxs_]|d_out[maxs_]|d_sizePackage[nThreads][nBufferSize]|d_beginPackage[nThreads][nBufferSize]
	checkCudaErrors(cudaMalloc((void **)&d_data_int_, (
                    (maxs_ * MAXC_ + maxs_ + maxs_) // d_g[maxs_][MAXC_] + d_f[maxs_] + d_out[maxs_] 
                    + 1 + nThreads                  // nBufferSize + nThreads(nPackets each buffer)
                    + 2 * nThreads * bufferSize     // beginPackage[bufferSize] + sizePackage[bufferSize] (for each buffer)
                    ) * sizeof(int)));
    
    // d_data_char_ Allocation Patter
    // d_package[nThreads][nBufferSize]
	checkCudaErrors(cudaMalloc((void **)&d_data_char_, (
                    nThreads * bufferSize           // Package data (for each buffer)
                    ) * sizeof(char)));
    

    // init pointers to each data in d_data_int_ and d_data_char_
	int *d_bufferSize = d_data_int_;
	int *d_g = d_bufferSize + 1 + nThreads;
	int *d_f = d_g + maxs_ * MAXC_;
	int *d_out = d_f + maxs_;
    
    // Copy to GPU the Machine State(g_, f_, out_) and bufferSize
	checkCudaErrors(cudaMemcpyAsync(d_bufferSize,&bufferSize, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(d_g, g_, maxs_ * MAXC_ * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(d_f, f_, maxs_ * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(d_out, out_, maxs_ * sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaDeviceSynchronize());

    stream_ = std::vector<cudaStream_t>(nThreads);

    for(int i = 0; i < nThreads; i++)
    {
	    cudaStreamCreate(&stream_[i]);
    }

    return states;
}

int GPUInspection::getIndex(uint pid)
{
    for(int i = 0; i < thread_.size(); i++)
    {
        if(thread_[i] == pid)
        {
            return i;
        }
    }
    thread_.push_back(pid);
    return thread_.size() - 1;
}

