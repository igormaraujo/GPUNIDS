#include "CPUinspection.h"

CPUInspection::CPUInspection()
{
    out_ = nullptr;
    f_ = nullptr;
    g_ = nullptr;
}

CPUInspection::~CPUInspection()
{
    if(out_ != nullptr)
        delete[] out_;

    if(f_ != nullptr)
        delete[] f_;

    if(g_ != nullptr)
        delete[] g_;
}

int CPUInspection::exec(std::shared_ptr<Packet> pkt)
{
#ifdef VERBOSE
    struct timeval start, end, diff;
    gettimeofday(&end, nullptr);
    timersub(&end, &(pkt.get()->virtualTime), &diff);
    this->lock();
    pkt->computeStatistics(this->getStats());
    this->getStats()->sumWaitingTime += diff.tv_sec * 1000.0 + diff.tv_usec / 1000.0;
    this->unlock();

    if( gettimeofday(&start, nullptr) != 0)
    {
        std::cerr << "Fail to get current time" << std::endl;
        exit(-1);
    }
#endif

    pkt.get()->init();

#ifdef DEBUG
    this->lock();
    std::cout << *(pkt.get()) << std::endl;
    this->unlock();
#endif

    int currentState = 0;
    bool found = false;
    for(int i = 0; i < pkt.get()->size_payload_; i++)
    {
        currentState = this->findNextState(currentState, pkt.get()->payload_[i]);

        if(out_[currentState] == 0)
        {
            continue;
        }
    }
#ifdef VERBOSE
    
    if( gettimeofday(&end, nullptr) != 0)
    {
        std::cerr << "Fail to get current time" << std::endl;
        exit(-1);
    }
    timersub(&end, &start, &diff);
    this->lock();
    this->getStats()->sumProcTime += diff.tv_sec * 1000.0 + diff.tv_usec / 1000.0;
    this->unlock();
#endif
    return pkt.get()->header_->caplen;
}

int CPUInspection::buildMatchingMachine(std::string rules)
{
    std::vector<std::string> words;
    std:string line;
    ifstream file(rules.c_str(), std::ifstream::in);
    maxs_ = 0;
    while(getline(file,line))
    {
        words.push_back(line);    
        maxs_ += line.size();
    }

    //TODO alloc memory
    out_ = new int[maxs_];
    f_ = new int[maxs_];
    g_ = new int[maxs_ * MAXC_];
    
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

    return states;
}

int CPUInspection::findNextState(int currentState, char nextInput)
{
    int answer = currentState;
    int ch = nextInput;

    // If goto is not defined, use failure function
    while (g_[answer * MAXC_ + ch] == -1)
        answer = f_[answer];

    return g_[answer * MAXC_ + ch];
}
