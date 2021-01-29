#pragma once

#include <pthread.h>
#include <unistd.h>
#include <deque>
#include <iostream>
#include <vector>
#include <errno.h>
#include <string.h>

#include "Global.h"
#include "../inspection/inspection.h"
#include "../util/Packet.h"

using namespace std;

class Task
{
public:
  Task(Inspection *insp, std::shared_ptr<Packet> pkt); // pass a free function pointer
  ~Task();
  void operator()();
  int run();
  int getSize();
private:
    Inspection *inspection_;
    std::shared_ptr<Packet> pkt_;
};
