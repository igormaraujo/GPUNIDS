#include "Task.h"

Task::Task(Inspection *inspection, std::shared_ptr<Packet> pkt){
    inspection_ = inspection;
    pkt_ = pkt;
}


Task::~Task() {
}

void Task::operator()() {
  inspection_->exec(pkt_);
}

int Task::run() {
  return inspection_->exec(pkt_);
}

int Task::getSize() {
    if(pkt_ != nullptr && pkt_.get() == 0){
    std::cout << "ERRO" << std::endl;
    return 0;
    }
    return pkt_.get()->header_->caplen;
}
