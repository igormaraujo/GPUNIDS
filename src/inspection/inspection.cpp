#include "inspection.h"

Inspection::Inspection()
{
    stats_ = new statistics_t();
}

Inspection::~Inspection()
{
    delete stats_;
}

void Inspection::lock()
{
    m_inspection_mutex_.lock();    
}

void Inspection::unlock()
{

    m_inspection_mutex_.unlock();

}

statistics_t* Inspection::getStats()
{
    return stats_;    
}
