#include "ThreadPool.h"

ThreadPool::ThreadPool() : m_pool_size(DEFAULT_POOL_SIZE)
{
	m_pool_state = STOPPED;
    busy_ = 0;
  //cout << "Constructed ThreadPool of size " << m_pool_size << endl;
}

ThreadPool::ThreadPool(int pool_size) : m_pool_size(pool_size)
{
	m_pool_state = STOPPED;
    busy_ = 0;
    memBuffer_ = 0;
  //cout << "Constructed ThreadPool of size " << m_pool_size << endl;
}

ThreadPool::~ThreadPool()
{
  // Release resources
  if (m_pool_state != STOPPED) {
    destroy_threadpool();
  }
}

// We can't pass a member function to pthread_create.
// So created the wrapper function that calls the member function
// we want to run in the thread.
extern "C"
void* start_thread(void* arg)
{
  ThreadPool* tp = (ThreadPool*) arg;
  tp->execute_thread();
  return NULL;
}

int ThreadPool::initialize_threadpool()
{
  // TODO: COnsider lazy loading threads instead of creating all at once
  m_pool_state = STARTED;
  int ret = -1;
  for (int i = 0; i < m_pool_size; i++) {
    pthread_t tid;
    ret = pthread_create(&tid, NULL, start_thread, (void*) this);
    if (ret != 0) {
      cerr << "pthread_create() failed: " << ret << endl;
      return -1;
    }
    m_threads.push_back(tid);
  }
  //cout << m_pool_size << " threads created by the thread pool" << endl;

  return 0;
}

void ThreadPool::look(){
	m_task_mutex.lock();
}

void ThreadPool::unlook(){
	m_task_mutex.unlock();
}

int ThreadPool::destroy_threadpool()
{
  // Note: this is not for synchronization, its for thread communication!
  // destroy_threadpool() will only be called from the main thread, yet
  // the modified m_pool_state may not show up to other threads until its
  // modified in a lock!
  m_task_mutex.lock();
  m_pool_state = STOPPED;
  m_task_mutex.unlock();
  //cout << "Broadcasting STOP signal to all threads..." << endl;
  m_task_cond_var.broadcast(); // notify all threads we are shttung down

  int ret = -1;
  for (int i = 0; i < m_pool_size; i++) {
    void* result;
    ret = pthread_join(m_threads[i], &result);
    //cout << "pthread_join() returned " << ret << ": " << strerror(errno) << endl;
    m_task_cond_var.broadcast(); // try waking up a bunch of threads that are still waiting
  }
  //cout << m_pool_size << " threads exited from the thread pool" << endl;
  return 0;
}

void* ThreadPool::execute_thread()
{
    Task* task = NULL;
  //cout << "Starting thread " << getIndex(pthread_self()) << endl;
  while(true) {
    // Try to pick a task
	  //cout << "Locking: " << getIndex(pthread_self()) << endl;
    m_task_mutex.lock();

    // We need to put pthread_cond_wait in a loop for two reasons:
    // 1. There can be spurious wakeups (due to signal/ENITR)
    // 2. When mutex is released for waiting, another thread can be waken up
    //    from a signal/broadcast and that thread can mess up the condition.
    //    So when the current thread wakes up the condition may no longer be
    //    actually true!
    while ((m_pool_state != STOPPED) && (m_tasks.empty())) {
      // Wait until there is a task in the queue
      // Unlock mutex while wait, then lock it back when signaled
    	//cout << "Unlocking and waiting: " << getIndex(pthread_self()) << endl;
      m_task_cond_var.wait(m_task_mutex.get_mutex_ptr());
      //cout << "Signaled and locking: " << getIndex(pthread_self()) << endl;
    }

    // If the thread was woken up to notify process shutdown, return from here
    if (m_pool_state == STOPPED) {
      //cout << "Unlocking and exiting: " << getIndex(pthread_self()) << endl;
      m_task_mutex.unlock();
      pthread_exit(NULL);
    }

    task = m_tasks.front();
    m_tasks.pop_front();
    //cout << "Unlocking: " << getIndex(pthread_self()) << endl;
    m_task_mutex.unlock();
    
    busy_ = 1 << getIndex(pthread_self()) | busy_;
    //cout << "Executing thread " << getIndex(pthread_self()) << endl;
    // execute the task
    int size = task->getSize();
    (*task)(); // could also do task->run(arg);
    m_task_mutex.lock();
    memBuffer_ -= size;
    m_task_mutex.unlock();
    //cout << "Done executing thread " << getIndex(pthread_self()) << endl;
    busy_ = ~(1 << getIndex(pthread_self())) & busy_;
    delete task;
  }
  return NULL;
}

int ThreadPool::add_task(Task* task)
{
  
  if(memBuffer_ < 536870912) { // 128 MB - 134217728 | 512 MB - 536870912  | 1 GB - 1073741824
    m_task_mutex.lock();
    memBuffer_ += task->getSize();
    // TODO: put a limit on how many tasks can be added at most
    m_tasks.push_back(task);

    m_task_cond_var.signal(); // wake up one thread that is waiting for a task to be available

    m_task_mutex.unlock();
   
    return 1;
  }
  return 0;
}

int ThreadPool::remove_task(){

  m_task_mutex.lock();

  Task* task = m_tasks.front();
  m_tasks.pop_front();

  m_task_mutex.unlock();
  int a = task->getSize();
  //delete task;
  return a;
}

int ThreadPool::getIndex(pthread_t tid){
	for(int i = 0; i < m_threads.size(); i++){
		if(pthread_equal(m_threads[i],tid)) {
			return i;
		}
	}
	return -1;
}

bool ThreadPool::hasTasks(){
	return (m_tasks.size() > 0);
}

bool ThreadPool::isBusy()
{
    return (busy_ > 0);
}
