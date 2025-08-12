import threading
from time import sleep, time

import psutil


_cpu_info_event = threading.Event()
_cpu_info_event.clear()


def get_cpu_mem_info(end_event):
    p = psutil.Process()
    data = ["cpu_id\t| cputime\t| threads\t| cpu%\t| RES MB| Virtual MB"]
    while not end_event.is_set():
        sleep(2)
        __mem = p.memory_info()
        m_rss = __mem.rss >> 20  # MB
        m_vms = __mem.vms >> 20  # MB
        _data = "{} \t| {}[s] \t| {} \t\t| {}\t| {} \t| {}".format(
            p.cpu_num(), p.cpu_times().user, p.num_threads(), p.cpu_percent(), m_rss, m_vms)
        data.append(_data)
    for line in data:
        print(line)


def start_proc_info():
    _th = threading.Thread(target=get_cpu_mem_info, args=(_cpu_info_event,))
    _th.start()


def stop_proc_info():
    _cpu_info_event.set()
    print("\nWaiting for threads to end")
    # _cpu_info_event.clear()


def timing(func):
    def wrap(*args, **kwargs):
        time1 = time()
        ret = func(*args, **kwargs)
        time2 = time()
        print('{:s} function took {:.3f} ms'.format(func.__name__, (time2 - time1) * 1000.0))
        return ret

    return wrap
