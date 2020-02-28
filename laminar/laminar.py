from collections import OrderedDict
import logging
from multiprocessing import Queue, Process, cpu_count
import traceback
from typing import Collection, Callable

import numpy as np

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

class Laminar:

    def __init__(self, cores=cpu_count()):
        self.cores = cores
        self._processes = OrderedDict()
        self._queue = Queue()
        self.results = {}

    def add_process(self, name: str, function: Callable, dataset: Collection, *args, **kwargs) -> None:
        if name in self._processes.keys():
            logging.info(f" A process with name '{name}' already existed. It has been replaced with new process '{name}'.")

        new_process = Process(target=self.__converter, args=(name, function, dataset, args, kwargs))
        self._processes[name] = new_process

    def show_processes(self) -> None:
        proc_string = ""
        for key in self._processes.keys():
            proc_string = f"{proc_string + key}\n"
        print(proc_string.strip())

    def drop_process(self, name: str) -> None:
        if name in self._processes:
            del self._processes[name]
        else:
            logging.info(f" Process '{name}' not found.")

    def launch_processes(self) -> str:
        for p in self._processes.values():
            p.start()

        for p in self._processes.values():
            q = self._queue.get()
            self.results[q[0]] = q[1]

        for p in self._processes.values():
            p.join()

        self._processes = OrderedDict()

        return "Processes finished."

    def clear_processes(self) -> None:
        self._processes = OrderedDict()

    def get_results(self) -> dict:
        return self.results

    def clear_results(self) -> None:
        self.results = {}

    def __converter(self, name: str, function: Callable, data_shard: Collection, *args) -> None:
        """Module function that calls the passed function with the passed data_shard
        as an argument, then places the result in the queue. Also passes through any
        args required for the function (if passed in).

        Args:
            function: Function object the user wishes to parallelize.
            data_shard: Data object that is a subset of the master data object passed
                to the laminar function.
            queue: Multiprocessing queue that holds process results.

        Returns:
            None

        """

        kwargs, args = args[-1], args[0]
        try:
            result = function(data_shard, *args, **kwargs)
        except Exception as e:
            logging.warning(f" Exception occurred for process '{name}.'")
            logging.exception(e)
            
            result = traceback.format_exc()

        self._queue.put((name, result))


def __converter(name: str, function: Callable, data_shard: Collection, queue: Queue, *args) -> None:
    """Module function that calls the passed function with the passed data_shard
    as an argument, then places the result in the queue. Also passes through any
    args required for the function (if passed in).

    Args:
        function: Function object the user wishes to parallelize.
        data_shard: Data object that is a subset of the master data object passed
            to the laminar function.
        queue: Multiprocessing queue that holds process results.

    Returns:
        None

    """

    kwargs, args = args[-1], args[0]
    try:
        result = function(data_shard, *args, **kwargs)
    except Exception as e:
        logging.warning(f" Exception occurred for process {name}.")
        logging.exception(e)
            
        result = traceback.format_exc()

    queue.put((name, result))


def iter_flow(function: Callable, data: Collection, *args, **kwargs) -> dict:
    """Parallelizes analysis of a list.

    Parallelization function that breaks up an iterable into data shards,
    then analyzes each data shard in parallel. Returns a list of results from each
    data shard.

    Args:
        function: Function with which to analyze data.
        data: The iterable to be analyzed in parallel.
        *args: Positional arguments required by function.
        **kwargs: Keyword arguments required by function.
            - cores: Can be included in **kwargs. Number of cores to run in parallel.
                    Default is number of cores present on the current machine.
            - sort_results: Can be included in **kwargs. Sorts results dict.

    Returns:
        results (dict): Dictionary of results from each parallel process, named
            according to position in data iterable.

        Example:
            {'data[0-25]': 17,
             'data[26-50]': 37,
             'data[51-75]': 60,
             'data[76-100]': 86,
             'data[101-125]: 115,
             'data[126-150]': 105,
             'data[151-175]': 120,
             'data[176-200]': 135}

    """

    cores = kwargs.pop("cores", cpu_count())
    sort_results = kwargs.pop("sort_results", False)

    if cores > cpu_count():
        cores = cpu_count()

    if len(data) == 0:

        return {"data[empty]": None}

    elif len(data) > cores:

        data_split = np.array_split(data, cores)

    else:

        data_split = np.array_split(data, len(data))

    queue = Queue()

    processes = []

    ordered_names = []
    end = -1
    for dataset in data_split:
        start = end + 1
        end += len(dataset)
        name = f"data[{start}-{end}]"
        ordered_names.append(name)
        new_process = Process(target=__converter, args=(name, function, dataset, queue, args, kwargs))
        processes.append(new_process)

    for p in processes:
        p.start()

    results = {}
    for p in processes:
        q = queue.get()
        results[q[0]] = q[1]

    for p in processes:
        p.join()

    if sort_results:
        results = {k:results[k] for k in ordered_names}

    return results


def list_flow(function: Callable, data_list: Collection, *args, **kwargs) -> dict:
    """Parallelizes analysis of a list.

    Parallelization function that sends each data object in a list to its own
    process to be analyzed in parallel. Returns a list of results from each process.

    Args:
        function: Function with which to analyze data.
        data_lsit (list): List of data objects to be analyzed in parallel.
        *args: Positional arguments required by function.
        **kwargs: Keyword arguments required by function.
            - cores: Can be included in **kwargs. Number of cores to run in parallel.
                    Default is number of cores present on the current machine.

    Returns:
        results (dict): Dictionary of results from each parallel process, named
            according to position in data_list iterable.

        Example:
            {'data_position_0': 675,
            'data_position_1': 1800,
            'data_position_2': 2925}


    """

    cores = kwargs.pop("cores", cpu_count())

    if cores > cpu_count():
        cores = cpu_count()

    queue = Queue()

    processes = []

    i = 0
    for dataset in data_list:
        name = f"data_position_{i}"
        new_process = Process(target=__converter, args=(name, function, dataset, queue, args, kwargs))
        processes.append(new_process)
        i += 1

    for p in processes:
        p.start()

    results = {}
    for p in processes:
        q = queue.get()
        results[q[0]] = q[1]

    for p in processes:
        p.join()

    return results
