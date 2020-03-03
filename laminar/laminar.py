from collections import OrderedDict
import logging
from multiprocessing import Queue, Process, cpu_count
import traceback
from typing import Collection, Callable

import numpy as np

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

class Laminar:
    """
    A class that encompasses everything needed for simple multiprocessing.
    
    ...
    
    Attributes
    ----------
    cores : int
        number of cores the Laminar object should utilize (default 4)
    _processes : collections.OrderedDict
        ordered dictionary that holds processes added by use
    _queue : multiprocessing.Queue
        multiprocessing object that holds process results before joining
    results : dict
        dictionary that holds user-facing results after joining
    
    Methods
    -------
    add_process(name, function, dataset, *args, **kwargs)
        Places a valid process in the object's _processes queue
    show_processes()
        Prints names of processes currently in the object's _processes queue
    drop_process(name)
        Removes process with passed name from the object's _processes queue
    launch_processes()
        Runs processes stored in the object's _processes queue in parallel
    clear_processes()
        Removes all processes from the object's _processes queue
    get_results()
        Returns results from the object's results dictionary
    clear_results()
        Removes all results from the object's results dictionary
            
    """
    def __init__(self, cores=4):
        """
        Parameters
        ----------
        cores : TYPE, optional
            Number of "cores" the object will have. This represents the number
            of parallel processes that will run concurrently. The default is 4.

        Returns
        -------
        None.

        """
        self.cores = cores if cores != "auto" else cpu_count()
        self._processes = OrderedDict()
        self._queue = Queue()
        self.results = {}

    def add_process(self, name: str, function: Callable, dataset: Collection, *args, **kwargs) -> None:
        """
        Creates a 'process' that will execute function on dataset on its own
        core and adds it to the object's queue.
        
        Parameters
        ----------
        name : str
            User-chosen name of process to add.
        function : Callable
            Function or method the process should execute.
        dataset : Collection
            Data object function should operate on.
        *args : TYPE
            Positional arguments to pass in with function.
        **kwargs : TYPE
            Keyword arguments to pass in with function.

        Returns
        -------
        None.

        """
        if name in self._processes.keys():
            logging.info(f" A process with name '{name}' already existed. It has been replaced with new process '{name}'.")

        new_process = Process(target=self.__converter, args=(name, function, dataset, args, kwargs))
        self._processes[name] = new_process

    def show_processes(self) -> None:
        """
        Prints names of processes in object's process queue to the console.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.

        """
        proc_string = ""
        for key in self._processes.keys():
            proc_string = f"{proc_string + key}\n"
        print(proc_string.strip())

    def drop_process(self, name: str) -> None:
        """
        Removes process with name from the object's process queue.

        Parameters
        ----------
        name : str
            Name of process to drop.

        Returns
        -------
        None.

        """
        if name in self._processes:
            del self._processes[name]
        else:
            logging.info(f" Process '{name}' not found.")

    def launch_processes(self) -> None:
        """
        Executes processes in object's process queue in parallel. If there are
        more processes than machine cores, processes will be executed as cores
        become free. All results are stored in object's results dictionary.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.

        """
        for p in self._processes.values():
            p.start()

        for p in self._processes.values():
            q = self._queue.get()
            self.results[q[0]] = q[1]

        for p in self._processes.values():
            p.join()

        self._processes = OrderedDict()

        logging.info("Processes finished.")

    def clear_processes(self) -> None:
        """
        Clears all processes from object's process queue.
        
        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        self._processes = OrderedDict()

    def get_results(self) -> dict:
        """
        Returns results from object's results dictionary.
        
        Parameters
        ----------
        None.

        Returns
        -------
        dict
            Dictionary containing process names as keys and completed results
            as values.

        """
        return self.results

    def clear_results(self) -> None:
        """
        Clears all results from object's results dictionary.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        self.results = {}

    def __converter(self, name: str, function: Callable, data_shard: Collection, *args) -> None:
        """
        Private class method that calls the passed function with the passed data_shard
        as an argument, then places the result in the object's results dictionary.
        Also passes through any args required for the function.
        
        Parameters
        ----------
        name : str
            User-chosen name of process to add.
        function : Callable
            Function or method the process should execute.
        dataset : Collection
            Data object function should operate on.
        *args : TYPE
            Positional arguments to pass in with function. This will emcompass
            any **kwargs that were given to add_process().

        Returns
        -------
        None.

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
    """Private module function that calls the passed function with the passed data_shard
    as an argument, then places the result in the queue. Also passes through any
    args required for the function.

    Parameters
    ----------
    function : Callable
        Function or method the user wishes to parallelize.
    data_shard : Collection
        Data object that is a subset of the master data object passed to the
        laminar function.
    queue : Queue
        Multiprocessing queue that holds process results.

    Returns
    -------
    None.

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
    """Parallelizes analysis of an iterable.

    Parallelization function that breaks up an iterable into data shards, then
    analyzes each data shard in parallel. Returns a list of results from each
    data shard.

    Parameters
    ----------
    function : Callable
        Function with which to analyze data.
    data : Collection
        The iterable to be analyzed in parallel.
    *args : 
        Positional arguments required by passed function.
    **kwargs : 
        Keyword arguments required by passed function.
        - cores: Can be included in **kwargs. Number of cores to run in parallel.
                Default is 4 cores.
        - sort_results: Can be included in **kwargs. Sorts results dictionary.

    Returns
    -------
    results : dict
        Dictionary of results from each parallel process, named according to
        position in data iterable.

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

    cores = kwargs.pop("cores", 4)
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

    Parameters
    ----------
    function : Callable
        Function with which to analyze data.
    data_list : list
        List of data objects to be analyzed in parallel.
    *args : None
        Positional arguments required by function.
    **kwargs : None
        Keyword arguments required by function.
        - cores: Can be included in **kwargs. Number of cores to run in parallel.
                Default is 4 cores.

    Returns
    -------
    results : dict
        Dictionary of results from each parallel process, named according to 
        position in data_list iterable.

    Example:
        {'data_position_0': 675,
        'data_position_1': 1800,
        'data_position_2': 2925}


    """

    cores = kwargs.pop("cores", 4)

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
