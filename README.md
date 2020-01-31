[![CircleCI](https://img.shields.io/circleci/build/github/dgellerup/laminar?logo=circleci&token=8ecea183c3e45f955afdad617348f592d4fc4346)](https://circleci.com/gh/dgellerup/laminar/tree/master)
[![Coverage Status](https://coveralls.io/repos/github/dgellerup/laminar/badge.svg?branch=master)](https://coveralls.io/github/dgellerup/laminar?branch=master)
![PyPI](https://img.shields.io/pypi/v/laminar)
![PyPI - Python Version](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8-blue)
![PyPI - Downloads](https://img.shields.io/pypi/dm/laminar)

*Special thanks to Jim Fan for relinquishing the name "laminar" to me on pypi.org. Much appreciated, Jim!*

# laminar  
__Laminar seeks to take most of the hassle out of parallel processing in Python by providing user-friendly parallelization functionality.__  

__Module Functions__  
results = laminar.iter_flow(my_function, my_iterable)  
results = laminar.list_flow(my_function, my_list_of_data)

__Class Usage__  
my_lam = laminar.Laminar()  
my_lam.add_process("process_1", function_1, my_data)  
my_lam.add_process("process_2", function_2, my_other_data)  
my_lam.launch_processes()  
results = my_lam.get_results()  

## Usage
### Installation
Laminar is delivered as a package. To install, activate your preferred environment, then use:  

`pip install laminar`  

Laminar's `laminar` module only requires one third-party library, which is `numpy`. laminar_examples, a module with some practice functions and data objects, also requires `pandas`. Both libraries will be automatically included in the laminar installation.

### Importing
You can use laminar by placing `from laminar import laminar` or `import laminar.laminar as <some_alias>` at the top of your python file. In order to practice/test laminar with built-in functions and data, place `from laminar import laminar_examples` or `import laminar.laminar_examples as <some_alias>` at the top of your python file.  

If only using the class `Laminar` you can import it directly with `from laminar.laminar import Laminar`.

### Using laminar
laminar currently consists of a class `Laminar` as well as two module functions that are designed to work with different data configurations, `laminar.iter_flow` and `laminar.list_flow`.

The Laminar class provides an instance that manages distinct processes and stores results. Class methods are available that allow the user to view, drop, clear, and launch processes.  

To use the Laminar class, create a Laminar instance:  

`my_lam = Laminar()`  

Laminar class declarations have one optional argument for number of cores, which defaults to the number of cores on the current machine. Thus, if the user only wants to utilize two cores, the declaration would be:  

`my_lam = Laminar(2)`  

To add a process to the object's process batch, simply use the add_process() class method, which is very similar to the module function calls listed below, except add_process() also requires the user to pass a string as the name of the process. This name can be any string.  

`my_lam.add_process('process_1', function_1, my_data)`  

If more processes are added than the number of cores available, the process batch acts like a first in/first out queue. The most recent process will be added and the first process added to the batch will be removed.

Both of the module functions accept \*args and \*\*kwargs, which should be passed after `data`, so if `function` takes arg1 and arg2, like:

`function(arg1, arg2)`  

you should call `laminar` like so:  

`laminar.iter_flow(function, data, arg1, arg2)`  
or  
`laminar.iter_flow(function, data, arg1=arg1, arg2=arg2)`  
or in the case of \*args with \*\*kwargs  
`laminar.iter_flow(function, data, arg1, arg2, kwarg=other_arg)`

`laminar.iter_flow` is designed to work with a single iterable, such as a pandas DataFrame, a python list, etc. When you pass an iterable to `laminar.iter_flow`, it will automatically break your data up into chunks based on how many cores your machine has. It then queues up each chunk to be given to a core, which performs the work, then passes the data back as a descriptive dictionary of results. For example, a list of 1,000,000 integers is broken into chunks of length 250,000 on a machine with four cores. Each chunk is summed (as an example) by a core, and the results from each core are returned in a dict of size N = # cores. You are then able to combine the results in whatever way fits the computation that you need. For example, if the function passed to `laminar.iter_flow` computes the sum, then the values in the results dict should be summed to produce a total for the entire iterable.

### Laminar Class Definition  
| Attribute | Description |
| :----: | :----: |
| `cores` | Number of cores available in an instance. This can be set manually in the instance declaration; it defaults to `cpu_count()`, which is number of cores available on your machine. |
| `results` | Dictionary that holds the results from the `launch_processes` method. Initializes to an empty dict. |
| `_processes` | `collections.OrderedDict()` that holds processes added by `add_process()`. |
| `_queue` | `multiprocessing.Queue()` that manages parallel processes. |  

| Method | Argument(s) | Returns | Description |
|:----:|:----:|:----:| :----: |
| `add_process()` | `name: str`, `function: Callable`, `dataset: Collection`, `*args`, `**kwargs`) | `None` | Add a named process to an instance's process pool. Process must include a name, function, and some data (in reality, this can be anything). |
| `show_processes()` | `None` | `None` | Displays processes currently in instance process pool. |
| `drop_process()` | `name: str` | `None` | Removes process with name of `name` from instance process pool. |
| `clear_processes()` | `None` | `None` | Removes all processes from instance process pool. |
| `launch_processes()` | `None` | `str: "Processes finished."` | Run all instance processes in parallel. |
| `get_results()` | `None` | `self.results: dict` | Returns the instance results dictionary. |
| `clear_results()` | `None` | `None` | Removes all results from instance results dictionary. |

### Module Function Examples
To illustrate how one would use laminar in their workflow, we'll use some premade functions and data structures located in `laminar_examples`. To shorten the following code examples up, we'll import `laminar_examples` as an alias `le` and use this alias throughout the rest of this readme.  

`from laminar import laminar_examples as le`

#### laminar_examples.single_total
`le.single_total` is a simple function that accepts a single iterable and returns the sum total of the values in that iterable. `le.single_total([1, 2, 1])` returns `4`.

#### laminar_examples.multi_tally
`le.multi_tally` is a simple funtion that accepts a Pandas DataFrame and returns the number of rows that sum to greater than 25. `le.multi_tally(pd.DataFrame({'Col1': [12, 12], 'Col2': [12, 14]})` returns `1` because the row at index 1 sums to `12 + 14 = 26`, which meets the function's criteria, but the row at index 0 sums to `12 + 12 = 24`, which does not.

#### laminar_examples.laminar_df
`le.laminar_df` is a Pandas DataFrame that constitutes 3 columns ['Col1', 'Col2', 'Col3'], each of which contains different integer values.

| Col1 | Col2 | Col3 |
|:----:|:----:|:----:|
|1|6|11|
|2|7|12|
|3|8|13|
|4|9|14|
|5|10|15|
|2|12|22|
|4|6|16|
|...|...|...|

#### Example 1: Single iterable, single_total()
`laminar.iter_flow(le.single_total, le.laminar_df['Col1'])` returns  

`{`  
`'data[0-5]': 17,`  
`'data[12-17]': 60,`  
`'data[18-23]': 86,`  
`'data[24-29]': 115,`  
`'data[30-34]': 105,`  
`'data[35-39]': 120,`  
`'data[40-44]': 135,`  
`'data[6-11]': 37,`  
`}`

which is a dictionary describing the results for each section of your data. Each key/value pair in the returned dict corresponds to a segment of the iterable that was broken out and given to a process, with the key containing which portion of the data the result matches to. To complete your analysis, you can use whichever function coincides with the intended behavior of your analysis. In this case, since we are summing values, we can use `sum()`.

The end result can look like one of these examples, although it doesn't have to:
`result = sum(laminar.iter_flow(le.single_total, le.laminar_df['Col1']).values())`

or

`result = laminar.iter_flow(le.single_total, le.laminar_df['Col1'])`

`result = sum(result.values())`

where

`result = 675`


#### Example 2: Pandas DataFrame, multi_tally()
`laminar.iter_flow(le.multi_tally, le.laminar_df)` returns  

`{`  
`'data[0-5]': 3,`  
`'data[12-17]': 6,`  
`'data[18-23]': 6,`   
`'data[24-29]': 6,`   
`'data[30-34]': 5,`   
`'data[35-39]': 5,`   
`'data[40-44]': 5,`   
`'data[6-11]': 6,`  
`}`

 which is a dict of counts. Each count is the return value for a segment of the data that was broken out and given to a process. To complete your analysis, you can use whichever function coincides with the intended behavior of your analysis. In this case, since we are counting values, it makes sense to use `sum()`.

The end result can look like one of these examples, although it doesn't have to:  
`result = sum(laminar.iter_flow(le.multi_tally, le.laminar_df).values())`  

or

`result = laminar.iter_flow(le.multi_tally, le.laminar_df)`

`result = sum(result.values())`

where

`result = 42`


#### Example 3: List of single iterables, single_total()
`laminar.list_flow(le.single_total, [le.laminar_df[col] for col in le.laminar_df.columns])` returns  
`{`  
`'data_position_0': 675,`  
`'data_position_1': 1800,`  
`'data_position_2': 2925,`  
`}`  
which is a list of the totals for each column in `le.laminar_df`. With this usage, a user can pass a list of iterables to `list_flow`; each iterable will be passed to its own process. This is useful for when a user intends to use the same function on multiple iterables, which can be columns in the same DataFrame, or independent lists.
`laminar.list_flow(laminar_examples.single_total, [laminar_examples.laminar_df[col] for col in laminar_examples.laminar_df.columns])` returns `[675, 1800, 2925]`, which is a list of the totals for each column. With this usage, a user can pass a list of iterables to `list_flow`; each iterable will be passed to its own process. This is useful for when a user intends to use the same function on multiple iterables, which can be columns in the same DataFrame, or independent lists.

`columns_list = [le.laminar_df[col] for col in le.laminar_df.columns]`

`result = laminar.list_flow(le.single_total, columns_list)`

where

`result = {'data_position_0': 675, 'data_position_1': 1800, 'data_position_2': 2925}`


#### Example 4: List of Pandas DataFrames, multi_tally()
`laminar.list_flow(le.multi_tally, [le.laminar_df]*3)` returns  
`{`  
`'data_position_0': 42,`  
`'data_position_1': 42,`  
`'data_position_2': 42,`  
`}`.  
The result values are the same because we passed a list of 3 identical DataFrames; feel free to test this with different DataFrames of your own making.

`data_frames_list = [le.laminar_df]*3`

`result = laminar.list_flow(le.multi_tally, data_frames_list)`

where

`result = {'data_position_0': 42, 'data_position_1': 42, 'data_position_2': 42}`


## Benchmarks
To date, laminar has been tested against traditional iterative analysis on the following functions:  

String search function: count_snps()  

### Parameters

**Files:**  

sample-1_S1_R1_001.fastq.gz  
sample-1_S1_R2_001.fastq.gz  

**Total size of files:**  

26M  

**Length of Pandas DataFrame (going forward referred to as pd.DataFrame) object representation of combined files:**  

224706 rows

**Results:**  

Traditional count_snps(pd.DataFrame): 42.6 seconds  

Parallelized laminar.iter_flow(count_snps, pd.DataFrame): 17.49 seconds  

Percent speedup: 58.96% faster


### Final Notes
Which laminar tool a user will use depends on the structure of their data and the function that will be applied to that data. `laminar.list_flow` is not confined to operating on Pandas DataFrames; any list of iterable data objects can be passed to list_flow.

A basic rule of thumb is to use `laminar.iter_flow` for a single data object that one wishes to break into pieces in order to process it faster. `laminar.list_flow` is to be used in a situation where the user has multiple data objects that he or she wishes to be analyzed by the same function in parallel.
