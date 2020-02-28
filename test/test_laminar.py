from multiprocessing import Queue, Process, cpu_count

from context import laminar
from context import laminar_examples as le


def test_converter():
    queue = Queue()

    def sum_func(sum_list):
        return sum(sum_list)

    laminar.__converter("test12", sum_func, [2, 4, 6, 8], queue, [], {})

    q = queue.get()

    assert q[0] == "test12"
    assert q[1] == 20

    def add_to_list(ls):
        return ls.append('caboose')

    laminar.__converter("testfail", add_to_list, 7, queue, [], {})

    q = queue.get()

    assert q[0] == "testfail"
    assert str(q[1]).startswith('Traceback')


def test_iter_flow():
    result = laminar.iter_flow(le.single_total, le.laminar_df['Col1'], cores=8)

    assert result == {
                        'data[0-5]': 17,
                        'data[12-17]': 60,
                        'data[18-23]': 86,
                        'data[24-29]': 115,
                        'data[30-34]': 105,
                        'data[35-39]': 120,
                        'data[40-44]': 135,
                        'data[6-11]': 37,
                        }

    def sum_func(sum_list):
        return sum(sum_list)

    short_list = [1, 2, 3]

    result = laminar.iter_flow(sum_func, short_list, cores=4)

    assert result == {
                        'data[2-2]': 3,
                        'data[0-0]': 1,
                        'data[1-1]': 2
                        }

    result = laminar.iter_flow(sum_func, short_list, cores=9)

    assert result == {
                        'data[2-2]': 3,
                        'data[0-0]': 1,
                        'data[1-1]': 2
                        }

    result = laminar.iter_flow(le.single_total, short_list, True, cores=1)

    assert result.get('data[0-2]') == 4

    result = laminar.iter_flow(le.multi_tally, le.laminar_df, cores=8)

    assert result.get('data[0-5]') == 3

    result = laminar.iter_flow(lambda x: sum(x), [])

    assert result == {
                        'data[empty]': None
                        }

def test_list_flow():
    result = laminar.list_flow(le.single_total, [le.laminar_df[col] for col in le.laminar_df.columns])

    assert result == {
                        'data_position_0': 675,
                        'data_position_1': 1800,
                        'data_position_2': 2925,
                        }

    def sum_func(sum_list):
        return sum(sum_list)

    multi_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    result = laminar.list_flow(sum_func, multi_list, cores=9)

    assert result == {
                        'data_position_0': 6,
                        'data_position_1': 15,
                        'data_position_2': 24
                        }


def test_init_my_lam(my_lam):
    assert my_lam.cores == 2
    assert my_lam.results == {}


def test_my_lam(my_lam):

    def square(ls):
        result = [x*x for x in ls]
        return result

    def cube(ls):
        result = [x*x*x for x in ls]
        return result

    my_lam.add_process('square1', square, [0, 1, 2, 3])

    assert len(my_lam._processes) == 1

    my_lam.add_process('cube1', cube, [0, 1, 2, 3])

    assert len(my_lam._processes) == 2

    my_lam.drop_process('cube1')

    assert len(my_lam._processes) == 1

    my_lam.add_process('cube2', cube, [1, 1, 2, 2])

    def sum_func(sum_list):
        return sum(sum_list)

    my_lam._Laminar__converter("test12", sum_func, [2, 4, 6, 8], [], {})

    q = my_lam._queue.get()

    assert q[0] == "test12"
    assert q[1] == 20

    my_lam.add_process('cube3', cube, [1, 2, 1, 2])

    assert my_lam._processes.get('cube3') != None

    proc = my_lam.show_processes()

    assert proc == None

    my_lam.launch_processes()

    result = my_lam.get_results()

    assert result == {'cube2': [1, 1, 8, 8], 'cube3': [1, 8, 1, 8], 'square1': [0, 1, 4, 9]}
    assert len(my_lam._processes) == 0

    my_lam.add_process('square2', square, [0, 1, 2, 3])
    my_lam.add_process('cube3', cube, [1, 1, 2, 2])

    assert len(my_lam._processes) == 2

    my_lam.clear_processes()

    assert len(my_lam._processes) == 0

    my_lam.add_process('square2', square, [0, 1, 2, 3])
    my_lam.launch_processes()

    assert my_lam.get_results().get('square2') == [0, 1, 4, 9]

    def add_to_list(ls):
        return ls.append('caboose')

    my_lam.add_process('fail', add_to_list, 7)
    my_lam.launch_processes()

    result = my_lam.get_results()
    assert str(result.get('fail')).startswith('Traceback')
