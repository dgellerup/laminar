import pytest
import numpy as np
import itertools
from laminar import laminar

def test_iterflow_basic():
    """
    Basic tests from the tutorial.
    """
    import numpy as np
    import pandas as pd
    from laminar import laminar
    from laminar import laminar_examples as le

    result = laminar.iter_flow(le.single_total, le.laminar_df['Col1'])
    assert le.laminar_df['Col1'].sum() == np.sum([x for x in result.values()])

    result = laminar.iter_flow(le.single_total, le.laminar_df['Col1'], odd=True)
    assert np.sum([x for x in le.laminar_df['Col1'] if x%2 != 0 ]) == np.sum([x for x in result.values()])

def test_iterflow_complex_example():
    """
    This is an interesting example because
    the function foo was not passed directly into iter.flow.
    """
    import numpy as np
    import pandas as pd
    from laminar import laminar

    reference_list = ["a","b","c","d","e","f","g","h","i","j",
                      "k","l","m","n","o","p","q","r","s","t","u","v", "w"]
    iterable_list = [7,4,11,11,14,22,14,17,11,3]

    def foo(x, rl):
        """
        >>> foo(0,["A","B","C"])
        'A'
        >>> foo(1,["A","B","C"])
        'B'
        >>> foo(3,["A","B","C"])
        IndexError
        """
        return(rl[x])

    def wrapfoo(l,rl):
        return("".join([foo(x,rl) for x in l]))

    result = laminar.iter_flow(wrapfoo, iterable_list, rl = reference_list, cores=2, sort_results=True)
    print(result)
    assert list(result.values()) == ['hello', 'world']
