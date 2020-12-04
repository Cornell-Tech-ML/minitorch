import math

## Task 0.1
## Mathematical operators


def mul(x, y):
    ":math:`f(x, y) = x * y`"
    return x * y
    # return float(x * y)


def id(x):
    ":math:`f(x) = x`"
    return x
    # return float(x)


def add(x, y):
    ":math:`f(x, y) = x + y`"
    # TODO: Implement for Task 0.1.
    return x + y
    # return float(x + y)


def neg(x):
    ":math:`f(x) = -x`"
    # TODO: Implement for Task 0.1.
    return -x
    # return (-1.0) * x


def lt(x, y):
    ":math:`f(x) =` 1.0 if x is less than y else 0.0"
    # TODO: Implement for Task 0.1.
    if x < y:
        return 1.0
    else:
        return 0.0


def eq(x, y):
    ":math:`f(x) =` 1.0 if x is equal to y else 0.0"
    # TODO: Implement for Task 0.1.
    if x == y:
        return 1.0
    else:
        return 0.0


def max(x, y):
    ":math:`f(x) =` x if x is greater than y else y"
    # TODO: Implement for Task 0.1.
    if x > y:
        return x
        # return float(x)
    else:
        return y
        # return float(y)


def sigmoid(x):
    if x >= 0:
        return 1 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x):
    if x > 0:
        return x
        # return float(x)
    else:
        return 0.0


def relu_back(x, y):
    ":math:`f(x) =` y if x is greater than 0 else 0"
    # TODO: Implement for Task 0.1.
    if x > 0:
        return y
        # return float(y)
    else:
        return 0.0


EPS = 1e-6


def log(x):
    ":math:`f(x) = log(x)`"
    return math.log(x + EPS)


def exp(x):
    ":math:`f(x) = e^{x}`"
    return math.exp(x)


def log_back(a, b):
    return b / (a + EPS)


def inv(x):
    ":math:`f(x) = 1/x`"
    return 1.0 / x


def inv_back(a, b):
    return -(1.0 / a ** 2) * b


## Task 0.3
## Higher-order functions.


def map(fn):
    """
    Higher-order map.

    .. image:: figs/Ops/maplist.png


    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (one-arg function): Function from one value to one value.

    Returns:
        function : A function that takes a list, applies `fn` to each element, and returns a
        new list
    """
    # raise NotImplementedError('Need to include this file from past assignment.')
    def nested_function(ls):
        return_list = [None] * len(ls)
        for i, l in enumerate(ls):
            return_list[i] = fn(l)
        return return_list

    return nested_function


def negList(ls):
    "Use :func:`map` and :func:`neg` to negate each element in `ls`"
    return map(neg)(ls)


def zipWith(fn):
    """
    Higher-order zipwith (or map2).

    .. image:: figs/Ops/ziplist.png

    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) on each pair of elements.

    """
    # raise NotImplementedError('Need to include this file from past assignment.')
    def nested_function(ls1, ls2):
        # Check both lists have the same lengths
        if len(ls1) != len(ls2):
            raise ValueError("Lists must have the same length")

        return_list = [None] * len(ls1)
        for i, (l1, l2) in enumerate(zip(ls1, ls2)):
            return_list[i] = fn(l1, l2)

        return return_list

    return nested_function


def addLists(ls1, ls2):
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    return zipWith(add)(ls1, ls2)


def reduce(fn, start):
    r"""
    Higher-order reduce.

    .. image:: figs/Ops/reducelist.png


    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`

    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 \ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`
    """
    # raise NotImplementedError('Need to include this file from past assignment.')
    # def nested_function(ls):
    #     if len(ls) == 0:
    #         if start is None:
    #             return 0
    #         else:
    #             return start
    #     it = iter(ls)
    #     if start is None:
    #         value = next(it)
    #     else:
    #         value = start
    #     for element in it:
    #         value = fn(value, element)
    #     return value

    # return nested_function
    def _reduce(ls):
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def sum(ls):
    "Sum up a list using :func:`reduce` and :func:`add`."
    return reduce(add, 0.0)(ls)


def prod(ls):
    "Product of a list using :func:`reduce` and :func:`mul`."
    # return reduce(mul)(ls)
    return reduce(mul, 1.0)(ls)
