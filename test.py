import time
from math import sqrt

from joblib import Parallel, delayed


def my_function(i, param):
    return i * param


import concurrent.futures

if __name__ == "__main__":

    parameters = [5, 6]
    myList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 89]
    inputs = myList
    # with threads
    print("with thread")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        start = time.perf_counter()
        response_process = []
        for i in range(500):
            response_process.append(executor.submit(my_function, {'i': 5, 'param': parameters}))
    print(f'Duration: {time.perf_counter() - start}')
    print(response_process)

    print("with process")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        start = time.perf_counter()
        response_process = []
        for i in range(500):
            response_process.append(executor.submit(my_function, {'i': 5, 'param': parameters}))
    print(f'Duration: {time.perf_counter() - start}')
    print(response_process)

    print("without anything")
    start = time.perf_counter()
    list_ = list()
    for i in range(500):
        res = my_function(i, parameters)
        list_.append(res)
    print(f'Duration: {time.perf_counter() - start}')

    print("other exp:")
    start = time.perf_counter()
    Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
    print(f'Duration: {time.perf_counter() - start}')

    print("without parralell last exp")
    start = time.perf_counter()
    for i in range(10):
        res = sqrt(i ** 2)
    print(f'Duration: {time.perf_counter() - start}')
