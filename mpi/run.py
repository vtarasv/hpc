#!/bin/env python3
# -*- coding: utf-8 -*-

from mpi4py import MPI
from os import environ
from typing import *

def main():
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    if mpi_rank == 0:
        return mpi_root(mpi_comm)
    else:
        return mpi_nonroot(mpi_comm)

def mpi_root(mpi_comm):
    import random
    random_number = random.randrange(2**32)
    mpi_comm.bcast(random_number)
    print('Controller @ MPI Rank   0:  Input {}'.format(random_number))
    response_array = mpi_comm.gather(None)
    mpi_size = mpi_comm.Get_size()

    for i in range(1, mpi_size):
        if random_number + i == response_array[i][1]:
            result = 'OK'
        else:
            result = 'BAD'
        print('   Worker at MPI Rank {: >3}: Output {} is {} (from {})'
            .format(
                i,
                response_array[i][1],
                result,
                response_array[i][0],
            )
        )
    return 0

def mpi_nonroot(mpi_comm):
    mpi_rank = mpi_comm.Get_rank()
    random_number = mpi_comm.bcast(None)
    response_number = random_number + mpi_rank
    response = (
        MPI.Get_processor_name(),
        response_number,
    )
    mpi_comm.gather(response)
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
