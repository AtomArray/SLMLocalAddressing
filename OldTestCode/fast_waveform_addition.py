import numpy as np
import multiprocessing

def workerProcess():
    print("Hello")

def main():
    print("Main")

    t1 = multiprocessing.Process(target=workerProcess)
    t2 = multiprocessing.Process(target=workerProcess)
    t1.start()
    t2.start()

if __name__ == "__main__":
    main()
