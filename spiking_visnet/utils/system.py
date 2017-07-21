from os import mkdir
from os.path import isdir


def mkdir_ifnot(path):
    if not isdir(path):
        mkdir(path)
