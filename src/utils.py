import string
from typing import Dict

def load_frequency_dict (filename:str, sep: str = "\t") -> Dict[str, int]:

    ret = {}

    with open(filename, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip().split(sep)
            ret[line[0]] = int(line[1])

    return ret


def is_punctuation(x:str) -> bool:

    ret = True

    for c in x:
        if not c in string.punctuation:
            ret = False

    return ret
