from typing import Dict

def load_frequency_dict (filename:str, sep: str = "\t") -> Dict[str, int]:

    ret = {}

    with open(filename, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip().split(sep)
            ret[line[0]] = int(line[1])

    return ret
