from src import functions as fun

import glob

fun.extract_adj_contexts( # [CORPUS FILEPATH],
                         "data/triples.txt",
                         "output/contesti/")

fun.compute_TTR("data/triples.txt",
                glob.glob("output/contesti/*"),
                150,
                True,
                "output/samples_contesti/")