import numpy as np
import collections
import tqdm
import csv
import random
from typing import List

import src.utils as u

def extract_adj_frequencies(corpus_filepaths: List[str],
                            output_folder:str,
                            min_freq: int = 10,
                            min_freq_verbs: int = 200) -> None:


    frequenze_adj_in = collections.defaultdict(int)
    frequenze_adj_gen = collections.defaultdict(int)

    frequenze_verbs = collections.defaultdict(int)


    for filename in corpus_filepaths:
        with open(filename, encoding="utf-8") as fin:
            for line in tqdm.tqdm(fin):
                if not line.startswith("<"):
                    line = line.strip().split("\t")

                    if line[3] == "A":

                        if line[2].startswith("in") or line[2].startswith("il") or \
                        line[2].startswith("im") or line[2].startswith("ir"):
                            frequenze_adj_in[line[2]] += 1
                        else:
                            frequenze_adj_gen[line[2]] += 1

                    if line[3] == "V" and all(x.isalpha() for x in line[2]):
                        frequenze_verbs[line[2]] += 1

        # TODO: move to utils
        sorted_in = sorted(frequenze_adj_in.items(), key= lambda x: -x[1])
        sorted_gen = sorted(frequenze_adj_gen.items(), key= lambda x: -x[1])
        sorted_verbs = sorted(frequenze_verbs.items(), key= lambda x: -x[1])

        with open(f"{output_folder}/frequenze_adj_inprefix.txt", "w", encoding="utf-8") as fout:
            for el, f in sorted_in:
                if f > min_freq:
                    print(f"{el}\t{f}", file=fout)

        with open(f"{output_folder}/frequenze_adj_generic.txt", "w", encoding="utf-8") as fout:
            for el, f in sorted_gen:
                if f > min_freq:
                    print(f"{el}\t{f}", file=fout)

        with open(f"{output_folder}/frequenze_verbs.txt", "w", encoding="utf-8") as fout:
            for el, f in sorted_verbs:
                if f > min_freq_verbs:
                    print(f"{el}\t{f}", file=fout)


def build_frequency_table(in_adjectives: str, gen_adjectives: str, triples: str,
                          output_folder:str) -> None:

    freqs_in = u.load_frequency_dict(in_adjectives)
    freqs_gen = u.load_frequency_dict(gen_adjectives)


    with open(triples, encoding="utf-8") as fin, \
        open(f"{output_folder}/triples_with_frequencies.txt", "w", encoding="utf-8") as fout:
            header = fin.readline().strip().split("\t")
            w1, w2, w3 = header

            print(f"{w1}\tF\t{w2}\tF\t{w3}\tF", file=fout)

            for line in fin:
                line = line.strip().split("\t")
                w1, w2, w3 = line

                s = ""
                for el in (w1, w2, w3):

                    f = 0

                    if el in freqs_in:
                        f = freqs_in[el]

                    if el in freqs_gen:
                        f = freqs_gen[el]

                    s+=f"{el}\t{f}\t"

                print(s, file=fout)



def build_pairs(in_adjectives: str, gen_adjectives: str,
                output_folder: str,
                min_freq: int = 20,
                min_cumulative_freq: int = 200) -> None:

    freqs_in = u.load_frequency_dict(in_adjectives)
    freqs_gen = u.load_frequency_dict(gen_adjectives)

    with open(f"{output_folder}/tentative_pairs.txt", "w", encoding="utf-8") as fout_1, \
        open(f"{output_folder}/tentative_groups2-3.txt", "w", encoding="utf-8") as fout_23:

        for adj in freqs_in:
            possible_base = adj[2:]
            if possible_base in freqs_gen and \
                freqs_gen[possible_base] > min_freq and freqs_in[adj] > min_freq and\
                freqs_gen[possible_base] + freqs_in[adj] > min_cumulative_freq:
                print(f"{adj}\t{freqs_in[adj]}\t{possible_base}\t{freqs_gen[possible_base]}", file=fout_1)
            else:
                print(f"{adj}\t{freqs_in[adj]}", file=fout_23)


def extract_cxn_occurrences(corpus_filepaths: List[str],
                            verbs_filepath: str,
                            output_folder:str,
                            min_freq: int = 4) -> None:

    frequenze_verbs = u.load_frequency_dict(verbs_filepath)

    frequenze_cxn = collections.defaultdict(int)
    frequenze_adj_in_cxn = collections.defaultdict(int)

    for filename in corpus_filepaths:
        with open(filename, encoding="utf-8") as fin:

            frase = []
            id_candidates = []
            i=0

            for line in tqdm.tqdm(fin):

                if not line.startswith("<"):
                    line = line.strip().split("\t")

                    frase.append(line)

                    if line[3] == "A":
                        id_candidates.append(i)

                    i+=1

                else:
                    for el in id_candidates:
                        if el > 0 and el < len(frase)-1:
                            x = frase[el-1]
                            adj = frase[el]
                            y = frase[el+1]

                            if x[2] == "essere" and \
                                y[3] == "V" and y[2] in frequenze_verbs and \
                                "mod=f" in y[5]:

                                candidate = tuple(l[2] for l in frase[el-1:el+2])

                                frequenze_cxn[candidate] += 1
                                frequenze_adj_in_cxn[adj[2]] += 1


                    frase = []
                    id_candidates = []
                    i = 0

    sorted_cxns = sorted(frequenze_cxn.items(), key = lambda x : (x[0][1], -x[1]))

    with open(f"{output_folder}/constructions.txt", "w", encoding="utf-8") as fout:

        for candidate, f in sorted_cxns:
            _, adj, _ = candidate

            if frequenze_adj_in_cxn[adj] > min_freq:
                print(f"{' '.join(candidate)}\t{f}", file=fout)


    with open(f"{output_folder}/adj_appear_in_cxn.txt", "w", encoding="utf-8") as fout:

        for adj, f in frequenze_adj_in_cxn.items():
            if f > min_freq:
                print(f"{adj}\t{f}", file=fout)


def build_input_sca (frequenze_adj_in: str,
                     frequenze_adj_generic: str,
                     frequenze_adj_in_cxn: str,
                     output_folder: str) -> None:

    CORPUS_LEN = 415_000_000
    freqs_adj_in = u.load_frequency_dict(frequenze_adj_in)
    freqs_adj_generic = u.load_frequency_dict(frequenze_adj_generic)
    freqs_adj_in_cxn = u.load_frequency_dict(frequenze_adj_in_cxn)

    with open(f"{output_folder}/input_sca.txt", "w", encoding="utf-8") as fout:
        print("WORD\tCXN.FREQ\tCORP.FREQ", file=fout)

        for adj in freqs_adj_in_cxn:
            overall_f = 0
            if adj in freqs_adj_in:
                overall_f = freqs_adj_in[adj]
            if adj in freqs_adj_generic:
                overall_f = freqs_adj_generic[adj]

            print(f"{adj}\t{freqs_adj_in_cxn[adj]}\t{overall_f}", file=fout)


def clean_pairs_outliers(coppie_fname:str,
                         output_folder:str) -> None:

    differences = {}
    values = []

    with open(coppie_fname, encoding="utf-8") as fin:
        for line in fin:
            linesplit = line.strip().split("\t")
            antonimo, f_ant, base, f_base = linesplit
            f_ant, f_base = int(f_ant), int(f_base)

            differences[(antonimo, base)] = (f_ant, f_base, f_ant-f_base)
            values.append(f_ant-f_base)


    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    upper_limit = q3 + 1.5*iqr
    lower_limit = q1 - 1.5*iqr

    print(f"Computed Q1: {q1}, Q3: {q3}, IQR: {iqr}")
    print(f"Computer Upper Limit as Q3 + 1.5*IQR = {upper_limit}")
    print(f"Computer Lower Limit as Q1 - 1.5*IQR = {lower_limit}")

    with open(f"{output_folder}/fant_moltomaggiore_fbase.txt", "w", encoding="utf-8") as fout:
        for x, y in differences.items():
            if y[2] > upper_limit:
                print(f"{x[0]}\t{y[0]}\t{x[1]}\t{y[1]}", file=fout)

    with open(f"{output_folder}/fant_moltominore_fbase.txt", "w", encoding="utf-8") as fout:
        for x, y in differences.items():
            if y[2] < lower_limit:
                print(f"{x[0]}\t{y[0]}\t{x[1]}\t{y[1]}", file=fout)

    with open(f"{output_folder}/coppie_pulite.txt", "w", encoding="utf-8") as fout:
        for x, y in differences.items():
            if y[2] >= lower_limit and y[2] <= upper_limit:
                print(f"{x[0]}\t{y[0]}\t{x[1]}\t{y[1]}", file=fout)


def add_logL_values(coppie_fname:str, gruppo2_fname:str, gruppo3_fname:str,
                    overall_frequenze_in:str, overall_frequenze_gen:str,
                    cxn_frequenze_in:str, cxn_frequenze_gen:str,
                    sca_fname:str,
                    output_folder:str) -> None:

    freqs_adj_in = u.load_frequency_dict(overall_frequenze_in)
    freqs_adj_gen = u.load_frequency_dict(overall_frequenze_gen)

    cxn_freqs_adj_in = u.load_frequency_dict(cxn_frequenze_in)
    cxn_freqs_adj_gen = u.load_frequency_dict(cxn_frequenze_gen)

    sca = {}

    with open(sca_fname, encoding="utf-8") as csv_fin:
        reader = csv.reader(csv_fin)
        header = reader.__next__()

        for line in reader:
            d = dict(zip(header, line))

            sca[d["COLLEX"]] = d["COLL.STR.LOGL"]


    with open(f"{output_folder}/gruppi_LOGL.tsv", "w", encoding="utf-8") as fout:
        print("TIPO\tLEMMA\tABS_FREQ\tFREQ_IN_CXN\tLOGL", file=fout)

        with open(coppie_fname, encoding="utf-8") as fin:

            for line in fin:
                line = line.strip().split("\t")
                antonimo, base = line[0], line[2]

                f_abs_antonimo = freqs_adj_in[antonimo]
                f_cxn_antonimo = 0
                if antonimo in cxn_freqs_adj_in:
                    f_cxn_antonimo = cxn_freqs_adj_in[antonimo]
                logl_antonimo = 0
                if antonimo in sca:
                    logl_antonimo = sca[antonimo]

                print(f"gruppo1\t{antonimo}\t{f_abs_antonimo}\t{f_cxn_antonimo}\t{logl_antonimo}", file=fout)


                f_abs_base = freqs_adj_gen[base]
                f_cxn_base = 0
                if base in cxn_freqs_adj_gen:
                    f_cxn_base = cxn_freqs_adj_gen[base]
                logl_base = 0
                if base in sca:
                    logl_base = sca[base]

                print(f"basi\t{base}\t{f_abs_base}\t{f_cxn_base}\t{logl_base}", file=fout)

        with open(gruppo2_fname, encoding="utf-8") as fin:
            for line in fin:
                line = line.strip().split("\t")
                lemma = line[0]

                f_abs_lemma = freqs_adj_in[lemma]
                f_cxn_lemma = 0
                if lemma in cxn_freqs_adj_in:
                    f_cxn_lemma = cxn_freqs_adj_in[lemma]
                logl_lemma = 0
                if lemma in sca:
                    logl_lemma = sca[lemma]

                print(f"gruppo2\t{lemma}\t{f_abs_lemma}\t{f_cxn_lemma}\t{logl_lemma}", file=fout)

        with open(gruppo3_fname, encoding="utf-8") as fin:
            for line in fin:
                line = line.strip().split("\t")
                lemma = line[0]

                f_abs_lemma = freqs_adj_in[lemma]
                f_cxn_lemma = 0
                if lemma in cxn_freqs_adj_in:
                    f_cxn_lemma = cxn_freqs_adj_in[lemma]
                logl_lemma = 0
                if lemma in sca:
                    logl_lemma = sca[lemma]

                print(f"gruppo3\t{lemma}\t{f_abs_lemma}\t{f_cxn_lemma}\t{logl_lemma}", file=fout)


#TODO: finish here
def extract_adj_contexts(corpus_filepaths: List[str],
                        triples_filepath: str,
                        output_folder:str,
                        ctx_windows: int = 7) -> None:


    adjs = set()
    with open(triples_filepath, encoding="utf-8") as fin:
        fin.readline()

        for line in fin:
            line = line.strip().split("\t")
            w1, w2, w3 = line
            adjs.add(w1)
            adjs.add(w2)
            adjs.add(w3)



    files = {}
    for adj in adjs:
        files[adj] = open(f"{output_folder}/{adj}.txt", "w")

    for filename in corpus_filepaths:
        with open(filename, encoding="utf-8") as fin:

            frase = []
            id_candidates = []
            i=0

            for line in tqdm.tqdm(fin):

                if not line.startswith("<"):
                    line = line.strip().split("\t")

                    frase.append(line)

                    if line[3] == "A" and line[2] in adjs:
                        id_candidates.append(i)

                    i+=1

                else:
                    if len(frase) > 2*ctx_windows+1:

                        for el in id_candidates:
                            if el >= ctx_windows and el < len(frase)-ctx_windows:
                                left_ctx = [l[2] for l in frase[el-ctx_windows:el]]
                                right_ctx = [l[2] for l in frase[el+1:el+ctx_windows+1]]
                                adj = frase[el][2]

                                print(f"{' '.join(left_ctx)}\t{adj}\t{' '.join(right_ctx)}", file=files[adj])

                    frase = []
                    id_candidates = []
                    i = 0


def compute_TTR(triples_filepath:str,
                input_files:List[str], number_of_sentences: int,
                remove_punctuation:bool, output_folder:str,
                random_seed: int = 5642):

    random.seed(random_seed)

    adjs = []
    TTR_dict = {}
    with open(triples_filepath, encoding="utf-8") as fin:
        fin.readline()

        for line in fin:
            line = line.strip().split("\t")
            adjs.append(line)

            for el in line:
                TTR_dict[el] = {"TTR":0,
                              "len_types":0,
                              "n_tokens": 0}





    with open(f"{output_folder}/riepilogo.txt", "w", encoding="utf-8") as fout_riepilogo:
        print("BASE\tTYPES\tTOKENS\tTTR\tMORFOLOGICO\tTYPES\tTOKENS\tTTR\tLESSICALE\tTYPES\tTOKENS\tTTR", file=fout_riepilogo)

        for filename in input_files:
            filename_suffix = filename.split("/")[-1]

            adj = filename_suffix.split(".")[0]
            with open(filename, encoding="utf-8") as fin:
                content = fin.readlines()

                sample = content
                if len(content) > number_of_sentences:
                    sample = random.sample(content, number_of_sentences)

                with open(f"{output_folder}/{filename_suffix}", "w", encoding="utf-8") as fout:
                    print("".join(sample), file=fout)

                types = set()
                n_tokens = 0

                for sentence in sample:
                    sentence = sentence.strip().split("\t")
                    left_ctx, _, right_ctx = sentence

                    left_ctx = left_ctx.split(" ")
                    right_ctx = right_ctx.split(" ")

                    if remove_punctuation:
                        left_ctx = [x for x in left_ctx if not u.is_punctuation(x)]
                        right_ctx = [x for x in right_ctx if not u.is_punctuation(x)]


                    for tok in left_ctx:
                        if all(x.isdigit() for x in tok):
                            types.add("NUM")
                        else:
                            types.add(tok)
                        n_tokens += 1

                    for tok in right_ctx:
                        if all(x.isdigit() for x in tok):
                            types.add("NUM")
                        else:
                            types.add(tok)
                        n_tokens += 1



            TTR = 0
            if n_tokens > 0:
                TTR = len(types)/n_tokens
            TTR_dict[adj] = {"TTR":TTR,
                             "len_types":len(types),
                             "n_tokens": n_tokens}

        for tripla in adjs:
            s = ""
            for el in tripla:
                s+= f"{el}\t{TTR_dict[el]['len_types']}\t{TTR_dict[el]['n_tokens']}\t{TTR_dict[el]['TTR']}\t"
            print(s, file=fout_riepilogo)