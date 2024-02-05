from src import functions as fun

# extract frequecy of all adjectives in corpus
fun.extract_adj_frequencies( # [ CORPUS FILENAME ],
                            "output/")

# build file with frequencies from selected adjs
fun.build_frequency_table("output/frequenze_adj_inprefix.txt",
                          "output/frequenze_adj_generic.txt",
                          "data/triples.txt",
                          "output/")


# build tentative groups
fun.build_pairs("output/frequenze_adj_inprefix.txt",
                "output/frequenze_adj_generic.txt",
                "output/")

#extract constructions
fun.extract_cxn_occurrences(# [ CORPUS FILENAME ],
                            "output/frequenze_verbs.txt",
                            "output/")


#build INPUT for Simple Collostructional Analysis
fun.build_input_sca("output/frequenze_adj_inprefix.txt",
                    "output/frequenze_adj_generic.txt",
                    "output/adj_appear_in_cxn.txt",
                    "output/")


# Run R script to perform Collostructional Analysis -> sca.R

# manually clean pairs and build group2 and group3

# remove outliers
fun.clean_pairs_outliers("data/coppie.txt",
                         "output/")

# manually add new group2 members

# project SCA on data

fun.add_logL_values("output/coppie_pulite.txt", "data/gruppo 2_aumentato.txt", "data/gruppo 3.txt",
                    "output/frequenze_adj_inprefix.txt", "output/frequenze_adj_generic.txt",
                    "output/adj_appear_in_cxn.txt", "output/adj_appear_in_cxn.txt",
                    "output/output_sca_onlysignificant.txt",
                    "output/")


# produce boxplots -> boxplots.R (da sistemare)

