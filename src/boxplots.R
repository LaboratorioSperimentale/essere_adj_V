data <- read.csv("output/gruppi_LOGL.tsv", sep = "\t", header = TRUE)

boxplot(LOGL ~ tipo, data = data)