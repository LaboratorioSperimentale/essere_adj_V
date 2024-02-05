library(collostructions)


data <- read.csv("output/input_sca.txt", sep = "\t", header = TRUE)
data$WORD <- as.factor(data$WORD)

analysis.result <- collex(data, corpsize = 415000000)

write.csv(analysis.result, "output/output_sca.txt", row.names=FALSE)