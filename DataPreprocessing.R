# setwd("C:/Users/ntruz/Downloads")
# Used to set working directory  

# import need packages 
# use library.

#install.packages("data.table")
#install.packages("tm)

library(data.table)
library(tm)

tweetsFiltered = fread("tweetsFiltered.csv")

df_title <- data.frame(doc_id=row.names(tweetsFiltered),
                       text = tweetsFiltered$Text)

# replace values for easy of understanding
Target = replace(tweetsFiltered$Target, tweetsFiltered$Target == 4, 1)

tweetsFiltered = VCorpus(DataframeSource(df_title))

#process data removing unnecessary infromation
tweetsFiltered = tm_map(tweetsFiltered, removeWords, stopwords("english"))
tweetsFiltered = tm_map(tweetsFiltered, removeNumbers)
tweetsFiltered = tm_map(tweetsFiltered, removePunctuation)
tweetsFiltered = tm_map(tweetsFiltered, stripWhitespace)
tweetsFiltered = tm_map(tweetsFiltered, content_transformer(tolower))
tweetsFiltered = tm_map(tweetsFiltered, stemDocument)

#conduction TF-IDF calcualtion on data
dtm <- DocumentTermMatrix(tweetsFiltered, control = list(weighting =
                                                      function(x)
                                                      weightTfIdf(x, normalize = FALSE)))

#inspect(dtm)

# removing spares words eg. URLS, and mentions ex.
dtm2 <- removeSparseTerms(dtm, sparse = 0.98)

#making a data frame for analysis
tweetData = data.frame(as.matrix(dtm2), y = Target)

fwrite(tweetData, "tweetProcessedData.csv")
