# Load libraries
library(caret)
library(randomForest)
library(pROC)

sink(file = "drive/MyDrive/Cyber\ Security/grams_data/output/output-completo.txt") # Crea un file con i print
print("START")
print(Sys.time())

print("Loading data")
clean   <- read.table( "drive/MyDrive/Cyber\ Security/grams_data/csv/clean-completo.csv", header=FALSE, sep=",", na.strings="NaN", dec=",", strip.white=TRUE)
dga     <- read.table( "drive/MyDrive/Cyber\ Security/grams_data/csv/dga-completo.csv",   header=FALSE, sep=",", na.strings="NaN", dec=",", strip.white=TRUE)
print("data loaded")

m = ncol(clean)
for (j in 4:m)
{
    clean[,j]   <- as.numeric( clean[,j]   )
    dga[,j]     <- as.numeric( dga[,j]     )
}

selected_rows = 3:ncol(clean)

data_clean_full <- clean[,selected_rows]
data_dga_full   <- dga[,selected_rows]
data <- rbind( data_clean_full, data_dga_full )

# Create train and test sets for each dataset
set.seed(1234)

# Perform 10 fold cross validation
# Parameters
l<-c()
accuracy<-0
sensitivity<-0
f1<-0
precision<-0
vec <- 0

# n-folds
n<-10

#Randomly shuffle the data
yourData<-data[sample(nrow(data)),]

#Create N equally size folds
folds <- cut(seq(1,nrow(yourData)),breaks=n,labels=FALSE)

for(i in 1:n){
    print(sprintf("Iteration: %i", i))
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    trainData <- yourData[-testIndexes, ]
    testData <- yourData[testIndexes, ]

    ctrl <- trainControl(method = "none",repeats = 1,classProbs = TRUE,summaryFunction = twoClassSummary)
    system.time({
      rfFit <- train(V3~.,data = trainData,method = "rf",trControl = ctrl)
    })
    # print(rfFit)
    system.time({
      rfPred <- predict(rfFit, testData)
    })
    # print(rfPred)

    importance = varImp(rfFit)
    # print(importance)

    cm <- confusionMatrix(rfPred,as.factor(testData[,1]))
    # print(cm)
    print(cm$overall['Accuracy']) # Per Accuracy
    print(cm$byClass) # Per Sensitivity, F1, Precision
    print(cm$table) # Per matrice confusione

    # print(cm$overall['Accuracy'])
    # print(cm$byClass['Sensitivity'])
    # print(cm$byClass['F1'])
    # print(cm$byClass['Precision'])

    accuracy <- accuracy + cm$overall['Accuracy']
    sensitivity <- sensitivity + cm$byClass['Sensitivity']
    f1 <- f1 + cm$byClass['F1']
    precision <- precision + cm$byClass['Precision']

    cm_d <- as.data.frame(cm$table)
    # print(cm_d) # Per matrice confusione

    vec <- vec + cm_d['Freq'] # Matrice di confusione
    l <- c(l,cm)
}
print("Media K-Fold")
accuracy <- accuracy/n
sensitivity <- sensitivity/n
f1 <- f1/n
precision <- precision/n
vec <- vec/n

print(accuracy)
print(sensitivity)
print(f1)
print(precision)
print("Matrice di confusione")
print(sprintf("TP: %.3f", vec$Freq[[1]]))
print(sprintf("FN: %.3f", vec$Freq[[2]]))
print(sprintf("FP: %.3f", vec$Freq[[3]]))
print(sprintf("TN: %.3f", vec$Freq[[4]]))

warnings()

print(Sys.time())
print("END")
sink(file = NULL) # Chiude il file