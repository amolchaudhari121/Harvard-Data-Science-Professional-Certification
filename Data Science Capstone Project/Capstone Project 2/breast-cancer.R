#######################################################################################
#   Author ---  Amol Chaudhari
#   Title ---   Harvard Data Science Capstone Project 
#   Topic ---   Breast Cancer Diagnosys Prediction 
#   Date  ---   22/06/2025
#######################################################################################

# Open required package libraries
# Installing the required packeges we want in the study

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(ggdendro)) install.packages("ggdendro", repos = "http://cran.us.r-project.org")
if(!require(dendextend)) install.packages("dendextend", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(RColorBrewer)) install.packages("RColorBrewer", repos = "http://cran.us.r-project.org")
if(!require(xfun)) install.packages("xfun", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", dependencies = c("Depends", "Suggests"), repos = "http://cran.us.r-project.org")

# Set all numeric outputs to 3 digits (unless otherwise specified in code chunks)
options(digits = 3)

# Download data-set containing both features and outcome data
data_url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

# Download data-set description as a 
names_url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names"

# download data and save to data.frame
wdbc_data <- fread(data_url)

# add column names based on information provided in WPBC.names text file
column_names <- c("id", "diagnosis", "radius_m", "texture_m", "perimeter_m", "area_m", "smoothness_m", "compactness_m",
           "concavity_m", "concave_points_m", "symmetry_m", "fractal_dim_m", "radius_se", "texture_se", "perimeter_se", 
           "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave_points_se", "symmetry_se",
           "fractal_dim_se", "radius_w", "texture_w", "perimeter_w", "area_w", "smoothness_w", "compactness_w",
           "concavity_w", "concave_points_w", "symmetry_w", "fractal_dim_w")

colnames(wdbc_data) <- column_names

# Create plot theme to apply to ggplot2 element text throughout report
plot_theme <- theme(plot.caption = element_text(size = 12, face = "italic"), axis.title = element_text(size = 12))

#######################################################################################
### EXPLORATORY ANALYSIS
#######################################################################################

### EXPLORE FEATURES

# Extract unique features from column names object using stringr functions
features <- column_names[-c(1,2)] %>% str_replace("[^_]+$", "") %>% unique() %>% str_replace_all("_", " ") %>% str_to_title()

# Create data.frame with descriptions for each of the unique features
features_list <- data.frame(Feature = features, Description = 
                              c("Mean of distances from center to points on the perimeter of individual nuclei",
                                "Variance (standard deviation) of grey-scale intensitities in the component pixels",
                                "Perimeter length of each nucleus",
                                "Area as measured by counting pixels within each nucleus",
                                "Local variation in radius lengths",
                                "Combination of perimeter and area using the formula: (perimeter^2 / area - 1.0)",
                                "Number and severity of concavities (indentations) in the nuclear contour",
                                "Number of concavities in the nuclear contour",
                                "Symmetry of the nuclei as measured by length differences between lines perpendicular to the major axis and the cell boundary",
                                "Fractal dimension based on the 'coastline approximation' - 1.0"))

# Create table showing summary data for the mean scores for each of the ten features
summary(wdbc_data[,3:12])

# Create table showing summary data for the worst scores for each of the ten features
summary(wdbc_data[,23:32])

# Create table showing summary data for the standard error scores for each of the ten features
summary(wdbc_data[,13:22])

# Plot and facet wrap density plots for each feature by diagnosis
wdbc_data %>% select(-id) %>%
  gather("feature", "value", -diagnosis) %>%
  ggplot(aes(value, fill = diagnosis)) +
  geom_density(alpha = 0.5) +
  xlab("Feature values") +
  ylab("Density") +
  theme(legend.position = "top",
        axis.text.x = element_blank(), axis.text.y = element_blank(),
        legend.title=element_blank()) +
  scale_fill_discrete(labels = c("Benign", "Malignant")) +
  facet_wrap(~ feature, scales = "free", ncol = 3)

### INITIAL DAT TIDY

# Remove id column
wdbc_data <- wdbc_data[,-1]

# Convert diagnosis to a factor
wdbc_data$diagnosis <- as.factor(wdbc_data$diagnosis)

# Generate features matrix and outcome (diagnosis) list
wdbc <- list(x = as.matrix(wdbc_data[,-1]), y = wdbc_data$diagnosis)

### CREATE TRAIN/TEST DATA-SETS

# Split into train and test sets (80:20)
set.seed(200, sample.kind = "Rounding")
test_index <- createDataPartition(wdbc$y, times = 1, p = 0.2, list = FALSE)
train <- list(x = wdbc$x[-test_index,], y = wdbc$y[-test_index])
test <- list(x = wdbc$x[test_index,], y = wdbc$y[test_index])

### NORMALISE TRAIN DATA

# Centre the train$x data around zero by subtracting the column mean 
train_c <- sweep(train$x, 2, colMeans(train$x))

# Scale the train$x data by dividing by the column standard deviation
train_s <- sweep(train_c, 2, colSds(train$x), FUN = "/")

### DISTANCE BETWEEN SAMPLES

# Use dist function to calculate distance between each sample
d_samples <- dist(train_s)

# Average distance between all samples
ave_dist_samples <- round(mean(as.matrix(d_samples)),2)

# Distance between benign samples
distb_b_samples <- round(mean(as.matrix(d_samples)[train$y=="B"]),2)

# Distance between malignant samples
distm_m_samples <- round(mean(as.matrix(d_samples)[train$y=="M"]),2)

# Distance between benign and malignant samples
distb_m_samples <- round(mean(as.matrix(d_samples)[train$y=="M", train$y=="B"]),2)

# Create heatmap of distance between samples
heatmap(as.matrix(d_samples), symm = T, revC = T,
        col = brewer.pal(4, "Blues"),
        ColSideColors = ifelse(train$y=="B", "green", "red"),
        RowSideColors = ifelse(train$y=="B", "green", "red"),
        labRow = NA, labCol = NA)

### VARIANCE BBETWEEN FEATURES

# Identifying zero variance predictors
nzv <- nearZeroVar(train_s, saveMetrics= TRUE)

# Hierarchical clustering of features
h <- hclust(dist(t(train_s)))

# Create dendrogram structure from h object
dend <- as.dendrogram(h) %>% set("branches_k_color", k = 7) %>%
  set("labels_cex", 0.5) %>%
  set("labels_colors", k = 7)

# Create ggdend object using dendextend package to visualise with the ggplot2 package
ggd <- as.ggdend(dend)

# Plot dendrogram from ggd object using ggplot2 package
ggplot(ggd, horiz = TRUE, theme = plot_theme) +  labs(x = "Features", y = "Distance")


### CORRELATION BETWEEN FEATURES

# Identifying correlated predictors
trainCor <- abs(cor(train_s))
cutoff <- 0.9
corr_index <- findCorrelation(trainCor, cutoff = cutoff, names = FALSE)
corr <- findCorrelation(trainCor, cutoff = cutoff, names = TRUE)

# Create heatmap of correlation between features
heatmap(as.matrix(trainCor), col = brewer.pal(9, "RdPu"), labCol = NA)

### PRINCIPAL COMPONENT ANALYSIS

# Save principal component analysis in pca object
pca <- prcomp(train_s)

# Calculate variance scores per principal component
pca.var <- pca$sdev^2
pca.var.per <- pca.var/sum(pca.var)

# Create table showing the first 10 Principal Components
summary(pca)$importance[,1:10]

# Create box plot to show top 10 PCs by diagnosis
data.frame(pca$x[,1:10], Diagnosis = train$y) %>%
    gather(key = "PC", value = "value", -Diagnosis) %>%
    ggplot(aes(PC, value, fill = Diagnosis)) +
    geom_boxplot() +
    scale_fill_discrete(name="Diagnosis",
                        breaks=c("B", "M"),
                        labels=c("Benign", "Malignant"))
  
# Create scatter plot of PC1 and PC2 by diagnosis
data.frame(pca$x[,1:2], Diagnosis = train$y) %>%
  ggplot(aes(PC1, PC2, color = Diagnosis)) +
  geom_point() +
  stat_ellipse() +
  xlab(paste("PC1: ", percent(pca.var.per[1],0.1))) +
  ylab(paste("PC2: ", percent(pca.var.per[2],0.1))) +
  scale_color_discrete(name="Diagnosis",
                       breaks=c("B", "M"),
                       labels=c("Benign", "Malignant"))

#######################################################################################
### METHODS
#######################################################################################

### NORMALISE TEST DATA

# Centre the test$x data around zero by subtracting the train$x column mean 
test_c <- sweep(test$x, 2, colMeans(train$x))

# Scale the test$x data by dividing by the train$x column standard deviation
test_s <- sweep(test_c, 2, colSds(train$x), FUN = "/")

# Create empty data frame to capture key performance measures in results table
model_results <- data.frame(Method = character(),
                            Accuracy = double(),
                            Sensitivity = double(),
                            Specificity = double(),
                            F1 = double(),
                            FNR = double(),
                            FPR = double())

# Define train control parameters for appropriate models
fitControl <- trainControl(method = "repeatedcv",
                           number = 10, # 10-fold cross-validation
                           repeats = 10, # repeat each cross-validation 10 times
                           classProbs = TRUE, # class probabilities computed
                           returnResamp = "final", # only save the final resampled summary metrics
                           savePredictions = "final") # only save the final predictions for each resample

### RANDOM SAMPLING MODEL

# Probability for random sampling set to 0.5, i.e. no weighting between different outcomes
p <- 0.5

# Set seed to enable result to be reproduced
set.seed(3, sample.kind = "Rounding")

# Sample outcomes matching length and factor levels from the test set
random_pred <- sample(c("B", "M"), length(test_index), prob = c(1-p, p), replace = TRUE) %>%
  factor(levels = levels(test$y))

# Store confusion matrix in 'random' object
random <- confusionMatrix(random_pred, test$y, positive = "M")

# Add model model results to model_results data frame
model_results[1, ] <- c("Random sample",
                        format(round(random$overall["Accuracy"],2), nsmall = 2),
                        format(round(random$byClass["Sensitivity"],2), nsmall = 2),
                        format(round(random$byClass["Specificity"],2), nsmall = 2),
                        format(round(random$byClass["F1"],2), nsmall = 2),
                        percent(1-(random$byClass["Sensitivity"])),
                        percent(1-(random$byClass["Specificity"])))

### WEIGHTED RANDOM SAMPLING MODEL

# Probability for random sampling set to match the prevalence of benign and malignant samples in the train set
p <- mean(train$y=="M")

# Set seed to enable result to be reproduced
set.seed(3, sample.kind = "Rounding")

# Sample outcomes matching length and factor levels from the test set
weighted_random_pred <- sample(c("B", "M"), length(test_index), prob = c(1-p, p), replace = TRUE) %>%
  factor(levels = levels(test$y))

# Store confusion matrix in 'weighted_random' object
weighted_random <- confusionMatrix(weighted_random_pred, test$y, positive = "M")

# Add model results to model_results data frame
model_results[2, ] <- c("Weighted random sample",
                        format(round(weighted_random$overall["Accuracy"],2), nsmall = 2),
                        format(round(weighted_random$byClass["Sensitivity"],2), nsmall = 2),
                        format(round(weighted_random$byClass["Specificity"],2), nsmall = 2),
                        format(round(weighted_random$byClass["F1"],2), nsmall = 2),
                        percent(1-(weighted_random$byClass["Sensitivity"])),
                        percent(1-(weighted_random$byClass["Specificity"])))

#######################################################################################
### UNSUPERVISED LEARNING (K-MEANS CLUSTERING)
#######################################################################################

# Build k-means prediction function
predict_kmeans <- function(x, k) {
  # extract cluster centres
  centres <- k$centers
  # calculate distance from data-points to the cluster centres
  distances <- sapply(1:nrow(x), function(i){
    apply(centres, 1, function(y) dist(rbind(x[i,], y)))
  })
  # select cluster with min distance to centre
  max.col(-t(distances))
}

### K-MEANS MODEL USING FULL DATA

# Define x as normalised train data-set, train_s
x <- train_s

# Set seed to enable result to be reproduced
set.seed(25, sample.kind = "Rounding")

# Predict outcome using kmeans model with k=2 and 25 random sets
k <- kmeans(x, centers = 2, nstart = 25)
kmeans_pred <- factor(ifelse(predict_kmeans(test_s, k) == 1, "M", "B"))

# Store confusion matrix in 'kmeans_results_1' object
kmeans_results_1 <- confusionMatrix(kmeans_pred, test$y, positive = "M")

# Add model results to model_results data frame
model_results[3, ] <- c("K-means clustering",
                        format(round(kmeans_results_1$overall["Accuracy"],2), nsmall = 2),
                        format(round(kmeans_results_1$byClass["Sensitivity"],2), nsmall = 2),
                        format(round(kmeans_results_1$byClass["Specificity"],2), nsmall = 2),
                        format(round(kmeans_results_1$byClass["F1"],2), nsmall = 2),
                        percent(1-(kmeans_results_1$byClass["Sensitivity"])),
                        percent(1-(kmeans_results_1$byClass["Specificity"])))

### K-MEANS MODEL USING SELECTED DATA

# K-means model with feature selection based on correlation coefficient score

#Define x_select as normalised train data-set, train_s extracting those included in the corr_index based on a correlation coefficient score over the cutoff (0.9)
x_select <- train_s[,-c(corr_index)]

#Apply the same selection process to the test data-set
test_s_select <- test_s[,-c(corr_index)]

# Set seed to enable result to be reproduced
set.seed(25, sample.kind = "Rounding")

# Predict outcome using kmeans model with k=2 and 25 random sets
k <- kmeans(x_select, centers = 2)
kmeans_pred_2 <- factor(ifelse(predict_kmeans(test_s_select, k) == 1, "M", "B"))

# Store confusion matrix in 'kmeans_results_2' object
kmeans_results_2 <- confusionMatrix(kmeans_pred_2, test$y, positive = "M")

# Add model results to model_results data frame
model_results[4, ] <- c("K-means (without highly correlated features)",
                        format(round(kmeans_results_2$overall["Accuracy"],2), nsmall = 2),
                        format(round(kmeans_results_2$byClass["Sensitivity"],2), nsmall = 2),
                        format(round(kmeans_results_2$byClass["Specificity"],2), nsmall = 2),
                        format(round(kmeans_results_2$byClass["F1"],2), nsmall = 2),
                        percent(1-(kmeans_results_2$byClass["Sensitivity"])),
                        percent(1-(kmeans_results_2$byClass["Specificity"])))

#######################################################################################
### SUPERVISED LEARNING - GENERARIVE MODELS
#######################################################################################

### NAIVE BAYES MODEL

# Set seed to enable result to be reproduced
set.seed(28, sample.kind = "Rounding")

# Use caret package to train and then predict outcomes using normalised train/test data-sets
train_nb <- train(train_s, train$y, 
                  method = "nb",
                  tuneGrid = expand.grid(usekernel = c(FALSE, TRUE), fL = c(0, 1), adjust = c(0, 1)),
                  trControl = fitControl)
nb_pred <- predict(train_nb, test_s)

# Store confusion matrix in 'nb_results' object
nb_results <- confusionMatrix(nb_pred, test$y, positive = "M")

# Add model results to model_results data frame
model_results[5, ] <- c("Naive Bayes",
                        format(round(nb_results$overall["Accuracy"],2), nsmall = 2),
                        format(round(nb_results$byClass["Sensitivity"],2), nsmall = 2),
                        format(round(nb_results$byClass["Specificity"],2), nsmall = 2),
                        format(round(nb_results$byClass["F1"],2), nsmall = 2),
                        percent(1-(nb_results$byClass["Sensitivity"])),
                        percent(1-(nb_results$byClass["Specificity"])))

### LINEAR DISCRIMINANT ANALYSIS (LDA) MODEL

# Set seed to enable result to be reproduced
set.seed(30, sample.kind = "Rounding")

# Use caret package to train and then predict outcomes using normalised train/test data-sets
train_lda <- train(train_s, train$y, 
                   method = "lda", 
                   trControl = fitControl)
lda_pred <- predict(train_lda, test_s)

# Store confusion matrix in 'lda_results' object
lda_results <- confusionMatrix(lda_pred, test$y, positive = "M")

# Add results to model_results data frame
model_results[6, ] <- c("Linear Discriminant Analysis",
                        format(round(lda_results$overall["Accuracy"],2), nsmall = 2),
                        format(round(lda_results$byClass["Sensitivity"],2), nsmall = 2),
                        format(round(lda_results$byClass["Specificity"],2), nsmall = 2),
                        format(round(lda_results$byClass["F1"],2), nsmall = 2),
                        percent(1-(lda_results$byClass["Sensitivity"])),
                        percent(1-(lda_results$byClass["Specificity"])))

### QUADRATIC DISCRIMINANT ANALYSIS (QDA) MODEL

# Set seed to enable result to be reproduced
set.seed(32, sample.kind = "Rounding")

# Use caret package to train and then predict outcomes using normalised train/test data-sets
train_qda <- train(train_s, train$y, 
                   method = "qda", 
                   trControl = fitControl)
qda_pred <- predict(train_qda, test_s)

# Store confusion matrix in 'qda_results' object
qda_results <- confusionMatrix(qda_pred, test$y, positive = "M")

# Add results to model_results data frame
model_results[7, ] <- c("Quadratic Discriminant Analysis",
                        format(round(qda_results$overall["Accuracy"],2), nsmall = 2),
                        format(round(qda_results$byClass["Sensitivity"],2), nsmall = 2),
                        format(round(qda_results$byClass["Specificity"],2), nsmall = 2),
                        format(round(qda_results$byClass["F1"],2), nsmall = 2),
                        percent(1-(qda_results$byClass["Sensitivity"])),
                        percent(1-(qda_results$byClass["Specificity"])))

### QUADRATIC DISCRIMINANT ANALYSIS (QDA) WITH PCA MODEL

# Set seed to enable result to be reproduced
set.seed(32, sample.kind = "Rounding")

# Use caret package to train and then predict outcomes using non-normalised train/test data-sets
train_qda_pca <- train(train$x, train$y,
                       method = "qda",
                       trControl = fitControl,
                       # Pre-processing function to centre, scale and apply pca
                       preProcess = c("center", "scale", "pca"))
qda_pca_pred <- predict(train_qda_pca, test$x)

# Store confusion matrix in 'qda_pca_results' object
qda_pca_results <- confusionMatrix(qda_pca_pred, test$y, positive = "M")

# Add results to model_results data frame
model_results[8, ] <- c("Quadratic Discriminant Analysis (with PCA)",
                        format(round(qda_pca_results$overall["Accuracy"],2), nsmall = 2),
                        format(round(qda_pca_results$byClass["Sensitivity"],2), nsmall = 2),
                        format(round(qda_pca_results$byClass["Specificity"],2), nsmall = 2),
                        format(round(qda_pca_results$byClass["F1"],2), nsmall = 2),
                        percent(1-(qda_pca_results$byClass["Sensitivity"])),
                        percent(1-(qda_pca_results$byClass["Specificity"])))

#######################################################################################
### SUPERVISED LEARNING - DISCRIMINATIVE MODELS
#######################################################################################

### LOGISTIC REGRESSION MODEL

# Set seed to enable result to be reproduced
set.seed(36, sample.kind = "Rounding")

# Use caret package to train and then predict outcomes using normalised train/test data-sets
train_glm <- train(train_s, train$y, 
                   method = "glm", 
                   trControl = fitControl)
glm_pred <- predict(train_glm, test_s)

# Store confusion matrix in 'glm_results' object
glm_results <- confusionMatrix(glm_pred, test$y, positive = "M")

# Add results to model_results data frame
model_results[9, ] <- c("Logistic regression",
                        format(round(glm_results$overall["Accuracy"],2), nsmall = 2),
                        format(round(glm_results$byClass["Sensitivity"],2), nsmall = 2),
                        format(round(glm_results$byClass["Specificity"],2), nsmall = 2),
                        format(round(glm_results$byClass["F1"],2), nsmall = 2),
                        percent(1-(glm_results$byClass["Sensitivity"])),
                        percent(1-(glm_results$byClass["Specificity"])))

### LOGISTIC REGRESSION WITH PCA MODEL

# Set seed to enable result to be reproduced
set.seed(36, sample.kind = "Rounding")

# Use caret package to train and then predict outcomes using non-normalised train/test data-sets
train_glm_pca <- train(train$x, train$y,
                       method = "glm",
                       trControl = fitControl,
                       # Pre-processing function to centre, scale and apply pca
                       preProcess = c("center", "scale", "pca"))
glm_pca_pred <- predict(train_glm_pca, test$x)

# Store confusion matrix in 'glm_pca_results' object
glm_pca_results <- confusionMatrix(glm_pca_pred, test$y, positive = "M")

# Add results to model_results data frame
model_results[10, ] <- c("Logistic regression (with PCA)",
                        format(round(glm_pca_results$overall["Accuracy"],2), nsmall = 2),
                        format(round(glm_pca_results$byClass["Sensitivity"],2), nsmall = 2),
                        format(round(glm_pca_results$byClass["Specificity"],2), nsmall = 2),
                        format(round(glm_pca_results$byClass["F1"],2), nsmall = 2),
                        percent(1-(glm_pca_results$byClass["Sensitivity"])),
                        percent(1-(glm_pca_results$byClass["Specificity"])))

### K-NEAREST NEIGHBOUR (KNN) MODEL

# Set seed to enable result to be reproduced
set.seed(5, sample.kind = "Rounding")

# Use caret package to train and then predict outcomes using normalised train/test data-sets
train_knn <- train(train_s, train$y,
                   method = "knn",
                   tuneGrid = data.frame(k = seq(1, 30, 2)),
                   trControl = fitControl)
knn_pred <- predict(train_knn, test_s)

# Store confusion matrix in 'knn_results' object
knn_results <- confusionMatrix(knn_pred, test$y, positive = "M")

# Add results to model_results data frame
model_results[11, ] <- c("K Nearest Neighbour",
                        format(round(knn_results$overall["Accuracy"],2), nsmall = 2),
                        format(round(knn_results$byClass["Sensitivity"],2), nsmall = 2),
                        format(round(knn_results$byClass["Specificity"],2), nsmall = 2),
                        format(round(knn_results$byClass["F1"],2), nsmall = 2),
                        percent(1-(knn_results$byClass["Sensitivity"])),
                        percent(1-(knn_results$byClass["Specificity"])))

### RANDOM FOREST (RF) MODEL

# Set seed to enable result to be reproduced
set.seed(7, sample.kind = "Rounding")

# Use caret package to train and then predict outcomes using normalised train/test data-sets
train_rf <- train(train_s, train$y,
                  method = "rf",
                  tuneGrid = data.frame(mtry = seq(3, 15, 2)),
                  importance = TRUE,
                  trControl = fitControl)
rf_pred <- predict(train_rf, test_s)

# Store confusion matrix in 'rf_results' object
rf_results <- confusionMatrix(rf_pred, test$y, positive = "M")

# Add results to model_results data frame
model_results[12, ] <- c("Random Forest",
                         format(round(rf_results$overall["Accuracy"],2), nsmall = 2),
                         format(round(rf_results$byClass["Sensitivity"],2), nsmall = 2),
                         format(round(rf_results$byClass["Specificity"],2), nsmall = 2),
                         format(round(rf_results$byClass["F1"],2), nsmall = 2),
                         percent(1-(rf_results$byClass["Sensitivity"])),
                         percent(1-(rf_results$byClass["Specificity"])))

### NEURAL NETWORK MODEL

# Set seed to enable result to be reproduced
set.seed(9, sample.kind = "Rounding")

# Use caret package to train and then predict outcomes using normalised train/test data-sets
train_nn <- train(train_s, train$y,
                  method = "nnet",
                  trace = FALSE,
                  trControl = fitControl)
nn_pred <- predict(train_nn, test_s)

# Store confusion matrix in 'nn_results' object
nn_results <- confusionMatrix(nn_pred, test$y, positive = "M")

# Add results to model_results data frame
model_results[13, ] <- c("Neural Network",
                        format(round(nn_results$overall["Accuracy"],2), nsmall = 2),
                        format(round(nn_results$byClass["Sensitivity"],2), nsmall = 2),
                        format(round(nn_results$byClass["Specificity"],2), nsmall = 2),
                        format(round(nn_results$byClass["F1"],2), nsmall = 2),
                        percent(1-(nn_results$byClass["Sensitivity"])),
                        percent(1-(nn_results$byClass["Specificity"])))

### ENSEMBLE MODEL

# Build ensemble model with all supervised models tested
ensemble <- cbind(glm = ifelse(glm_pred == "B", 0, 1), glm_pca = ifelse(glm_pca_pred == "B", 0, 1), nb = ifelse(nb_pred =="B", 0, 1), lda = ifelse(lda_pred == "B", 0, 1), qda = ifelse(qda_pred == "B", 0, 1), qda_pca = ifelse(qda_pca_pred == "B", 0, 1), rf = ifelse(rf_pred == "B", 0, 1), knn = ifelse(knn_pred == "B", 0, 1), nn = ifelse(nn_pred == "B", 0, 1))

# Predict final classification based on majority vote across all models, using the rowMeans function
ensemble_preds <- as.factor(ifelse(rowMeans(ensemble) < 0.5, "B", "M"))

# Store confusion matrix in 'ensemble_results' object
ensemble_results <- confusionMatrix(ensemble_preds, test$y, positive = "M")

# Add results to model_results data frame
model_results[14, ] <- c("Ensemble",
                        format(round(ensemble_results$overall["Accuracy"],2), nsmall = 2),
                        format(round(ensemble_results$byClass["Sensitivity"],2), nsmall = 2),
                        format(round(ensemble_results$byClass["Specificity"],2), nsmall = 2),
                        format(round(ensemble_results$byClass["F1"],2), nsmall = 2),
                        percent(1-(ensemble_results$byClass["Sensitivity"])),
                        percent(1-(ensemble_results$byClass["Specificity"])))

#######################################################################################
### RESULTS
#######################################################################################

### OVERALL PERFORMANCE MEASURES

# Print table of results from each model
model_results

### TUNING NAIVE BAYES MODEL

# Plot accuracy during cross-validation for tuning laplace smoothing (0/1), usekernel (true/false) and adjust (0/1)
plot(train_nb,
     ylab = "Accuracy (repeated cross-validation)",
     xlab = "Laplace correction")

### TUNING NEAREST NEIGHBOURS MODEL

# Plot accuracy during cross-validation for each value of k (number of neighbours) tuned
plot(train_knn,
     ylab = "Accuracy (repeated cross-validation)",
     xlab = "# Neighbours (k)")

### TUNING RANDOM FOREST MODEL

#Change number of digits to 4 to aid reading y-axis of plot
options(digits = 4)
# Plot accuracy during cross-validation for each value of mtry (number of neighbours) tuned
plot(train_rf,
     ylab = "Accuracy (repeated cross-validation)",
     xlab = "# Randomly selected predictors (mtry)")

### COMPARE MODEL PERFORMANCE DURING CROSS-VALIDATION

# Generate character string of models to be included in resampling analysis
modelNames <-  c("Naive Bayes", "Linear Discriminant Analysis", "Quadratic Discriminant Analysis", "Quadratic Discriminant Analysis with PCA", "Logistic Regression", "Logistic Regression with PCA", "K Nearest Neighbours", "Random Forest", "Neural Network")

# Generate list of results of cross-validation of training sets with each of these models
modelTrains <- list(train_nb, train_lda, train_qda, train_qda_pca, train_glm, train_glm_pca, train_knn, train_rf, train_nn)

# Use the resamples function to compare performance during cross-validation, ranking them in order of performance
model_compare <- resamples(modelTrains, modelNames = modelNames, decreasing = TRUE)

# Plot dot plots to show accuracy and kappa scores for each model
dotplot(model_compare)

### COMPARE VARIABLE IMPORTANCE FOR NEAREST NEIGHBOUR, RANDOM FOREST, NEURAL NETWORK AND LOGISITIC REGRESSION MODELS

# Create varImp objects for each of the models
varimp_knn <- (varImp(train_knn))
varimp_rf <- (varImp(train_rf))
varimp_nn <- (varImp(train_nn))
varimp_glm <- (varImp(train_glm))

# Define the number of variables to include in plots ranked by importance
top <- 10

# Generate variable importance plots for each of the varImp objects
plot(varimp_knn, top = top)
plot(varimp_rf, top = top) 
plot(varimp_nn, top = top) 
plot(varimp_glm, top = top)

