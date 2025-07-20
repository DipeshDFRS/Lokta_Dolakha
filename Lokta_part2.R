# Load required libraries ####
library(raster)            # Read raster data
library(sf)                # Spatial vector data handling
library(randomForest)      # Random forest modeling
library(caret)             # Machine learning
library(CAST)              # Spatial cross-validation
library(Metrics)           # Model evaluation metrics
library(e1071)             # Misc ML utilities
library(ExtractTrainData)  # Extract training data
library(readxl)            # Read Excel files
library(dplyr)             # Data manipulation
library(sp)                # Spatial objects
library(usdm)              # Multicollinearity analysis
library(DescTools)         # Descriptive statistics
library(NeuralNetTools)    # Visualization for ANN
library(xgboost)           # Extreme gradient boosting
library(caTools)           # Data splitting
library(broom)             # Tidy model summaries
library(visreg)            # Visualize regression
library(margins)           # Marginal effects
library(rcompanion)        # Pseudo R²
library(ROCR)              # ROC curves
library(pROC)              # ROC curves and AUC
library(yardstick)         # Accuracy metrics
library(GGally)            # Correlation plots
library(performance)       # Model performance checks
library(ggpmisc)           # ggplot enhancements
library(car)               # Companion to Applied Regression
library(MuMIn)             # AIC model selection
library(tidyverse)         # Data science tools
library(rms)               # Regression modeling
library(dismo)             # Species distribution modeling
library(ellipse)           # Confidence ellipses
library(rJava)             # Java integration
library(XML)               # XML parsing
library(RColorBrewer)      # Color palettes

#Set working directory####
setwd("D:/dolakha")

#Load study area boundary####
dkh_boundary<-st_read("predictors/Dolkha_buffer_100m.shp")
dkh_boundary_crs <- st_transform(dkh_boundary, crs = wgs1984.proj )

# Read the shapefile####
db <- st_read('Data/Lokta_all_pred.shp')

# Convert sf object to a regular data frame
db1 <- as.data.frame(db)

## Rename columns and select specific columns####
db2 <- db1 %>%
  rename(
    Elevation = elevatn,
    Slope = slope,
    Aspect = aspect,
    NDVI = ndv_lkt,
    Nearest_distance_waterbody = NEAR_RI,
    Nearest_distance_forest = NEAR_FO,
    Silt = slt515_,
    Sand = snd515_,
    Clay = cly515_,
    pH = ph515_l,
    Carbon = SOC515_,
    Nitrogen = Ntr515_,
    nitrogen2 = Nitrogn,
    sand2 = Sand,
    clay2 = Clay,
    Silt2 = Slit,
    pH2 = pH
  ) 

# Check the dimensions of db
print(dim(db2))
db3 <- na.omit(db2)

# Load the data with p/a points and rename####
occ_points<- read_excel("Data/lokta_points_3.26.xlsx") ## Load lokta_points_3.26.xlsx ####

occ_points <- occ_points%>% 
  rename(Elevation = srtm_dolakha1,Slope =slope,Aspect = aspect,
            BIO1 = layer.1,BIO2 = layer.2, BIO3 = layer.3,BIO4 = layer.4,
             BIO5 = layer.5,BIO6 = layer.6,BIO7 = layer.7,BIO8 = layer.8,
             BIO9 = layer.9,BIO10 = layer.10,BIO11 = layer.11,
             BIO12 = layer.12,BIO13 = layer.13,BIO14 = layer.14,
             BIO15 = layer.15,BIO16 = layer.16,BIO17 = layer.17,
             BIO18 = layer.18,BIO19 = layer.19,
             NDVI =ndvi_lokta,MSAVI2=msavi2_lokta,Silt=silt515_lokta,
             Sand=sand515_lokta, Clay= clay515_lokta,
             pH= ph515_lokta,Nitrogen =nitrogen515_lokta,Carbon =SOC515_lokta,
             Waterbody.dist =NEAR_DIST_Waterbody,Forest.dist =NEAR_DIST_forest)

var_all <- occ_points [, c(2:34)]

## Add training data###

var_all$p_a<-as.factor(var_all$p_a)
table(var_all$p_a)

# Generation of training (70%) and test (30%) data using training data points#
set.seed(190)
p_a<-sample(2,nrow(var_all),replace=TRUE,prob=c(0.7,0.3))
train<-var_all[p_a==1,]
test<-var_all[p_a==2,]

train
test

paste("train sample size: ", dim(train)[1])
paste("test sample size: ", dim(test)[1])



# RandomForest model ####
## Define the tuning grid for RF####
CV <- trainControl(method = "cv", number = 5, savePredictions = TRUE)
rfGrid <-expand.grid(mtry = (1:10))

rf <- train(p_a~.,data=train, 
            method = "rf", 
            trcontrol = cv,
            verbose = FALSE,
            tuneGrid = rfGrid,
            importance = TRUE)

#save(rf, file = "Model/rf.RData") # to save the model
#load("Model/rf.RData") # to load the model

rf 

#windows()
plot(rf)

varImp_rf <- varImp(rf, scale = TRUE)
varImp_rf
plot(varImp_rf)

# Convert to data frame
vImp_rf <- data.frame(Variable = rownames(varImp_rf$importance),
                        Importance = varImp_rf$importance[, 1])

## Select the top 10 variables####
top10_rf<- vImp_rf %>%
  arrange(desc(Importance)) %>%
  slice_head(n = 10)

## Plot variable importance####
plot_var_rf <-ggplot(top10_rf, aes(x = Importance, y = reorder(Variable, Importance))) +
  geom_segment(aes(x = 0, xend = Importance, yend = Variable), color = "black") +  # Horizontal lines
  geom_point(color = "steelblue", size = 2) +  # Blue circles at the end
  geom_vline(xintercept = 0, linetype = "solid", color = "black") +  # Vertical line at 0
  labs(x = "Importance", y = "") +  # Keep x-axis label but remove y-axis label
  scale_x_continuous(position = "bottom") +
  theme_classic(base_size = 14) +  # Classic theme keeps axis lines but removes background
  theme(axis.text.x = element_text(size = 14, color = "black"),
        axis.text.x.top = element_blank(),
        axis.text.y = element_text(size = 14, color = "black"),  # Keep y-axis labels
        axis.ticks.y = element_blank(),  # Remove y-axis tick marks
        axis.ticks.x = element_line(color = "black"),
        axis.ticks.x.top = element_line(color = "black"),  # Add tick marks at the top
        axis.line = element_line(color = "black"),  # Ensures left and bottom axis lines
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.7))  # Adds top and right border
plot(plot_var_rf)

## Accuracy assessment w.r.t test data point####
pred_rf <-predict(rf, test)
glimpse(test)
confusionMatrix(pred_rf, reference = test$p_a)

# Predict probabilities for the test dataset
pred_rf_prob <- predict(rf, newdata = test, type = "prob")[,2]

## AUC and ROC using pROC####
roc_rf <- roc(test$p_a, pred_rf_prob)
auc_rf <- auc(roc_rf)

plot(roc_rf, main = paste("AUC:", round(auc_rf, 4)))
abline(a = 0, b = 1)

# Creating an actual/observed vs predicted dataframe
act_pred_rf <- data.frame(observed = test$p_a, predicted = factor(pred_rf))

## Calculating precision, recall, and F1 score####
prec_rf <- precision(act_pred_rf, observed, predicted)
rec_rf <- recall(act_pred_rf, observed, predicted)
F1_rf <- f_meas(act_pred_rf, observed, predicted) # called f_measure

print(prec_rf)
print(rec_rf)
print(F1_rf)

##Map prediction####

pred_data_rf = cbind(db3 [c('x','y')], predict(rf, db3 , type = "prob"))
names(pred_data_rf) = c("x", "y", "no", "yes")

xyrf = SpatialPointsDataFrame(as.data.frame(pred_data_rf)[, c("x", "y")], data = pred_data_rf)
rast_rf = rasterFromXYZ(as.data.frame(xyrf)[, c("x", "y", "yes")])
proj4string(rast_rf)= wgs1984.proj

#windows()
plot(rast_rf)

writeRaster(rast_rf, 'Output/rast_rf_3.26.tif', overwrite= TRUE)

##Create color palette and plot####
palfunc <- function (n, alpha = 1, begin = 0, end = 1, direction = 1) 
{
  colors <- rev(brewer.pal(11, "RdYlGn"))
  if (direction > 0) colors <- rev(colors)
  colorRampPalette(colors, alpha = alpha)(n)
}

spplot(rast_rf, main="Daphne bholua Distribution Mapping using RF",col.regions=palfunc)

jpeg("Output/Lokta_RF_11.9.jpg", width = 800, height = 500)

dev.off()



# KNN  ####
## Define the tuning grid for KNN####
tune_grid <- expand.grid(k = c(3, 5, 7, 9)) 
trnctrl_knn<- trainControl(method = "cv", number = 10)

set.seed(223)
knn <- train(p_a ~., data = train, method = "knn", 
                       trControl = trnctrl_knn, 
                       preProc = c("center", "scale"),
                       tuneLength = 10)

#save(knn, file = 'Model/knn.RData') # to save the model
#load(file = 'Model/knn.RData') # to load the model
#windows()
knn
plot(knn)

# Original model accuracy
original_accuracy <- max(knn$results$Accuracy)

# Function to compute variable importance
compute_var_knn <- function(model, test_data, target_var) {
  importance <- c()
  
  for (var in colnames(test_data)[colnames(test_data) != target_var]) {
    set.seed(42)  # Ensures the same shuffle each time
    permuted_data <- test_data
    permuted_data[[var]] <- sample(permuted_data[[var]])  # Shuffle the variable
    
    predictions <- predict(model, permuted_data)
    permuted_accuracy <- mean(predictions == test_data[[target_var]])
    
    importance[var] <- original_accuracy - permuted_accuracy
  }
  
  return(importance)
}

## Compute variable importance####
var_knn <- compute_var_knn(
  model = knn,
  test_data = test,  # Replace with your test dataset
  target_var = "p_a"
)

# Convert to data frame for ggplot2
vImp_knn <- data.frame(Variable = names(var_knn),
                            Importance = var_knn)

# Normalize importance values before selecting the top 10
vImp_knn$Scaled_Importance <- 100 * (vImp_knn$Importance - min(vImp_knn$Importance, na.rm = TRUE)) / 
  (max(vImp_knn$Importance, na.rm = TRUE) - min(vImp_knn$Importance, na.rm = TRUE))

## Select the top 10 most important variables####
top10_knn <- vImp_knn %>%
  arrange(desc(Scaled_Importance)) %>%
  slice_head(n = 10)

## Plot variable importance####
plot_var_knn <- ggplot(top10_knn, aes(x = Scaled_Importance, y = reorder(Variable, Scaled_Importance))) +
  geom_segment(aes(x = 0, xend = Scaled_Importance, yend = Variable), color = "black") +  # Horizontal lines
  geom_point(color = "steelblue", size = 2) +  # Blue circles at the end
  geom_vline(xintercept = 0, linetype = "solid", color = "black") +  # Vertical line at 0
  labs(x = "Importance", y = "") +  # Keep x-axis label but remove y-axis label
  scale_x_continuous(position = "bottom") +
  theme_classic(base_size = 14) +  # Classic theme keeps axis lines but removes background
  theme(axis.text.x = element_text(size = 14, color = "black"),
        axis.text.x.top = element_blank(),
        axis.text.y = element_text(size = 14, color = "black"),  # Keep y-axis labels
        axis.ticks.y = element_blank(),  # Remove y-axis tick marks
        axis.ticks.x = element_line(color = "black"),
        axis.ticks.x.top = element_line(color = "black"),  # Add tick marks at the top
        axis.line = element_line(color = "black"),  # Ensures left and bottom axis lines
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.7))  # Adds top and right border

plot(plot_var_knn)

# varImp_knn <- varImp(knn, scale = TRUE)

##Accuracy assessment w.r.t. test data point ####
pred_knn <- predict(knn, test)
confusionMatrix(pred_knn, reference = test$p_a)

# Predict probabilities for the test dataset
pred_knn_prob <- predict(knn, newdata = test, type = "prob")[,2]

## AUC and ROC using pROC####
roc_knn <- roc(test$p_a, pred_knn_prob )
auc_knn <- auc(roc_knn)

plot(roc_knn, main = paste("AUC:", round(auc_knn, 4)))
abline(a = 0, b = 1)

# Creating an actual/observed vs predicted dataframe
act_pred_knn <- data.frame(observed = test$p_a, predicted = factor(pred_knn))

## Calculating precision, recall, and F1 score####
prec_knn <- precision(act_pred_knn, observed, predicted)
rec_knn <- recall(act_pred_knn, observed, predicted)
F1_knn <- f_meas(act_pred_knn, observed, predicted) # called f_measure 

print(prec_knn)
print(rec_knn)
print(F1_knn)


##Map prediction####

pred_data_knn = cbind(db3[c('x','y')], predict(knn,db3, type='prob'))
names(pred_data_knn) = c("x", "y", "no", "yes")

xy_knn = SpatialPointsDataFrame(as.data.frame(pred_data_knn)[, c("x", "y")], data = pred_data_knn)
rast_knn = rasterFromXYZ(as.data.frame(xy_knn)[, c("x", "y", "yes")])
proj4string(rast_knn)=wgs1984.proj

plot(rast_knn)
writeRaster(rast_knn, 'Output/rast_knn_3.26.tif', overwrite= TRUE)

##Create color palette and plot####
palfunc <- function (n, alpha = 1, begin = 0, end = 1, direction = 1) 
{
  colors <- rev(brewer.pal(11, "RdYlGn"))
  if (direction > 0) colors <- rev(colors)
  colorRampPalette(colors, alpha = alpha)(n)
}

spplot(rast_knn , main="Daphne bholua distribution mapping using KNN",col.regions=palfunc)

jpeg("Output/lokta_knn.jpg", width = 800, height = 500)


# ANN  ####

# Set seed for reproducibility
set.seed(24)

## Define the tuning grid for ANN####
tune_grid_ann <- expand.grid(size = c(1, 3, 5), decay = c(0, 0.1, 0.5))
grids <-  expand.grid(size = seq(from = 1, to = 7, by = 2),
                      decay = seq(from = 0, to = 0.1,by = 0.01))
ctrl <- trainControl(method = "repeatedcv",
                     number = 5, # 5 folds
                     repeats = 3, # 3 repeats
                     search = "grid") # grid search

set.seed(831)
ann <- train(form = p_a~., # use all other variables to predict target
                data = train, # training data
                preProcess = "range", # apply min-max normalization
                method = "nnet", # use nnet()
                trControl = ctrl, 
                tuneGrid = grids, # search over the created grid
                trace = FALSE) # suppress output

#save(ann, file = 'Model/ann.RData') #to save the model
#load(file = 'Model/ann.RData') # to load the model

ann
plot(ann)

varImp_ann<- varImp(ann, scale = TRUE)
varImp_ann
plot(varImp_ann)


# Convert to data frame
vImp_ann<- data.frame(Variable = rownames(varImp_ann$importance),
                        Importance = varImp_ann$importance[, 1])

## Select the top 10 variables####
top10_ann <- vImp_ann %>%
  arrange(desc(Importance)) %>%
  slice_head(n = 10)

## Plot variable importance####
plot_var_ann <-ggplot(top10_ann, aes(x = Importance, y = reorder(Variable, Importance))) +
  geom_segment(aes(x = 0, xend = Importance, yend = Variable), color = "black") +  # Horizontal lines
  geom_point(color = "steelblue", size = 2) +  # Blue circles at the end
  geom_vline(xintercept = 0, linetype = "solid", color = "black") +  # Vertical line at 0
  labs(x = "Importance", y = "") +  # Keep x-axis label but remove y-axis label
  scale_x_continuous(position = "bottom") +
  theme_classic(base_size = 14) +  # Classic theme keeps axis lines but removes background
  theme(axis.text.x = element_text(size = 14, color = "black"),
        axis.text.x.top = element_blank(),
        axis.text.y = element_text(size = 14, color = "black"),  # Keep y-axis labels
        axis.ticks.y = element_blank(),  # Remove y-axis tick marks
        axis.ticks.x = element_line(color = "black"),
        axis.ticks.x.top = element_line(color = "black"),  # Add tick marks at the top
        axis.line = element_line(color = "black"),  # Ensures left and bottom axis lines
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.7))  # Adds top and right border


plot(plot_var_ann)

plotnet(mod_in = ann$finalModel, # nnet object
        pos_col = "darkgreen", # positive weights are shown in green
        neg_col = "darkred", # negative weights are shown in red
        bias = FALSE, # do not plot bias
        circle_cex = 4, # reduce circle size (default is 5)
        cex_val = 0.6) # reduce text label size (default is 1)

## Accuracy assessment w.r.t test data point####
pred_ann <- predict(ann, test)
confusionMatrix(pred_ann, reference = test$p_a)

# Predict probabilities for the test dataset
pred_ann_prob <- predict(ann, newdata = test, type = "prob")[,2]
confusionMatrix(data = pred_ann_prob, # predictions
                reference = train$p_a, # actual
                positive = "1",
                mode = "everything")

## AUC and ROC using pROC####
roc_ann <- roc(test$p_a, pred_ann_prob )
auc_ann <- auc(roc_ann)

plot(roc_ann, main = paste("AUC:", round(auc_ann, 4)))
abline(a = 0, b = 1)

# Creating an actual/observed vs predicted dataframe
act_pred_ann <- data.frame(observed = test$p_a, predicted = factor(pred_ann))

# Calculating precision, recall, and F1 score
prec_ann <- precision(act_pred_ann, observed, predicted)
rec_ann <- recall(act_pred_ann, observed, predicted)
F1_ann <- f_meas(act_pred_ann, observed, predicted) # called f_measure

print(prec_ann)
print(rec_ann)
print(F1_ann)

##Map prediction####
pred_data_ann = cbind(db3[c('x','y')], predict(ann, db3, type = "prob"))
names(pred_data_ann) = c("x", "y", "no", "yes")

xy_ann = SpatialPointsDataFrame(as.data.frame(pred_data_ann)[, c("x", "y")], data = pred_data_ann)
rast_ann = rasterFromXYZ(as.data.frame(xy_ann)[, c("x", "y", "yes")])
proj4string(rast_ann)=wgs1984.proj

#windows()
plot(rast_ann)
writeRaster(rast_ann, 'Output/ann_3.26.tif')

##Create color palette and plot####
palfunc <- function (n, alpha = 1, begin = 0, end = 1, direction = 1) 
{
  colors <- rev(brewer.pal(11, "RdYlGn"))
  if (direction > 0) colors <- rev(colors)
  colorRampPalette(colors, alpha = alpha)(n)
}

spplot(rast_ann , main="Daphne bholua distribution mapping using ANN",col.regions=palfunc)

jpeg("Output/lokta_ann.jpg", width = 800, height = 500)



#XGB model####
## Define the tuning grid for XGB####
grid_default <- expand.grid(nrounds = 100,max_depth = 6,eta = 0.3,gamma = 0,
                            colsample_bytree = 1,min_child_weight = 1,subsample = 1)

train_control <- trainControl(method = "cv", number = 10)

set.seed(34)
xgb <- train(p_a~.,data=train,
             method = "xgbTree", 
             trcontrol = train_control,
             verbose = FALSE,
             tuneGrid = grid_default)
 
#save(xgb, file = "Model/xgb.RData") #to save the model
#load("Model/xgb.RData") #to load the model
xgb
plot(xgb)

varImp_xgb <- varImp(xgb, scale = TRUE)
varImp_xgb
#windows()
plot(varImp_xgb)


# Convert to data frame
vImp_xgb <- data.frame(Variable = rownames(varImp_xgb$importance),
                        Importance = varImp_xgb$importance[, 1])

## Select the top 10 variables####
top10_xgb <- vImp_xgb %>%
  arrange(desc(Importance)) %>%
  slice_head(n = 10)

# Plot variable importance
plot_var_xgb <- ggplot(top10_xgb, aes(x = Importance, y = reorder(Variable, Importance))) +
  geom_segment(aes(x = 0, xend = Importance, yend = Variable), color = "black") +  # Horizontal lines
  geom_point(color = "steelblue", size = 2) +  # Blue circles at the end
  geom_vline(xintercept = 0, linetype = "solid", color = "black") +  # Vertical line at 0
  labs(x = "Importance", y = "") +  # Keep x-axis label but remove y-axis label
  scale_x_continuous(position = "bottom") +
  theme_classic(base_size = 14) +  # Classic theme keeps axis lines but removes background
  theme(axis.text.x = element_text(size = 14, color = "black"),
        axis.text.x.top = element_blank(),
        axis.text.y = element_text(size = 14, color = "black"),  # Keep y-axis labels
        axis.ticks.y = element_blank(),  # Remove y-axis tick marks
        axis.ticks.x = element_line(color = "black"),
        axis.ticks.x.top = element_line(color = "black"),  # Add tick marks at the top
        axis.line = element_line(color = "black"),  # Ensures left and bottom axis lines
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.7))  # Adds top and right border

plot(plot_var_xgb)

## Accuracy assessment w.r.t test data point####
pred_xgb<- predict(xgb, test)
confusionMatrix(predict_xgb, reference = test$p_a)

# Predict probabilities for the test dataset
pred_xgb_prob <- predict(xgb, newdata = test, type = "prob")[,2]

## AUC and ROC using pROC####
roc_xgb <- roc(test$p_a, pred_xgb_prob )
auc_xgb <- auc(roc_xgb)

plot(roc_xgb, main = paste("AUC:", round(auc_xgb, 4)))
abline(a = 0, b = 1)

# Creating an actual/observed vs predicted dataframe
act_pred_xgb <- data.frame(observed = test$p_a, predicted = factor(pred_xgb))

# Calculating precision, recall, and F1 score
prec_xgb <- precision(act_pred_xgb, observed, predicted)
rec_xgb <- recall(act_pred_xgb, observed, predicted)
F1_xgb <- f_meas(act_pred_xgb, observed, predicted) # called f_measure

print(prec_xgb)
print(rec_xgb)
print(F1_xgb)


##Map prediction####
pred_data_xgb = cbind(db3[c('x','y')], predict(xgb, db3, type = "prob"))
names(pred_data_xgb) = c("x", "y", "no", "yes")

xy_xgb = SpatialPointsDataFrame(as.data.frame(pred_data_xgb)[, c("x", "y")], data = pred_data_xgb)
rast_xgb = rasterFromXYZ(as.data.frame(xy_xgb)[, c("x", "y", "yes")])
proj4string(rast_xgb)=wgs1984.proj
#windows()
plot(rast_xgb)
writeRaster(rast_xgb, 'Output/xgb_3.26.tif')

##Create color palette and plot####
palfunc <- function (n, alpha = 1, begin = 0, end = 1, direction = 1) 
{
  colors <- rev(brewer.pal(11, "RdYlGn"))
  if (direction > 0) colors <- rev(colors)
  colorRampPalette(colors, alpha = alpha)(n)
}
spplot(rast_xgb, main="Daphne bholua distribution mapping using XGB", col.regions=palfunc)

jpeg("Output/lokta_xgb.jpg", width = 800, height = 500)


#VarImp plot merged####
library(gridExtra)
# Arrange in 2x2 grid
grid.arrange(plot_var_rf, plot_var_knn, plot_var_ann,plot_var_xgb, ncol=2, nrow=2)
plot(plot_var_xgb)

#ROC combined####
# Create a data frame for each ROC curve
roc_data_rf <- data.frame(fpr = roc_rf$specificities, tpr = roc_rf$sensitivities, model = "RF")
roc_data_knn <- data.frame(fpr = roc_knn$specificities, tpr = roc_knn$sensitivities, model = "KNN")
roc_data_ann <- data.frame(fpr = roc_ann$specificities, tpr = roc_ann$sensitivities, model = "ANN")
roc_data_xgb <- data.frame(fpr = roc_xgb$specificities, tpr = roc_xgb$sensitivities, model = "XGB")

# Combine all ROC data frames into one
roc_combined <- rbind(roc_data_rf, roc_data_knn, roc_data_ann, roc_data_xgb)#, roc_data_lr)

# Plot ROC curves with ggplot2
ggplot(roc_combined, aes(x = 1 - fpr, y = tpr, color = model)) +
  geom_line() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(title = "ROC Curves for Models", x = "1 - Specificity", y = "Sensitivity") +
  theme_classic() +
  theme(
    legend.position = c(0.90, 0.15), # Position the legend at the bottom right
    legend.title = element_text(size = 12), # Adjust legend title size
    legend.text = element_text(size = 10), # Adjust legend text size
    legend.box.spacing = unit(0.5, "cm") # Adjust spacing between legend items
  ) +
  scale_color_manual(values = c("RF" = "blue", "KNN" = "darkorange", "ANN" = "magenta", 
                                "XGB" = "forestgreen"))#, "LR" = "forestgreen"))


#End of part 2####
