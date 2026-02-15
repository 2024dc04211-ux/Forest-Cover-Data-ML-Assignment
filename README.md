Machine Learning Classification Models – Streamlit Deployment


a)  Problem Statement

The objective of this project is to implement multiple machine learning classification models on a chosen dataset, evaluate their performance using standard metrics, and deploy an interactive web application using Streamlit to demonstrate predictions and model comparisons.
Classification of pixels into 7 forest cover types based on attributes such as elevation, aspect, slope, hillshade, soil-type, and more.


b) Dataset Description

Dataset Name: Forest Cover Data

Source: Kaggle - https://www.kaggle.com/c/forest-cover-type-prediction/data
		UCI - https://archive.ics.uci.edu/dataset/31/covertype

Type: Multi-class classification

Additional Information

Predicting forest cover type from cartographic variables only (no remotely sensed data).  The actual forest cover type for a given observation (30 x 30 meter cell) was determined from US Forest Service (USFS) Region 2 Resource Information System (RIS) data.  Independent variables were derived from data originally obtained from US Geological Survey (USGS) and USFS data.  Data is in raw form (not scaled) and contains binary (0 or 1) columns of data for qualitative independent variables (wilderness areas and soil types).

This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado.  These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices.

Some background information for these four wilderness areas: Neota (area 2) probably has the highest mean elevational value of the 4 wilderness areas. Rawah (area 1) and Comanche Peak (area 3) would have a lower mean elevational value, while Cache la Poudre (area 4) would have the lowest mean elevational value. 

As for primary major tree species in these areas, Neota would have spruce/fir (type 1), while Rawah and Comanche Peak would probably have lodgepole pine (type 2) as their primary species, followed by spruce/fir and aspen (type 5). Cache la Poudre would tend to have Ponderosa pine (type 3), Douglas-fir (type 6), and cottonwood/willow (type 4).  

The Rawah and Comanche Peak areas would tend to be more typical of the overall dataset than either the Neota or Cache la Poudre, due to their assortment of tree species and range of predictive variable values (elevation, etc.)  Cache la Poudre would probably  be more unique than the others, due to its relatively low  elevation range and species composition. 

Has Missing Values?  No


Number of Instances: 581012

Number of Features: 54

Target Variable: <Class label descriptiion>
## Class Labels

| Key | Class Name |
|-----|-----------|
| 1 | Spruce / Fir |
| 2 | Lodgepole Pine |
| 3 | Ponderosa Pine |
| 4 | Cottonwood / Willow |
| 5 | Aspen |
| 6 | Douglas-fir |
| 7 | Krummholz |

Model Performance Comparison (500000 samples)
| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------|---------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.736776 | 0.939121 | 0.722554 | 0.736776 | 0.725614 | 0.562635 |
| Decision Tree | 0.935992 | 0.939835 | 0.935925 | 0.935992 | 0.935951 | 0.896063 |
| KNN | 0.923952 | 0.982299 | 0.923686 | 0.923952 | 0.923734 | 0.876285 |
| Naive Bayes | 0.485264 | 0.887366 | 0.635980 | 0.485264 | 0.466807 | 0.321037 |
| Random Forest | 0.950624 | 0.997682 | 0.950764 | 0.950624 | 0.950359 | 0.919666 |
| XGBoost | 0.818408 | 0.974563 | 0.819222 | 0.818408 | 0.814498 | 0.700791 |

Model Performance Comparison (200000 samples)

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------|---------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.77218 | 0.943294 | 0.762576 | 0.77218 | 0.761355 | 0.506063 |
| Decision Tree | 0.93698 | 0.923061 | 0.936928 | 0.93698 | 0.936947 | 0.870354 |
| KNN | 0.91338 | 0.975828 | 0.912750 | 0.91338 | 0.912933 | 0.820792 |
| Naive Bayes | 0.70602 | 0.912880 | 0.690206 | 0.70602 | 0.664993 | 0.350269 |
| Random Forest | 0.94418 | 0.996392 | 0.944065 | 0.94418 | 0.943557 | 0.884138 |
| XGBoost | 0.87522 | 0.984504 | 0.873432 | 0.87522 | 0.872676 | 0.737944 |

Model Performance Comparison (100000 samples)
| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------|---------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.82228 | 0.956632 | 0.816041 | 0.82228 | 0.809688 | 0.628917 |
| Decision Tree | 0.93452 | 0.928243 | 0.934533 | 0.93452 | 0.934521 | 0.869980 |
| KNN | 0.90956 | 0.975105 | 0.908612 | 0.90956 | 0.908731 | 0.818798 |
| Naive Bayes | 0.74008 | 0.931426 | 0.719452 | 0.74008 | 0.692378 | 0.445634 |
| Random Forest | 0.94412 | 0.996113 | 0.943950 | 0.94412 | 0.943449 | 0.887926 |
| XGBoost | 0.90212 | 0.990711 | 0.901417 | 0.90212 | 0.899543 | 0.802173 |


Observations on Model Performance

The six classification models were evaluated on the Forest Cover Type dataset using standard performance metrics such as Accuracy, AUC, Precision, Recall, F1 Score, and MCC. This dataset contains complex relationships between geographical and soil-related features, which makes it useful for comparing different types of machine learning approaches. Some models are better at capturing linear patterns, while others can learn more complex non-linear structures. The observations below describe how each model performed on this multi-class classification task in terms of prediction quality and overall suitability for the dataset.

| ML Model | Observation about model performance |
|---------|------------------------------------|
| Logistic Regression | Shows moderate performance across all dataset sizes. Works well for linear decision boundaries but struggles to capture complex non-linear relationships present in the forest cover dataset. Performance improves with larger data but remains below tree-based ensembles. |
| Decision Tree | Achieves high accuracy and MCC, indicating strong ability to model non-linear feature interactions. However, slight drop in AUC compared to ensembles suggests some overfitting and less generalization than Random Forest. |
| kNN | Provides consistently high AUC and good accuracy, showing effective local neighborhood classification in feature space. Performance slightly decreases with larger datasets due to computational complexity and sensitivity to feature scaling. |
| Naive Bayes | Lowest performance among all models, especially on large dataset, due to strong independence assumption between features which is violated in this dataset where terrain attributes are correlated. |
| Random Forest (Ensemble) | Best overall performer across all dataset sizes with highest accuracy, AUC, F1 and MCC. Ensemble averaging reduces overfitting and captures complex feature interactions effectively, making it most suitable for this dataset. |
| XGBoost (Ensemble) | Strong performance with very high AUC and good MCC, outperforming single models but slightly below Random Forest in accuracy on largest dataset. Boosting improves classification of difficult classes and generalization. |
