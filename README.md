# AutoML vs Hyperparameter optimization: A case study in software defect prediction
## Take home: 
1. HPT xgboost always achieve the best performance. However, for the small dataset (i.e., camel 1.2), it takes a longer time to achieve 99% close-to-the-highest performance.
2. AML and HPT can both achieve 95% close-to-the-highest AUC_weighted within 13 iterations, So we don’t have to hyper-parameter tune/deploy AML for a lot of iterations.
3. Possible advice for the practitioner would be to run the AML for 20 iterations and Hyperparameter tune the suggested model for the best performance.
4. The interpretation varies between the models tuned by AML and HPT. Therefore, we recommend using the interpretation of the model with the best fit. 
