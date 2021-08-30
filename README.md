# PyTorch Implementation of Metric Learning with Supervision and Adaptive Sample Pair Formation (MeL-S-ASPF)
## Description
- feature: it includes the preprocessed feature for the four datasets (iemocap, enterface, crema-d and ravdess).  
- SNN_source_models: four pre-trained SNN models using four source datasets are included.  
- dataset_SNN.py: builds a dataloader for the Siamese model training. The likelihood is maintained through a dictionary (prob_dic) and is updated using the 'update_prob' function. It generates a pair of data (two samples) and a lable 0 or 1 indicating this pair contains the same or different emotions.   
- dataset_classification.py: builds a dataloader for the fine-tuning as well as the supervised learning process. It generates single data points with a emotion class label.  
- model_Siamese.py: it includes the model structure for the Siamese model. The output layer is 1 unit with sigmoid activation.  
- model_classification.py: it includes the model structure for the supervised model. The output layer is 3 units with softmax.  

## Results
Two sample running results are provided in the form of jupyter notebook.  
- train_MeL_S_ASPF.ipynb: a sample running result for the method Mel-S-ASPF.  
- train_MeL_S.ipynb: a sample running result for the method Mel-S. The code is almost the same as Mel-S-ASPF, except the calling of the 'update_prob' function is commented out, to avoid any likelihood update process.  

## References
will be added soon
