# model-generalization

# feature_extraction
Contains a wrapper script for existing code to extract features for a given dataset + model/models

## Prerequisite

1) Install brain-score candidate models: https://github.com/brain-score/candidate_models
2) Set brain-score environment variables in extract_features.py
3) Change your TF_slim path in extract_features.sh (tf_slim_path variable)


## To extract features for all models on brain-score/pytorch and save as .h5:

Run 

```
 ./extract_all_features.sh stims_path save_path
```

e.g.

```
 ./extract_all_features.sh "/braintree/data2/active/users/ratan/projects/gen/stims/downing/" "/om2/group/nklab/ighodgao/extracted_features/downing/"
```
 
 This will submit 1 job per model on openmind and save extracted features in the form of h5 files in the intended dir.
 
 
 ## To extract features for a given model
 
 Run 
 
 ```
 python extract_features.py --images_path --save_dir --model 
 ```

where model is the model name
 
