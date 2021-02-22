# model-generalization

# To extract features for all models on brain-score/pytorch and save as .h5:

1) Install brain-score candidate models: https://github.com/brain-score/candidate_models

2) Run 

'''
 ./extract_all_features.sh stims_path save_path
'''

e.g.
 '''
 ./extract_all_features.sh "/braintree/data2/active/users/ratan/projects/gen/stims/downing/" "/om2/group/nklab/ighodgao/extracted_features/downing/"
 '''
 
 This will submit 1 job per model on openmind
 
 
 # To extract features for a given model:
 
 

 
