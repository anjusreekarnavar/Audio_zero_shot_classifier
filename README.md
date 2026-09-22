# Instructions to Execute the model

**1.Download and unzip the repo , Create the environment as follows:**

python -m venv audio_env

source audio_env/bin/activate

pip install -r requirements.txt

**2.Create the dataset**
 
 Download the dataset from https://github.com/karolpiczak/esc-50 especially the audio and meta folders and store 

 Execute the given script to generate the input, clean(reference) and test data

 cd esper-audio-challenge

 python make_field_queries.py --esc50-root datasetsourceroot --out destinationfolder

 This query creates three important groups check and confirm

  Fold 1       → clean reference 

  Folds 2,3    → input 

  Folds 4,5    → test and 
   
   Later test  data church_bells, helicopter, chainsaw, hand_saw, engine

 **3.Implementation with CLAP baseline**

      model_name="laion/clap-htsat-unfused" will be automatically downloaded otherwise download and store in the local system and             provide the path in the file encoder.py

  ***Create the encoding of all the samples by running build_reference_bank.py change the fields in the file according to your folder      structure***
    
     ESC50_ROOT = "/scratch/s5284664/ESC-50-master"
     
     MANIFEST = "field_queries/manifest_library.csv"
     
     OUTPUT = "cache/reference_bank.npz"
     
     python build_reference_bank.py this saves cache/reference_bank.npz

  ***Experiment-1. Mean Strategy***

  Change the path in the file evaluate_baseline.py
  
  REFERENCE_BANK = "cache/reference_bank.npz"
  
  MANIFEST = "field_queries/manifest_dev.csv"
  
  GROUND_TRUTH = "field_queries/abundance_dev.csv"
  
  QUERY_ROOT = "field_queries"


  and execute

   python evaluate_baseline.py
    

 

 
