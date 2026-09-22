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

   Results will be available in results/baseline_dev.csv

   To avoid multiple executions of the embedding creation, it's possible to cache all the embeddings python cache_dev_embeddings.py

   **Experiment 2 Max Strategy**
   
   Change the paths in the file evaluate_max8.py

   
   REFERENCE_EMBEDDINGS = "cache/reference_embeddings.npz"
   
   MANIFEST = "field_queries/manifest_dev.csv"
   
   GROUND_TRUTH = "field_queries/abundance_dev.csv"
   
   QUERY_ROOT = "field_queries"
   
   OUTPUT = "results/max8_dev.csv" and run

   python evaluate_max8.py

   **Experiment 3 Top N strategy**

   Edit the path in evaluate_topn.py. If the embeddings are cached by executing the file cache_dev_embeddings.py to improve the change     
   the timing, change the path of the DEV_EMBEDDINGS accordingly 

   DEV_EMBEDDINGS = "cache/dev_embeddings.npz"
   
   REFERENCE_EMBEDDINGS = "cache/reference_embeddings.npz"

   MANIFEST = "field_queries/manifest_dev.csv"
   
   GROUND_TRUTH = "field_queries/abundance_dev.csv"

   OUTPUT_DIR = "results"

   Execute  python evaluate_topn.py

   **Abundance Estimation using TOP3**

    python evaluate_abundance_threshold.py change the paths as per strategy 3

    **Inference**
    
    Replace the paths accordingly

    time python final_predict.py \
    --esc50-root /scratch/s5284664/ESC-50-master \
    --library-manifest field_queries/manifest_library.csv \
    --query-manifest field_queries/manifest_test.csv \
    --query-dir field_queries/test \
    --output results/final/predictions_test.csv

    **Zero shot evaluation**
    
    time python final_predict.py \
    --esc50-root /scratch/s5284664/ESC-50-master \
    --library-manifest results/final/manifest_library_25.csv \
    --query-manifest field_queries/manifest_test_late.csv \
    --query-dir field_queries/test_late \
    --output results/final/predictions_test_late.csv

    This also gives the total evalaution time 

    **Failure Analysis**
    
    python analyze_failures.py

    # ABLATION STUDY USING AudioMAE

    AudioMAE also uses the same datasets and folds genearted . Uses the pretrained model from 
    https://github.com/facebookresearch/AudioMAE the ViT-B, AS-2M pretrained. Store the pretrained weights from the repo and use teh       path of this 

    

   

   
    

 

 
