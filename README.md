# Instructions to Execute the model

**1.Download and unzip the repo , Create the environment as follows:**

python -m venv audio_env

source audio_env/bin/activate

pip install -r requirements.txt

**2.Create the dataset**
 
 Download the dataset from https://github.com/karolpiczak/esc-50 especially the audio and meta folders and store 

 Execute the given script to generate the input, clean and test data

 cd esper-audio-challenge

 python make_field_queries.py --esc50-root datasetsourceroot --out destinationfolder

 
