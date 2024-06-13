# Instructions

## 0. Set environment
conda create -n env bd python=3.5 -y 
conda activate bd 
pip install -r requirements.txt

## 1. Convert .dat Files to CSV

Run the following script to process .dat files into CSV format:
- python dat2csv.py

## 2. Run Gemsec Embedding Code

Execute the following scripts to run the Gemsec embedding. The JSON files stored in each folder are the clustered community output files:
- python bigdata/src/embedding_clustering_tc.py  # For TC
- python bigdata/src/embedding_clustering_real_world.py  # For Real World

## 3. Convert JSON Files to .dat Files

Run the following script to convert the generated JSON files into .dat format:
python json2dat.py

## 4. Evaluate the Results

Execute the following scripts to evaluate the results:
- python bigdata/evaluation_tc.py  # For TC
- python bigdata/evaluation_rw.py  # For Real World
