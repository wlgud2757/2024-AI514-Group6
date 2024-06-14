# [2024-Spring AI514 Final Project] Group 6

## Contact
- Yongsik Jo (josik@unist.ac.kr)
- Soyeong Kwon (soyoung17@unist.ac.kr)
- Junsu Kim (jjunsssk@unist.ac.kr)
- Jihyoung Jang (jihyoung@unist.ac.kr)

## Requirements
```Shell
git clone https://github.com/wlgud2757/2024-AI514-Group6.git
conda create -n {YOUR_ENV_NAME} python=3.9
pip install -r requirements.txt
```
## Useage
Please place the dataset in the main directory before running the code (TC1-all_including-GT, TC2, and real-world dataset).

1. To represent the graph network in sentences, please use the `formatting.ipynb` file inside each dataset folder. A `sentence.jsonl` file should be generated for each network.
2. Once the files representing each network in sentences have been generated, please run the following command.

```Shell
cd llm
conda activate {YOUR_ENV_NAME}
bash TC1.sh
bash TC2.sh
bash real-world.sh
```

3. The experiment results will be output to stdout, and the embedding files will be automatically generated in the folder of each dataset.

## Acknowledgments
- This project is a repository for the final project of the AI514 course at UNIST AIGS for the Spring 2024 semester.
- Base code is from ["ClusterLLM: Large Language Models as a Guide for Text Clustering" paper's code repository](https://github.com/zhang-yu-wei/ClusterLLM?tab=readme-ov-file).
