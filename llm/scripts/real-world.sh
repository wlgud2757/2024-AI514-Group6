for dataset in dolphin football karate mexican polbooks railway strike
do
    echo -e "\n\n########## $dataset ##########"
    for scale in sentence
    do
        CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python get_embedding.py \
            --model_name hkunlp/instructor-large \
            --scale $scale \
            --task_name $dataset \
            --data_path ../../dataset/${dataset}/${scale}.jsonl \
            --result_file ../../dataset/${dataset}/${scale}_embeds.hdf5 \
            --measure
    done
done