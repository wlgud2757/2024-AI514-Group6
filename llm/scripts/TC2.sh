for dataset in TC2-1 TC2-2 TC2-3 TC2-4 TC2-5
do
    echo -e "\n\n########## $dataset ##########"
    for scale in sentence
    do
        CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python get_embedding.py \
            --model_name hkunlp/instructor-large \
            --scale $scale \
            --task_name TC \
            --data_path ../../dataset/${dataset}/${scale}.jsonl \
            --result_file ../../dataset/${dataset}/${scale}_embeds.hdf5 \
            --measure
    done
done