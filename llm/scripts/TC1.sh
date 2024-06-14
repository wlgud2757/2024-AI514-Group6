for dataset in TC1-1 TC1-2 TC1-3 TC1-4 TC1-5 TC1-6 TC1-7 TC1-8 TC1-9 TC1-10
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