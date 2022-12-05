export DB_ROOT=./db

python test_dir_patents.py \
    --dataset ./db \
    --arch="resnet50_rmac" \
    --checkpoint ./checkpoints/PatentNet_Tri_GeM.pth.tar \
    --image_size=256 \
    --gpu -1