python main.py --dataset shanghaitechpa \
--model CSRNet \
--train-files /home/jqgao/workspace/CrowdCounting/TrainingTestingFileLists/ShanghaiTechPartA_full_origin_train.txt \
--val-files /home/jqgao/workspace/CrowdCounting/TrainingTestingFileLists/ShanghaiTechPartA_full_origin_val.txt \
--gpu-devices 4 \
--lr 1e-5 \
--optim adam \
--loss mseloss \
--checkpoints ./checkpoints/demo \
--summary-writer ./runs/demo