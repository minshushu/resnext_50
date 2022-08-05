nohup python train.py --fold 0 --model_name resnext50 --nfolds 5 --batch_size 16 --epochs 20 --lr 1e-4 --expansion 0 --workers 16;
python train.py --fold 1 --model_name resnext50 --nfolds 5 --batch_size 16 --epochs 20 --lr 1e-4 --expansion 0 --workers 16;
python train.py --fold 2 --model_name resnext50 --nfolds 5 --batch_size 16 --epochs 20 --lr 1e-4 --expansion 0 --workers 16;
python train.py --fold 3 --model_name resnext50 --nfolds 5 --batch_size 16 --epochs 20 --lr 1e-4 --expansion 0 --workers 16;
python train.py --fold 4 --model_name resnext50 --nfolds 5 --batch_size 16 --epochs 20 --lr 1e-4 --expansion 0 --workers 16 &
