This repo contains the code for [MCD](https://github.com/jugechengzi/Rationalization-MCD) with pretrained bert/electra.

Here is a demo about how to run it. You can refer to [MCD](https://github.com/jugechengzi/Rationalization-MCD) for the environments.

**First, train a classifier.**  Since I find that the training of BERT is unstable, I no more share the parameters of the two approximaters as shown in Figure 3. You need to train a classifier and save it, which is used to approximate the distribution $P(Y|X)$.You can get the classifier with:  

python train_classifier_onegpu.py --correlated 0 --freeze_bert 0 --cell_type electra --dis_lr 0 --data_type beer --save 1 --dropout 0.2 --encoder_lr 0.00005 --fc_lr 0.0001 --batch_size 64 --sparsity_percentage 0.185 --sparsity_lambda 18 --continuity_lambda 12 --epochs 11 --aspect 0 --save_path ./trained_model/classifier/electra_decorrelated0.pth 


**Then, you can train the MCD**. For beer-appearance in table 5, you should run:    
python distillation_singlegpu.py --correlated 0 --freeze_bert 0 --cell_type electra --dis_lr 0 --data_type beer --save 0 --dropout 0.2 --encoder_lr 0.00001 --fc_lr 0.0001 --batch_size 128 --sparsity_percentage 0.175 --sparsity_lambda 18 --continuity_lambda 12 --epochs 80 --aspect 0 --pred_div 0 --classifier_path ./trained_model/classifier/electra_decorrelated0.pth 



Any questions, please open an issue, and I will appreciate your contribution.   
If you are interested in this repo, please star it before cloning it.
