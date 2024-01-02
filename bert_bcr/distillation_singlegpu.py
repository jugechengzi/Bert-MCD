


import argparse
import os
import sys
sys.path.append('../')
import time
import math
import torch.distributed as dist
import torch
import transformers as ppb

from process_data.bert_beer import BeerData_bert,BeerAnnotation_bert,Hotel_bert,HotelAnnotation_bert,BeerData_bert_correlated
from hotel import HotelData,HotelAnnotation
from embedding import get_embeddings,get_glove_embedding
from torch.utils.data import DataLoader

from model import Bert_rnp,Bert_classfier
from train_util_bert_bcr import train_bert_bcr_onegpu_distillation,validate_share_bert_onegpu,dev_bert_bcr_onetigpu
# from validate_util import validate_share, validate_dev_sentence, validate_annotation_sentence, validate_rationales
from tensorboardX import SummaryWriter

from  multi_gpu_util import init_distributed_mode





def parse():
    #默认： nonorm, dis_lr=1, data=beer, save=0
    parser = argparse.ArgumentParser(
        description="SR")
    # multi gpu
    parser.add_argument('--dist_url',
                        type=str,
                        default='env://',
                        help='')

    # pretrained embeddings
    parser.add_argument('--embedding_dir',
                        type=str,
                        default='../data/hotel/embeddings',
                        help='Dir. of pretrained embeddings [default: None]')
    parser.add_argument('--embedding_name',
                        type=str,
                        default='glove.6B.100d.txt',
                        help='File name of pretrained embeddings [default: None]')
    parser.add_argument('--max_length',
                        type=int,
                        default=256,
                        help='Max sequence length [default: 256]')

    # dataset parameters
    parser.add_argument('--data_dir',
                        type=str,
                        default='../data/beer',
                        help='Path of the dataset')
    parser.add_argument('--data_type',
                        type=str,
                        default='beer',
                        help='0:beer,1:hotel')
    parser.add_argument('--aspect',
                        type=int,
                        default=2,
                        help='The aspect number of beer review [0, 1, 2]')
    parser.add_argument('--correlated',
                        type=int,
                        default=0,
                        help='The aspect number of beer review [0, 1, 2]')
    parser.add_argument('--seed',
                        type=int,
                        default=12252018,
                        help='The aspect number of beer review [20226666,12252018]')
    parser.add_argument('--annotation_path',
                        type=str,
                        default='../data/beer/annotations.json',
                        help='Path to the annotation')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size [default: 100]')


    # model parameters
    parser.add_argument('--gen_acc',
                        type=int,
                        default=0,
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--pred_div',
                        type=int,
                        default=0,
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--gen_sparse',
                        type=int,
                        default=0,
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--div',
                        type=str,
                        default='kl',
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--freeze_bert',
                        type=int,
                        default=0,
                        help='')
    parser.add_argument('--sp_norm',
                        type=int,
                        default=0,
                        help='0:rnp,1:norm')
    parser.add_argument('--dis_lr',
                        type=float,
                        default=0,
                        help='0:rnp,1:dis')
    parser.add_argument('--save',
                        type=int,
                        default=0,
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--cell_type',
                        type=str,
                        default="electra",
                        help='Cell type: LSTM, GRU [default: GRU]')
    parser.add_argument('--num_layers',
                        type=int,
                        default=1,
                        help='RNN cell layers')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        help='Network Dropout')
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=100,
                        help='Embedding dims [default: 100]')
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=256,
                        help='RNN hidden dims [default: 100]')
    parser.add_argument('--num_class',
                        type=int,
                        default=2,
                        help='Number of predicted classes [default: 2]')

    # ckpt parameters
    parser.add_argument('--classifier_path',
                        type=str,
                        default='./trained_model/classifier/correlated0.pth',
                        help='Base dir of output files')

    # learning parameters
    parser.add_argument('--epochs',
                        type=int,
                        default=37,
                        help='Number of training epoch')
    parser.add_argument('--encoder_lr',
                        type=float,
                        default=0.00001,
                        help='compliment learning rate [default: 1e-3]')
    parser.add_argument('--fc_lr',
                        type=float,
                        default=0.0001,
                        help='compliment learning rate [default: 1e-3]')
    parser.add_argument('--sparsity_lambda',
                        type=float,
                        default=12.,
                        help='Sparsity trade-off [default: 1.]')
    parser.add_argument('--continuity_lambda',
                        type=float,
                        default=10.,
                        help='Continuity trade-off [default: 4.]')
    parser.add_argument(
        '--sparsity_percentage',
        type=float,
        default=0.1,
        help='Regularizer to control highlight percentage [default: .2]')
    parser.add_argument(
        '--cls_lambda',
        type=float,
        default=0.9,
        help='lambda for classification loss')
    parser.add_argument('--gpu',
                        type=int,
                        default=3,
                        help='id(s) for CUDA_VISIBLE_DEVICES [default: None]')
    parser.add_argument('--share',
                        type=int,
                        default=0,
                        help='')
    parser.add_argument(
        '--writer',
        type=str,
        default='./noname',
        help='Regularizer to control highlight percentage [default: .2]')
    args = parser.parse_args()
    return args


#####################
# set random seed
#####################
# torch.manual_seed(args.seed)

#####################
# parse arguments
#####################
args = parse()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
# print("\nParameters:")
# for attr, value in sorted(args.__dict__.items()):
#     print("\t{}={}".format(attr.upper(), value))


print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

######################
# device
######################
# os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

# torch.cuda.set_device(int(args.gpu))
device = torch.device("cuda:{}".format(args.gpu))


######################
# load embedding
######################
# pretrained_embedding, word2idx = get_glove_embedding(os.path.join(args.embedding_dir, args.embedding_name))
# args.vocab_size = len(word2idx)
# args.pretrained_embedding = pretrained_embedding

######################
# load dataset
######################
#tokenizer
if args.cell_type=='bert':
    tokenizer_class, pretrained_weights = (ppb.BertTokenizerFast, 'bert-base-uncased')
    args.hidden_dim=768
elif args.cell_type=='electra':
    tokenizer_class, pretrained_weights = (ppb.ElectraTokenizerFast, 'google/electra-small-discriminator')
    args.hidden_dim = 256
elif args.cell_type=='roberta':
    tokenizer_class, pretrained_weights = (ppb.RobertaTokenizerFast, "roberta-base")
    args.hidden_dim = 768
else:
    print('undefined cell type')

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

if args.data_type=='beer':       #beer
    args.data_dir = '../data/beer'
    args.annotation_path = '../data/beer/annotations.json'
    if args.correlated==1:
        train_data = BeerData_bert_correlated(tokenizer, args.data_dir, args.aspect, 'train', balance=True)
        dev_data = BeerData_bert_correlated(tokenizer, args.data_dir, args.aspect, 'dev')
    else:
        train_data = BeerData_bert(tokenizer, args.data_dir, args.aspect, 'train', balance=True)
        dev_data = BeerData_bert(tokenizer, args.data_dir, args.aspect, 'dev')
    annotation_data = BeerAnnotation_bert(tokenizer,args.annotation_path, args.aspect)
elif args.data_type == 'hotel':       #hotel
    args.data_dir='../data/hotel'
    args.annotation_path='../data/hotel/annotations'
    train_data = Hotel_bert(tokenizer,args.data_dir, args.aspect, 'train', balance=True)

    dev_data = Hotel_bert(tokenizer,args.data_dir, args.aspect, 'dev')

    annotation_data = HotelAnnotation_bert(args.annotation_path, args.aspect)

# shuffle and batch the dataset
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,drop_last=True)

dev_loader = DataLoader(dev_data, batch_size=args.batch_size)

annotation_loader = DataLoader(annotation_data, batch_size=args.batch_size)

######################
# load model
######################
writer=SummaryWriter(args.writer)
model=Bert_rnp(args)
model.to(device)

classifier=Bert_classfier(args)
if os.path.exists(args.classifier_path):
    classifier.load_state_dict(torch.load(args.classifier_path,map_location=device))
    print('load classifier from {}'.format(args.classifier_path))
classifier.eval()
classifier.to(device)
for name,p in classifier.named_parameters():
    p.requires_grad=False







######################
# Training


######################
# g_para=list(map(id, model.generator.parameters()))
# p_para=filter(lambda p: id(p) not in g_para, model.parameters())
# lr2=args.lr
# lr1=args.lr
# para=[
#     {'params': model.generator.parameters(), 'lr':lr1},
#     {'params':p_para,'lr':lr2}
# ]
# optimizer = torch.optim.Adam(para)
# print('lr1={},lr2={}'.format(lr1,lr2))


lr_gen_encoder=args.encoder_lr
lr_gen_fc=args.fc_lr
if args.dis_lr == 2:
    lr_pred_encoder=args.encoder_lr/2
    lr_pred_fc=args.fc_lr/2
elif args.dis_lr == 3:
    lr_pred_encoder = args.encoder_lr / 3
    lr_pred_fc = args.fc_lr / 3
else:
    lr_pred_encoder = args.encoder_lr
    lr_pred_fc = args.fc_lr

para_gen_encoder=filter(lambda p: p.requires_grad, model.gen_encoder.parameters())
para_gen_fc=filter(lambda p: p.requires_grad, model.gen_fc.parameters())
para_gen_layernorm=filter(lambda p: p.requires_grad, model.layernorm1.parameters())
para_pred_encoder=filter(lambda p: p.requires_grad, model.pred_encoder.parameters())
para_pred_fc=filter(lambda p: p.requires_grad, model.pred_fc.parameters())
para_pred_layernorm=filter(lambda p: p.requires_grad, model.layernorm2.parameters())
# para=[
#     {'params':para_gen_encoder,'lr': lr_gen_encoder},
#     {'params':para_gen_fc,'lr': lr_gen_fc},
#     {'params':para_gen_layernorm,'lr': lr_gen_fc},
# {'params':para_pred_encoder,'lr': lr_pred_encoder},
# {'params':para_pred_fc,'lr': lr_pred_fc},
# {'params':para_pred_layernorm,'lr': lr_pred_fc}
# ]
para_gen=[{'params':para_gen_encoder,'lr': lr_gen_encoder},
    {'params':para_gen_fc,'lr': lr_gen_fc},
    {'params':para_gen_layernorm,'lr': lr_gen_fc}]
para_pred=[{'params':para_pred_encoder,'lr': lr_pred_encoder},
{'params':para_pred_fc,'lr': lr_pred_fc},
{'params':para_pred_layernorm,'lr': lr_pred_fc}]

opt_gen=torch.optim.Adam(para_gen)
opt_pred=torch.optim.Adam(para_pred)
# optimizer = torch.optim.Adam(para)





######################
# Training
######################
strat_time=time.time()
best_all = 0
f1_best_dev = [0]
best_dev_epoch = [0]
acc_best_dev = [0]
grad=[]
grad_loss=[]
for epoch in range(args.epochs):

    start = time.time()
    model.train()
    precision, recall, f1_score, accuracy = train_bert_bcr_onegpu_distillation(model,classifier, opt_gen,opt_pred, train_loader, device, args,(writer,epoch),annotation_loader)
    # precision, recall, f1_score, accuracy = train_noshare(model, optimizer, train_loader, device, args)
    end = time.time()

    print('\nTrain time for epoch #%d : %f second' % (epoch, end - start))
    # print('gen_lr={}, pred_lr={}'.format(optimizer.param_groups[0]['lr'], optimizer.param_groups[3]['lr']))
    print("traning epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(epoch, recall,
                                                                                                   precision, f1_score,
                                                                                                   accuracy))
    writer.add_scalar('train_acc',accuracy,epoch)
    writer.add_scalar('time',time.time()-strat_time,epoch)
    model.eval()

    precision, recall, f1_score, accuracy=dev_bert_bcr_onetigpu(model,  dev_loader, device, args,(writer,epoch))

    print("Validate")
    print("dev epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(epoch, recall,
                                                                                               precision,
                                                                                               f1_score, accuracy))

    writer.add_scalar('dev_acc',accuracy,epoch)
    # print("Validate Sentence")
    # validate_dev_sentence(model, dev_loader, device,(writer,epoch))

    annotation_results = validate_share_bert_onegpu(model, annotation_loader, device)

    print("Annotation")
    print(
        "The annotation performance: sparsity: %.4f, precision: %.4f, recall: %.4f, f1: %.4f"
        % (100 * annotation_results[0], 100 * annotation_results[1],
           100 * annotation_results[2], 100 * annotation_results[3]))
    writer.add_scalar('f1',100 * annotation_results[3],epoch)
    writer.add_scalar('sparsity',100 * annotation_results[0],epoch)
    writer.add_scalar('p', 100 * annotation_results[1], epoch)
    writer.add_scalar('r', 100 * annotation_results[2], epoch)

    # print("Annotation Sentence")
    # validate_annotation_sentence(model, annotation_loader, device)
    # print("Rationale")
    # validate_rationales(model, annotation_loader, device,(writer,epoch))
    if accuracy>acc_best_dev[-1]:
        acc_best_dev.append(accuracy)
        best_dev_epoch.append(epoch)
        f1_best_dev.append(annotation_results[3])
    if best_all<annotation_results[3]:
        best_all=annotation_results[3]





print(best_all)
print(acc_best_dev)
print(best_dev_epoch)
print(f1_best_dev)







# if args.save==1:
#     if args.data_type=='beer':
#         torch.save(model.state_dict(),'./trained_model/beer/dr_aspect{}_dis{}_{}.pkl'.format(args.aspect,args.dis_lr,args.cell_type))
#         print('save the model')
#     elif args.data_type=='hotel':
#         torch.save(model.state_dict(), './trained_model/hotel/dr_aspect{}_dis{}_{}.pkl'.format(args.aspect, args.dis_lr,args.cell_type))
#         print('save the model')
# else:
#     print('not save')

