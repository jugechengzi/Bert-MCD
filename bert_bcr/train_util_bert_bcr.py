import torch
import torch.nn.functional as F
import torch.nn as nn

from metric import get_sparsity_loss, get_continuity_loss, computer_pre_rec
import numpy as np
import math
import torch.distributed as dist
import sys
sys.path.append('../')

from metric import compute_micro_stats

def reduce_value(value,world_size=2,average=False):
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value=value/world_size

        return value



class JS_DIV(nn.Module):
    def __init__(self):
        super(JS_DIV, self).__init__()
        self.kl_div=nn.KLDivLoss(reduction='batchmean',log_target=True)
    def forward(self,p,q):
        p=F.softmax(p,dim=-1)
        q=F.softmax(q,dim=-1)
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl_div(m, p.log()) + self.kl_div(m, q.log()))




def train_bert_bcr(model, optimizer, dataset, device, args,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len = len(dataset)

    for (batch, (inputs, masks, labels,special_masks)) in enumerate(dataset):

        optimizer.zero_grad()

        inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device),special_masks.to(device)

        # rationales, cls_logits
        rationales, logits = model(inputs, masks,special_masks)

        #full_text_logits
        full_text_logits=model.train_one_step(inputs, masks,special_masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        full_text_cls_loss=args.cls_lambda * F.cross_entropy(full_text_logits, labels)

        #jsd
        jsd_func = JS_DIV()
        jsd_loss = jsd_func(logits, full_text_logits)


        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales, masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales) / torch.sum(masks)).cpu().item()
        writer_epoch[0].add_scalar('train_sp', sparsity, writer_epoch[1]*len(dataset)+batch)
        # print(sparsity)
        # print(rationales[0,:10,1])
        train_sp.append(
            (torch.sum(rationales) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales)

        loss = cls_loss + sparsity_loss + continuity_loss+jsd_loss+full_text_cls_loss
        # update gradient


        loss.backward()
        optimizer.step()

        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # print('get grad')
    # grad_max, grad_mean = get_grad(model, dataset, 2, 1, device)  # 获取训练过程中的lipschitz常数
    # print('get grad end')
    # writer_epoch[0].add_scalar('max_grad', grad_max, writer_epoch[1])
    # writer_epoch[0].add_scalar('avg_grad', grad_mean, writer_epoch[1])

    writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    return precision, recall, f1_score, accuracy


def train_bert_bcr_multigpu(model, optimizer, dataset, device, args,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []

    for (batch, (inputs, masks, labels,special_masks)) in enumerate(dataset):

        optimizer.zero_grad()

        inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device),special_masks.to(device)

        # rationales, cls_logits
        rationales, logits = model.module(inputs, masks,special_masks)

        #full_text_logits
        full_text_logits=model.module.train_one_step(inputs, masks,special_masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        full_text_cls_loss=args.cls_lambda * F.cross_entropy(full_text_logits, labels)

        #jsd
        jsd_func = JS_DIV()
        jsd_loss = jsd_func(logits, full_text_logits)


        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales, masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales) / torch.sum(masks)).cpu().item()
        writer_epoch[0].add_scalar('train_sp', sparsity, writer_epoch[1]*len(dataset)+batch)
        # print(sparsity)
        # print(rationales[0,:10,1])
        train_sp.append(
            (torch.sum(rationales) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales)

        loss = cls_loss + sparsity_loss + continuity_loss+jsd_loss+full_text_cls_loss
        # update gradient


        loss.backward()
        optimizer.step()

        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    # 等待所有进程
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)


    TP=reduce_value(TP)
    FP = reduce_value(FP)
    FN = reduce_value(FN)
    TN = reduce_value(TN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # print('get grad')
    # grad_max, grad_mean = get_grad(model, dataset, 2, 1, device)  # 获取训练过程中的lipschitz常数
    # print('get grad end')
    # writer_epoch[0].add_scalar('max_grad', grad_max, writer_epoch[1])
    # writer_epoch[0].add_scalar('avg_grad', grad_mean, writer_epoch[1])
    if args.rank==0:
        writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
        writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
        writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
        writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    return precision, recall, f1_score, accuracy

def train_bert_bcr_multigpu_decouple(model, opt_gen,opt_pred, dataset, device, args,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []

    for (batch, (inputs, masks, labels,special_masks)) in enumerate(dataset):
        opt_gen.zero_grad()
        opt_pred.zero_grad()

        inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device),special_masks.to(device)

        # train classification
        # get rationales
        rationales, rationales_add_special_token = model.module.get_rationale(inputs, masks, special_masks)
        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales, masks, args.sparsity_percentage)

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales)

        if args.gen_acc == 0:
            forward_logit = model.module.pred_forward_logit(inputs, masks, torch.detach(rationales_add_special_token))
        elif args.gen_acc==1:
            forward_logit = model.module.pred_forward_logit(inputs, masks, rationales_add_special_token)
        else:
            print('wrong gen acc')



        # detach_logit = model.module.detach_gen_pred(inputs, masks, rationales_add_special_token)
        # forward_logit = model.module.pred_forward_logit(inputs, masks, rationales_add_special_token)

        #full_text_logits
        full_text_logits=model.module.train_one_step(inputs, masks,special_masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(forward_logit, labels)
        full_text_cls_loss=args.cls_lambda * F.cross_entropy(full_text_logits, labels)

        if args.gen_sparse==1:
            classification_loss=cls_loss+full_text_cls_loss+sparsity_loss + continuity_loss
        elif args.gen_sparse==0:
            classification_loss = cls_loss +  full_text_cls_loss
        else:
            print('gen sparse wrong')

        classification_loss.backward()

        opt_pred.step()
        opt_pred.zero_grad()

        if args.gen_acc==1:
            opt_gen.step()
            opt_gen.zero_grad()
        elif args.gen_sparse==1:
            opt_gen.step()
            opt_gen.zero_grad()
        else:
            pass

        #train divergence
        opt_gen.zero_grad()
        name1 = []
        name2 = []
        name3 = []
        for idx, p in model.module.pred_encoder.named_parameters():
            if p.requires_grad == True:
                name1.append(idx)
                p.requires_grad = False
        for idx, p in model.module.pred_fc.named_parameters():
            if p.requires_grad == True:
                name2.append(idx)
                p.requires_grad = False
        for idx, p in model.module.layernorm2.named_parameters():
            if p.requires_grad == True:
                name3.append(idx)
                p.requires_grad = False

        rationales, rationales_add_special_token = model.module.get_rationale(inputs, masks, special_masks)
        forward_logit = model.module.pred_forward_logit(inputs, masks, rationales_add_special_token)
        full_text_logits = model.module.train_one_step(inputs, masks, special_masks)


        #jsd
        if args.div == 'js':
            jsd_func = JS_DIV()
            jsd_loss = jsd_func(forward_logit, full_text_logits)
        elif args.div == 'kl':
            jsd_loss = nn.functional.kl_div(F.softmax(forward_logit, dim=-1).log(), F.softmax(full_text_logits, dim=-1),
                                            reduction='batchmean')
        else:
            print('div wrong')



        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales, masks, args.sparsity_percentage)


        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales)

        gen_loss = sparsity_loss + continuity_loss + jsd_loss
        # update gradient

        gen_loss.backward()
        opt_gen.step()
        opt_gen.zero_grad()
        n1 = 0
        n2 = 0
        n3 = 0
        for idx,p in model.module.pred_encoder.named_parameters():
            if idx in name1:
                p.requires_grad=True
                n1+=1
        for idx,p in model.module.pred_fc.named_parameters():
            if idx in name2:
                p.requires_grad = True
                n2 += 1
        for idx,p in model.module.layernorm2.named_parameters():
            if idx in name3:
                p.requires_grad = True
                n3 += 1





        cls_soft_logits = torch.softmax(forward_logit, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    # 等待所有进程
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)


    TP=reduce_value(TP)
    FP = reduce_value(FP)
    FN = reduce_value(FN)
    TN = reduce_value(TN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # print('get grad')
    # grad_max, grad_mean = get_grad(model, dataset, 2, 1, device)  # 获取训练过程中的lipschitz常数
    # print('get grad end')
    # writer_epoch[0].add_scalar('max_grad', grad_max, writer_epoch[1])
    # writer_epoch[0].add_scalar('avg_grad', grad_mean, writer_epoch[1])
    # if args.rank==0:
    #     writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    #     writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    #     writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    #     writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    return precision, recall, f1_score, accuracy

def train_bert_bcr_onegpu_decouple(model, opt_gen,opt_pred, dataset, device, args,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []

    for (batch, (inputs, masks, labels,special_masks)) in enumerate(dataset):
        opt_gen.zero_grad()
        opt_pred.zero_grad()

        inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device),special_masks.to(device)

        # train classification
        # get rationales
        rationales, rationales_add_special_token = model.get_rationale(inputs, masks, special_masks)
        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales, masks, args.sparsity_percentage)

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales)

        if args.gen_acc == 0:
            forward_logit = model.pred_forward_logit(inputs, masks, torch.detach(rationales_add_special_token))
        elif args.gen_acc==1:
            forward_logit = model.pred_forward_logit(inputs, masks, rationales_add_special_token)
        else:
            print('wrong gen acc')


        #full_text_logits
        full_text_logits=model.train_one_step(inputs, masks,special_masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(forward_logit, labels)
        full_text_cls_loss=args.cls_lambda * F.cross_entropy(full_text_logits, labels)

        if args.gen_sparse==1:
            classification_loss=cls_loss+full_text_cls_loss+sparsity_loss + continuity_loss
        elif args.gen_sparse==0:
            classification_loss = (cls_loss +  full_text_cls_loss)/2
        else:
            print('gen sparse wrong')

        classification_loss.backward()

        opt_pred.step()
        opt_pred.zero_grad()

        if args.gen_acc==1:
            opt_gen.step()
            opt_gen.zero_grad()
        else:
            pass

        #train divergence
        name1 = []
        name2 = []
        name3 = []
        for idx, p in model.pred_encoder.named_parameters():
            if p.requires_grad == True:
                name1.append(idx)
                p.requires_grad = False
        for idx, p in model.pred_fc.named_parameters():
            if p.requires_grad == True:
                name2.append(idx)
                p.requires_grad = False
        for idx, p in model.layernorm2.named_parameters():
            if p.requires_grad == True:
                name3.append(idx)
                p.requires_grad = False

        if args.gen_acc==1:
            rationales, rationales_add_special_token = model.get_rationale(inputs, masks, special_masks)
            sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
                rationales, masks, args.sparsity_percentage)

            continuity_loss = args.continuity_lambda * get_continuity_loss(
                rationales)
        forward_logit = model.pred_forward_logit(inputs, masks, rationales_add_special_token)
        full_text_logits = model.train_one_step(inputs, masks, special_masks)


        #jsd
        if args.div == 'js':
            jsd_func = JS_DIV()
            jsd_loss = jsd_func(forward_logit, full_text_logits)
        elif args.div == 'kl':
            jsd_loss = nn.functional.kl_div(F.softmax(forward_logit, dim=-1).log(), F.softmax(full_text_logits, dim=-1),
                                            reduction='batchmean')
        else:
            print('div wrong')





        gen_loss = sparsity_loss + continuity_loss + jsd_loss
        # update gradient

        gen_loss.backward()
        opt_gen.step()
        opt_gen.zero_grad()
        n1 = 0
        n2 = 0
        n3 = 0
        for idx,p in model.pred_encoder.named_parameters():
            if idx in name1:
                p.requires_grad=True
                n1+=1
        for idx,p in model.pred_fc.named_parameters():
            if idx in name2:
                p.requires_grad = True
                n2 += 1
        for idx,p in model.layernorm2.named_parameters():
            if idx in name3:
                p.requires_grad = True
                n3 += 1





        cls_soft_logits = torch.softmax(forward_logit, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()




    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # print('get grad')
    # grad_max, grad_mean = get_grad(model, dataset, 2, 1, device)  # 获取训练过程中的lipschitz常数
    # print('get grad end')
    # writer_epoch[0].add_scalar('max_grad', grad_max, writer_epoch[1])
    # writer_epoch[0].add_scalar('avg_grad', grad_mean, writer_epoch[1])
    # if args.rank==0:
    #     writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    #     writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    #     writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    #     writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    return precision, recall, f1_score, accuracy

def dev_bert_bcr_multigpu(model, dataset, device, args,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len = len(dataset)
    with torch.no_grad():
        for (batch, (inputs, masks, labels, special_masks)) in enumerate(dataset):


            inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device), special_masks.to(
                device)

            # rationales, cls_logits


            rationales, logits = model.module(inputs, masks, special_masks)



            cls_soft_logits = torch.softmax(logits, dim=-1)
            _, pred = torch.max(cls_soft_logits, dim=-1)

            # TP predict 和 label 同时为1
            TP += ((pred == 1) & (labels == 1)).cpu().sum()
            # TN predict 和 label 同时为0
            TN += ((pred == 0) & (labels == 0)).cpu().sum()
            # FN predict 0 label 1
            FN += ((pred == 0) & (labels == 1)).cpu().sum()
            # FP predict 1 label 0
            FP += ((pred == 1) & (labels == 0)).cpu().sum()

        # 等待所有进程
        if device!= torch.device('cpu'):
            torch.cuda.synchronize(device)
    TP = reduce_value(TP)
    FP = reduce_value(FP)
    FN = reduce_value(FN)
    TN = reduce_value(TN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy

def dev_bert_bcr_onetigpu(model, dataset, device, args,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len = len(dataset)
    with torch.no_grad():
        for (batch, (inputs, masks, labels, special_masks)) in enumerate(dataset):


            inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device), special_masks.to(
                device)

            # rationales, cls_logits


            rationales, logits = model(inputs, masks, special_masks)



            cls_soft_logits = torch.softmax(logits, dim=-1)
            _, pred = torch.max(cls_soft_logits, dim=-1)

            # TP predict 和 label 同时为1
            TP += ((pred == 1) & (labels == 1)).cpu().sum()
            # TN predict 和 label 同时为0
            TN += ((pred == 0) & (labels == 0)).cpu().sum()
            # FN predict 0 label 1
            FN += ((pred == 0) & (labels == 1)).cpu().sum()
            # FP predict 1 label 0
            FP += ((pred == 1) & (labels == 0)).cpu().sum()



    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy


def validate_share_bert_multigpu(model, annotation_loader, device):
    num_true_pos = 0.
    num_predicted_pos = 0.
    num_real_pos = 0.
    num_words = 0

    TP = 0
    TN = 0
    FN = 0
    FP = 0
    with torch.no_grad():
        for (batch, (inputs, masks, labels,
                     annotations,special_masks)) in enumerate(annotation_loader):
            inputs, masks, labels, annotations,special_masks = inputs.to(device), masks.to(device), labels.to(device), annotations.to(
                device), special_masks.to(device)

            # rationales -- (batch_size, seq_length, 2)
            rationales, cls_logits = model.module(inputs, masks,special_masks)

            num_true_pos_, num_predicted_pos_, num_real_pos_ = compute_micro_stats(
                annotations, rationales)

            soft_pred = F.softmax(cls_logits, -1)
            _, pred = torch.max(soft_pred, dim=-1)

            # TP predict 和 label 同时为1
            TP += ((pred == 1) & (labels == 1)).cpu().sum()
            # TN predict 和 label 同时为0
            TN += ((pred == 0) & (labels == 0)).cpu().sum()
            # FN predict 0 label 1
            FN += ((pred == 0) & (labels == 1)).cpu().sum()
            # FP predict 1 label 0
            FP += ((pred == 1) & (labels == 0)).cpu().sum()

            num_true_pos += num_true_pos_
            num_predicted_pos += num_predicted_pos_
            num_real_pos += num_real_pos_
            num_words += torch.sum(masks)

        # 等待所有进程
        if device != torch.device('cpu'):
            torch.cuda.synchronize(device)

        num_true_pos=reduce_value(num_true_pos)
        num_predicted_pos = reduce_value(num_predicted_pos)

        num_real_pos = reduce_value(num_real_pos)
        num_words = reduce_value(num_words)

        micro_precision = num_true_pos / num_predicted_pos
        micro_recall = num_true_pos / num_real_pos
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision +
                                                           micro_recall)
        sparsity = num_predicted_pos / num_words

        # cls
        TP = reduce_value(TP)
        FP = reduce_value(FP)
        FN = reduce_value(FN)
        TN = reduce_value(TN)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * recall * precision / (recall + precision)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        print(
            "annotation dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(recall, precision,
                                                                                                         f1_score,
                                                                                                         accuracy))
    return sparsity, micro_precision, micro_recall, micro_f1

def validate_share_bert_onegpu(model, annotation_loader, device):
    num_true_pos = 0.
    num_predicted_pos = 0.
    num_real_pos = 0.
    num_words = 0

    TP = 0
    TN = 0
    FN = 0
    FP = 0
    with torch.no_grad():
        for (batch, (inputs, masks, labels,
                     annotations,special_masks)) in enumerate(annotation_loader):
            inputs, masks, labels, annotations,special_masks = inputs.to(device), masks.to(device), labels.to(device), annotations.to(
                device), special_masks.to(device)

            # rationales -- (batch_size, seq_length, 2)
            rationales, cls_logits = model(inputs, masks,special_masks)

            num_true_pos_, num_predicted_pos_, num_real_pos_ = compute_micro_stats(
                annotations, rationales)

            soft_pred = F.softmax(cls_logits, -1)
            _, pred = torch.max(soft_pred, dim=-1)

            # TP predict 和 label 同时为1
            TP += ((pred == 1) & (labels == 1)).cpu().sum()
            # TN predict 和 label 同时为0
            TN += ((pred == 0) & (labels == 0)).cpu().sum()
            # FN predict 0 label 1
            FN += ((pred == 0) & (labels == 1)).cpu().sum()
            # FP predict 1 label 0
            FP += ((pred == 1) & (labels == 0)).cpu().sum()

            num_true_pos += num_true_pos_
            num_predicted_pos += num_predicted_pos_
            num_real_pos += num_real_pos_
            num_words += torch.sum(masks)





        micro_precision = num_true_pos / num_predicted_pos
        micro_recall = num_true_pos / num_real_pos
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision +
                                                           micro_recall)
        sparsity = num_predicted_pos / num_words

        # cls


        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * recall * precision / (recall + precision)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        print(
            "annotation dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(recall, precision,
                                                                                                         f1_score,
                                                                                                         accuracy))
    return sparsity, micro_precision, micro_recall, micro_f1


def train_bert_classifier(model,opt_pred, dataset, device, args,mode):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    if mode=='train':
        model.module.train()
    else:
        model.module.eval()
    for (batch, data) in enumerate(dataset):
        if mode=='test':
            inputs, masks, labels,annotations, special_masks=data
        else:
            inputs, masks, labels, special_masks=data
        opt_pred.zero_grad()

        inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device), special_masks.to(
            device)
        if mode == 'train':
            logits=model.module.train_one_step(inputs, masks, special_masks)

            cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
            cls_loss.backward()
            opt_pred.step()
        else:
            with torch.no_grad():
                logits = model.module.train_one_step(inputs, masks, special_masks)



        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()


        # 等待所有进程
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)

    TP = reduce_value(TP)
    FP = reduce_value(FP)
    FN = reduce_value(FN)
    TN = reduce_value(TN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy

def train_bert_classifier_onegpu(model,opt_pred, dataset, device, args,mode):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    if mode=='train':
        model.train()
    else:
        model.eval()
    for (batch, data) in enumerate(dataset):
        if mode=='test':
            inputs, masks, labels,annotations, special_masks=data
        else:
            inputs, masks, labels, special_masks=data
        opt_pred.zero_grad()

        inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device), special_masks.to(
            device)
        if mode == 'train':
            logits=model.train_one_step(inputs, masks, special_masks)

            cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
            cls_loss.backward()
            opt_pred.step()
        else:
            with torch.no_grad():
                logits = model.train_one_step(inputs, masks, special_masks)



        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()






    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy

def train_bert_bcr_multigpu_distillation(model,classifier, opt_gen,opt_pred, dataset, device, args,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []

    for (batch, (inputs, masks, labels,special_masks)) in enumerate(dataset):
        opt_gen.zero_grad()
        opt_pred.zero_grad()

        inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device),special_masks.to(device)

        # train classification
        # get rationales
        rationales, rationales_add_special_token = model.module.get_rationale(inputs, masks, special_masks)
        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales, masks, args.sparsity_percentage)

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales)

        if args.gen_acc == 0:
            forward_logit = model.module.pred_forward_logit(inputs, masks, torch.detach(rationales_add_special_token))
        elif args.gen_acc==1:
            forward_logit = model.module.pred_forward_logit(inputs, masks, rationales_add_special_token)
        else:
            print('wrong gen acc')



        # detach_logit = model.module.detach_gen_pred(inputs, masks, rationales_add_special_token)
        # forward_logit = model.module.pred_forward_logit(inputs, masks, rationales_add_special_token)

        #full_text_logits
        with torch.no_grad():
            full_text_logits=classifier.train_one_step(inputs, masks,special_masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(forward_logit, labels)
        full_text_cls_loss=args.cls_lambda * F.cross_entropy(full_text_logits, labels)

        if args.gen_sparse==1:
            classification_loss=cls_loss+full_text_cls_loss+sparsity_loss + continuity_loss
        elif args.gen_sparse==0:
            classification_loss = cls_loss +  full_text_cls_loss
        else:
            print('gen sparse wrong')

        classification_loss.backward()

        opt_pred.step()
        opt_pred.zero_grad()

        # if args.gen_acc==1:
        #     opt_gen.step()
        #     opt_gen.zero_grad()
        # elif args.gen_sparse==1:
        #     opt_gen.step()
        #     opt_gen.zero_grad()
        # else:
        #     pass

        #train divergence
        # opt_gen.zero_grad()
        name1 = []
        name2 = []
        name3 = []
        for idx, p in model.module.pred_encoder.named_parameters():
            if p.requires_grad == True:
                name1.append(idx)
                p.requires_grad = False
        for idx, p in model.module.pred_fc.named_parameters():
            if p.requires_grad == True:
                name2.append(idx)
                p.requires_grad = False
        for idx, p in model.module.layernorm2.named_parameters():
            if p.requires_grad == True:
                name3.append(idx)
                p.requires_grad = False

        # rationales, rationales_add_special_token = model.module.get_rationale(inputs, masks, special_masks)
        forward_logit = model.module.pred_forward_logit(inputs, masks, rationales_add_special_token)



        #jsd
        if args.div == 'js':
            jsd_func = JS_DIV()
            jsd_loss = jsd_func(forward_logit, full_text_logits)
        elif args.div == 'kl':
            jsd_loss = nn.functional.kl_div(F.softmax(forward_logit, dim=-1).log(), F.softmax(full_text_logits, dim=-1),
                                            reduction='batchmean')
        else:
            print('div wrong')



        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales, masks, args.sparsity_percentage)


        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales)

        gen_loss = sparsity_loss + continuity_loss + jsd_loss
        # update gradient

        gen_loss.backward()
        opt_gen.step()
        opt_gen.zero_grad()
        n1 = 0
        n2 = 0
        n3 = 0
        for idx,p in model.module.pred_encoder.named_parameters():
            if idx in name1:
                p.requires_grad=True
                n1+=1
        for idx,p in model.module.pred_fc.named_parameters():
            if idx in name2:
                p.requires_grad = True
                n2 += 1
        for idx,p in model.module.layernorm2.named_parameters():
            if idx in name3:
                p.requires_grad = True
                n3 += 1





        cls_soft_logits = torch.softmax(forward_logit, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    # 等待所有进程
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)


    TP=reduce_value(TP)
    FP = reduce_value(FP)
    FN = reduce_value(FN)
    TN = reduce_value(TN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)


    return precision, recall, f1_score, accuracy


def train_bert_bcr_onegpu_distillation(model,classifier, opt_gen,opt_pred, dataset, device, args,writer_epoch,annotation_loader):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    model.train()
    data_size=len(dataset)
    check_size=int(data_size/4)
    for (batch, (inputs, masks, labels,special_masks)) in enumerate(dataset):
        opt_gen.zero_grad()
        opt_pred.zero_grad()

        inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device),special_masks.to(device)

        # train classification
        # get rationales
        rationales, rationales_add_special_token = model.get_rationale(inputs, masks, special_masks)
        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales, masks, args.sparsity_percentage)

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales)

        if args.gen_acc == 0:
            forward_logit = model.pred_forward_logit(inputs, masks, torch.detach(rationales_add_special_token))
        elif args.gen_acc==1:
            forward_logit = model.pred_forward_logit(inputs, masks, rationales_add_special_token)
        else:
            print('wrong gen acc')




        #full_text_logits
        with torch.no_grad():
            full_text_logits=classifier.train_one_step(inputs, masks,special_masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(forward_logit, labels)
        full_text_cls_loss=args.cls_lambda * F.cross_entropy(full_text_logits, labels)

        if args.gen_sparse==1:
            classification_loss=cls_loss+sparsity_loss + continuity_loss
        elif args.gen_sparse==0:
            classification_loss = cls_loss
        else:
            print('gen sparse wrong')

        classification_loss.backward()

        opt_pred.step()
        opt_pred.zero_grad()

        if args.gen_acc==1:
            opt_gen.step()
            opt_gen.zero_grad()
        elif args.gen_sparse==1:
            opt_gen.step()
            opt_gen.zero_grad()
            rationales, rationales_add_special_token = model.get_rationale(inputs, masks, special_masks)
            sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
                rationales, masks, args.sparsity_percentage)

            continuity_loss = args.continuity_lambda * get_continuity_loss(
                rationales)
        else:
            pass

        #train divergence
        # opt_gen.zero_grad()

        if args.pred_div==0:
            name1 = []
            name2 = []
            name3 = []
            for idx, p in model.pred_encoder.named_parameters():
                if p.requires_grad == True:
                    name1.append(idx)
                    p.requires_grad = False
            for idx, p in model.pred_fc.named_parameters():
                if p.requires_grad == True:
                    name2.append(idx)
                    p.requires_grad = False
            for idx, p in model.layernorm2.named_parameters():
                if p.requires_grad == True:
                    name3.append(idx)
                    p.requires_grad = False

        # rationales, rationales_add_special_token = model.module.get_rationale(inputs, masks, special_masks)
        forward_logit = model.pred_forward_logit(inputs, masks, rationales_add_special_token)



        #jsd
        if args.div == 'js':
            jsd_func = JS_DIV()
            jsd_loss = jsd_func(forward_logit, full_text_logits)
        elif args.div == 'kl':
            jsd_loss = nn.functional.kl_div(F.softmax(forward_logit, dim=-1).log(), F.softmax(full_text_logits, dim=-1),
                                            reduction='batchmean')
        else:
            print('div wrong')





        gen_loss = sparsity_loss + continuity_loss + jsd_loss
        # update gradient

        gen_loss.backward()
        opt_gen.step()
        opt_gen.zero_grad()
        if args.pred_div==1:
            opt_pred.step()
            opt_pred.zero_grad()
        if args.pred_div == 0:
            n1 = 0
            n2 = 0
            n3 = 0
            for idx,p in model.pred_encoder.named_parameters():
                if idx in name1:
                    p.requires_grad=True
                    n1+=1
            for idx,p in model.pred_fc.named_parameters():
                if idx in name2:
                    p.requires_grad = True
                    n2 += 1
            for idx,p in model.layernorm2.named_parameters():
                if idx in name3:
                    p.requires_grad = True
                    n3 += 1





        cls_soft_logits = torch.softmax(forward_logit, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

        if (batch+1)%check_size==0:
            model.eval()
            annotation_results = validate_share_bert_onegpu(model, annotation_loader, device)
            print("check with {}".format(batch))
            print(
                "The annotation performance: sparsity: %.4f, precision: %.4f, recall: %.4f, f1: %.4f"
                % (100 * annotation_results[0], 100 * annotation_results[1],
                   100 * annotation_results[2], 100 * annotation_results[3]))
            model.train()






    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)


    return precision, recall, f1_score, accuracy



