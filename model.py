import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import transformers as ppb
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class Embedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, pretrained_embedding=None):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embedding is not None:
            self.embedding.weight.data = torch.from_numpy(pretrained_embedding)
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        """
        Inputs:
        x -- (batch_size, seq_length)
        Outputs
        shape -- (batch_size, seq_length, embedding_dim)
        """
        return self.embedding(x)





class Sp_norm_model(nn.Module):         #给predictor的encoder和linear加了sp norm 去掉了layer norm
    def __init__(self, args):
        super(Sp_norm_model, self).__init__()
        self.lay=True
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        self.gen = nn.GRU(input_size=args.embedding_dim,
                                  hidden_size=args.hidden_dim // 2,
                                  num_layers=args.num_layers,
                                  batch_first=True,
                                  bidirectional=True)
        self.cls = nn.GRU(input_size=args.embedding_dim,
                          hidden_size=args.hidden_dim // 2,
                          num_layers=args.num_layers,
                          batch_first=True,
                          bidirectional=True)
        if args.sp_norm==1:
            # self.cls = spectral_norm(spectral_norm(spectral_norm(spectral_norm(nn.GRU(input_size=args.embedding_dim,
            #                       hidden_size=args.hidden_dim // 2,
            #                       num_layers=args.num_layers,
            #                       batch_first=True,
            #                       bidirectional=True),name="weight_ih_l0")
            #                                                      ,name="weight_ih_l0_reverse"),name="weight_hh_l0"),name="weight_hh_l0_reverse")
            self.cls_fc = spectral_norm(nn.Linear(args.hidden_dim, args.num_class))
        elif args.sp_norm==0:
            self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)
        else:
            print('wrong norm')
        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.layernorm2 = nn.LayerNorm(args.hidden_dim)
        self.generator=nn.Sequential(self.gen,
                                     SelectItem(0),
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)

    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = self._independent_soft_sampling(rationale_logits)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def forward(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_logits=self.generator(embedding)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits

    def train_one_step(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        # (batch_size, seq_length, embedding_dim)
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.cls(embedding)  # (batch_size, seq_length, hidden_dim)
        outputs = outputs * masks_ + (1. -
                                      masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        # shape -- (batch_size, num_classes)
        logits = self.cls_fc(self.dropout(outputs))
        return logits

    def grad(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_logits=self.generator(embedding)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        ########## Classifier ##########
        embedding2=embedding.clone().detach()
        embedding2.requires_grad=True
        cls_embedding =embedding2  * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits,embedding2,cls_embedding

    def g_skew(self,inputs, masks):
        #  masks_ (batch_size, seq_length, 1)
        masks_ = masks.unsqueeze(-1)
        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        gen_output = self.layernorm1(gen_output)
        gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        soft_log=self._independent_soft_sampling(gen_logits)
        return soft_log


class Bert_rnp(nn.Module):
    def __init__(self, args):
        super(Bert_rnp, self).__init__()
        self.lay=False
        self.args = args
        if args.cell_type=='bert':
            self.gen_encoder=ppb.BertModel.from_pretrained('bert-base-uncased')
            self.pred_encoder = Rationale_Bert.from_pretrained('bert-base-uncased')
        elif args.cell_type=='electra':
            self.gen_encoder = ppb.ElectraModel.from_pretrained("google/electra-small-discriminator")
            self.pred_encoder = Rationale_Electra.from_pretrained("google/electra-small-discriminator")
        elif args.cell_type=='roberta':
            self.gen_encoder = ppb.RobertaModel.from_pretrained("roberta-base")
            self.pred_encoder = Rationale_Roberta.from_pretrained("roberta-base")
        else:
            print('not defined model type')
        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)        #bert:768, electra_small 256
        self.pred_fc=nn.Linear(args.hidden_dim, 2)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.layernorm2 = nn.LayerNorm(args.hidden_dim)
        self.generator_linear=nn.Sequential(
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)
        freeze_para=['embeddings']              #freeze the word embedding
        for name,para in self.gen_encoder.named_parameters():
            for ele in freeze_para:
                if ele in name:
                    para.requires_grad=False
                    print('freeze the generator embedding')
        for name,para in self.pred_encoder.named_parameters():
            for ele in freeze_para:
                if ele in name:
                    para.requires_grad=False
                    print('freeze the predictor embedding')
        if args.freeze_bert==1:                 #固定整个bert
            for name, para in self.gen_encoder.named_parameters():
                para.requires_grad = False
            for name, para in self.pred_encoder.named_parameters():
                para.requires_grad = False

        print('layernorm2={}'.format(self.lay))
    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        # z = self._independent_soft_sampling(rationale_logits)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def forward(self, inputs, masks,special_masks):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        # embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        # gen_logits=self.generator(embedding)
        # gen_logits=self.generator((inputs,masks))
        gen_bert_out=self.gen_encoder(inputs,masks)[0]
        gen_logits=self.generator_linear(gen_bert_out)

        ########## Sample ##########
        rationale_z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        z_add_special_tokens=torch.max(rationale_z[:,:,1],special_masks)     #cls和sep恒为1

        ########## Classifier ##########
        # inputs=inputs*rationale_z[:,:,1].int()
        # cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        # cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        pred_output=self.pred_encoder.forward_predictor(inputs,masks,rationale_mask=z_add_special_tokens)[0]
        if self.lay:
            pred_output=self.layernorm2(pred_output)
        # pred_output = self.pred_encoder(inputs, rationale_z[:,:,1].int())[0]
        cls_outputs = pred_output * masks_ + (1. -
                                              masks_) * (-1e6)
        # maxpool_z=rationale_z.clone().detach()
        # cls_outputs=cls_outputs*maxpool_z[:,:,1].unsqueeze(-1)+(1.-maxpool_z[:,:,1].unsqueeze(-1))*(-1e6)
        # # # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.pred_fc(self.dropout(cls_outputs))
        return rationale_z[:,:,1], cls_logits

    def get_rationale(self, inputs, masks,special_masks):

        gen_bert_out = self.gen_encoder(inputs, masks)[0]
        gen_logits = self.generator_linear(gen_bert_out)

        ########## Sample ##########
        rationale_z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        z_add_special_tokens = torch.max(rationale_z[:, :, 1], special_masks)  # cls和sep恒为1
        return rationale_z[:, :, 1], z_add_special_tokens

    def detach_gen_pred(self, inputs, masks,z_add_special_tokens):
        masks_ = masks.unsqueeze(-1)
        z_detach=z_add_special_tokens.detach()
        pred_output = self.pred_encoder.forward_predictor(inputs, masks, rationale_mask=z_detach)[0]
        if self.lay:
            pred_output = self.layernorm2(pred_output)
        # pred_output = self.pred_encoder(inputs, rationale_z[:,:,1].int())[0]
        cls_outputs = pred_output * masks_ + (1. -
                                              masks_) * (-1e6)
        # maxpool_z=rationale_z.clone().detach()
        # cls_outputs=cls_outputs*maxpool_z[:,:,1].unsqueeze(-1)+(1.-maxpool_z[:,:,1].unsqueeze(-1))*(-1e6)
        # # # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.pred_fc(self.dropout(cls_outputs))

        return cls_logits

    def pred_forward_logit(self, inputs, masks,z_add_special_tokens):
        masks_ = masks.unsqueeze(-1)
        pred_output = self.pred_encoder.forward_predictor(inputs, masks, rationale_mask=z_add_special_tokens)[0]
        if self.lay:
            pred_output = self.layernorm2(pred_output)
        # pred_output = self.pred_encoder(inputs, rationale_z[:,:,1].int())[0]
        cls_outputs = pred_output * masks_ + (1. -
                                              masks_) * (-1e6)
        # maxpool_z=rationale_z.clone().detach()
        # cls_outputs=cls_outputs*maxpool_z[:,:,1].unsqueeze(-1)+(1.-maxpool_z[:,:,1].unsqueeze(-1))*(-1e6)
        # # # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.pred_fc(self.dropout(cls_outputs))

        return cls_logits

    def train_one_step(self, inputs, masks,special_masks):
        masks_ = masks.unsqueeze(-1)


        pred_output = self.pred_encoder(inputs, masks)[0]
        if self.lay:
            pred_output = self.layernorm2(pred_output)
        # pred_output = self.pred_encoder(inputs, rationale_z[:,:,1].int())[0]
        cls_outputs = pred_output * masks_ + (1. -
                                              masks_) * (-1e6)
        # maxpool_z=rationale_z.clone().detach()
        # cls_outputs=cls_outputs*maxpool_z[:,:,1].unsqueeze(-1)+(1.-maxpool_z[:,:,1].unsqueeze(-1))*(-1e6)
        # # # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.pred_fc(self.dropout(cls_outputs))
        return cls_logits

    def grad(self, inputs, masks,special_masks):
        # masks_ = masks.unsqueeze(-1)
        #
        # ########## Genetator ##########
        # embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # # if self.lay:
        # #     gen_output = self.layernorm1(gen_output)
        # # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        # gen_logits=self.generator(embedding)
        # ########## Sample ##########
        # z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        # ########## Classifier ##########
        # embedding2=embedding.clone().detach()
        # embedding2.requires_grad=True
        # cls_embedding =embedding2  * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        # cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        # cls_outputs = cls_outputs * masks_ + (1. -
        #                                       masks_) * (-1e6)
        # # (batch_size, hidden_dim, seq_length)
        # cls_outputs = torch.transpose(cls_outputs, 1, 2)
        # cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # # shape -- (batch_size, num_classes)
        # cls_logits = self.cls_fc(self.dropout(cls_outputs))

        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        # embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        # gen_logits=self.generator(embedding)
        # gen_logits=self.generator((inputs,masks))
        gen_bert_out = self.gen_encoder(inputs, masks)[0]
        gen_logits = self.generator_linear(gen_bert_out)

        ########## Sample ##########
        rationale_z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        z_add_special_tokens = torch.max(rationale_z[:, :, 1], special_masks)  # cls和sep恒为1

        embedding2=self.pred_encoder.get_rationale_word_embedding(inputs, masks, rationale_mask=z_add_special_tokens).clone().detach()
        # print(embedding2.shape)
        # print(z_add_special_tokens.shape)
        embedding2.requires_grad = True

        cls_embedding = embedding2 * z_add_special_tokens.unsqueeze(-1)

        ########## Classifier ##########
        # inputs=inputs*rationale_z[:,:,1].int()
        # cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        # cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        pred_output = self.pred_encoder.pred_use_embed(inputs, masks, rationale_mask=z_add_special_tokens,rationale_embed=cls_embedding)[0]
        if self.lay:
            pred_output = self.layernorm2(pred_output)
        # pred_output = self.pred_encoder(inputs, rationale_z[:,:,1].int())[0]
        cls_outputs = pred_output * masks_ + (1. -
                                              masks_) * (-1e6)
        # maxpool_z=rationale_z.clone().detach()
        # cls_outputs=cls_outputs*maxpool_z[:,:,1].unsqueeze(-1)+(1.-maxpool_z[:,:,1].unsqueeze(-1))*(-1e6)
        # # # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.pred_fc(self.dropout(cls_outputs))
        # return rationale_z[:, :, 1], cls_logits
        return rationale_z, cls_logits,embedding2,cls_embedding


class Bert_FR(nn.Module):
    def __init__(self, args):
        super(Bert_FR, self).__init__()
        self.lay=True
        self.args = args
        if args.cell_type=='bert':
            self.gen_encoder=Rationale_Bert.from_pretrained('bert-base-uncased')
            # self.pred_encoder = Rationale_Bert.from_pretrained('bert-base-uncased')
        elif args.cell_type=='electra':
            self.gen_encoder = Rationale_Electra.from_pretrained("google/electra-small-discriminator")
            # self.pred_encoder = Rationale_Electra.from_pretrained("google/electra-small-discriminator")
        elif args.cell_type=='roberta':
            # self.gen_encoder = ppb.RobertaModel.from_pretrained("roberta-base")
            self.gen_encoder = Rationale_Roberta.from_pretrained("roberta-base")
        else:
            print('not defined model type')
        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)        #bert:768, electra_small 256
        self.pred_fc=nn.Linear(args.hidden_dim, 2)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.layernorm2 = nn.LayerNorm(args.hidden_dim)
        self.generator_linear=nn.Sequential(
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)
        freeze_para=['embeddings']              #freeze the word embedding
        for name,para in self.gen_encoder.named_parameters():
            for ele in freeze_para:
                if ele in name:
                    para.requires_grad=False
                    print('freeze the generator embedding')
        # for name,para in self.pred_encoder.named_parameters():
        #     for ele in freeze_para:
        #         if ele in name:
        #             para.requires_grad=False
        #             print('freeze the predictor embedding')
        if args.freeze_bert==1:                 #固定整个bert
            for name, para in self.gen_encoder.named_parameters():
                para.requires_grad = False
            # for name, para in self.pred_encoder.named_parameters():
            #     para.requires_grad = False
    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        # z = self._independent_soft_sampling(rationale_logits)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def forward(self, inputs, masks,special_masks):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        # embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        # gen_logits=self.generator(embedding)
        # gen_logits=self.generator((inputs,masks))
        gen_bert_out=self.gen_encoder(inputs,masks)[0]
        gen_logits=self.generator_linear(gen_bert_out)

        ########## Sample ##########
        rationale_z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        z_add_special_tokens=torch.max(rationale_z[:,:,1],special_masks)     #cls和sep恒为1

        ########## Classifier ##########
        # inputs=inputs*rationale_z[:,:,1].int()
        # cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        # cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        pred_output=self.gen_encoder.forward_predictor(inputs,masks,rationale_mask=z_add_special_tokens)[0]
        pred_output=self.layernorm2(pred_output)
        # pred_output = self.pred_encoder(inputs, rationale_z[:,:,1].int())[0]
        cls_outputs = pred_output * masks_ + (1. -
                                              masks_) * (-1e6)
        # maxpool_z=rationale_z.clone().detach()
        # cls_outputs=cls_outputs*maxpool_z[:,:,1].unsqueeze(-1)+(1.-maxpool_z[:,:,1].unsqueeze(-1))*(-1e6)
        # # # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.pred_fc(self.dropout(cls_outputs))
        return rationale_z[:,:,1], cls_logits

    def get_rationale(self, inputs, masks,special_masks):

        gen_bert_out = self.gen_encoder(inputs, masks)[0]
        gen_logits = self.generator_linear(gen_bert_out)

        ########## Sample ##########
        rationale_z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        z_add_special_tokens = torch.max(rationale_z[:, :, 1], special_masks)  # cls和sep恒为1
        return rationale_z[:, :, 1], z_add_special_tokens

    def pred_forward_logit(self, inputs, masks,z_add_special_tokens):
        masks_ = masks.unsqueeze(-1)
        pred_output = self.gen_encoder.forward_predictor(inputs, masks, rationale_mask=z_add_special_tokens)[0]
        if self.lay:
            pred_output = self.layernorm2(pred_output)
        # pred_output = self.pred_encoder(inputs, rationale_z[:,:,1].int())[0]
        cls_outputs = pred_output * masks_ + (1. -
                                              masks_) * (-1e6)
        # maxpool_z=rationale_z.clone().detach()
        # cls_outputs=cls_outputs*maxpool_z[:,:,1].unsqueeze(-1)+(1.-maxpool_z[:,:,1].unsqueeze(-1))*(-1e6)
        # # # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.pred_fc(self.dropout(cls_outputs))

        return cls_logits





class Bert_rnp_0812(nn.Module):
    def __init__(self, args):
        super(Bert_rnp, self).__init__()
        self.lay=False
        self.args = args
        if args.cell_type=='bert':
            self.gen_encoder=ppb.BertModel.from_pretrained('bert-base-uncased')
            self.pred_encoder = Rationale_Bert.from_pretrained('bert-base-uncased')
        elif args.cell_type=='electra':
            self.gen_encoder = ppb.ElectraModel.from_pretrained("google/electra-small-discriminator")
            self.pred_encoder = Rationale_Electra.from_pretrained("google/electra-small-discriminator")
        elif args.cell_type=='roberta':
            self.gen_encoder = ppb.RobertaModel.from_pretrained("roberta-base")
            self.pred_encoder = Rationale_Roberta.from_pretrained("roberta-base")
        else:
            print('not defined model type')
        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)        #bert:768, electra_small 256
        self.pred_fc=nn.Linear(args.hidden_dim, 2)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.layernorm2 = nn.LayerNorm(args.hidden_dim)
        self.generator_linear=nn.Sequential(
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)
        freeze_para=['embeddings']              #freeze the word embedding
        for name,para in self.gen_encoder.named_parameters():
            for ele in freeze_para:
                if ele in name:
                    para.requires_grad=False
                    print('freeze the generator embedding')
        for name,para in self.pred_encoder.named_parameters():
            for ele in freeze_para:
                if ele in name:
                    para.requires_grad=False
                    print('freeze the predictor embedding')
        if args.freeze_bert==1:                 #固定整个bert
            for name, para in self.gen_encoder.named_parameters():
                para.requires_grad = False
            for name, para in self.pred_encoder.named_parameters():
                para.requires_grad = False

        print('layernorm2={}'.format(self.lay))
    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        # z = self._independent_soft_sampling(rationale_logits)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def forward(self, inputs, masks,special_masks):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        # embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        # gen_logits=self.generator(embedding)
        # gen_logits=self.generator((inputs,masks))
        gen_bert_out=self.gen_encoder(inputs,masks)[0]
        gen_logits=self.generator_linear(gen_bert_out)

        ########## Sample ##########
        rationale_z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        z_add_special_tokens=torch.max(rationale_z[:,:,1],special_masks)     #cls和sep恒为1

        ########## Classifier ##########
        # inputs=inputs*rationale_z[:,:,1].int()
        # cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        # cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        pred_output=self.pred_encoder.forward_predictor(inputs,masks,rationale_mask=z_add_special_tokens)[0]
        if self.lay:
            pred_output=self.layernorm2(pred_output)
        # pred_output = self.pred_encoder(inputs, rationale_z[:,:,1].int())[0]
        cls_outputs = pred_output * masks_ + (1. -
                                              masks_) * (-1e6)
        # maxpool_z=rationale_z.clone().detach()
        # cls_outputs=cls_outputs*maxpool_z[:,:,1].unsqueeze(-1)+(1.-maxpool_z[:,:,1].unsqueeze(-1))*(-1e6)
        # # # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.pred_fc(self.dropout(cls_outputs))
        return rationale_z[:,:,1], cls_logits

    def get_rationale(self, inputs, masks,special_masks):

        gen_bert_out = self.gen_encoder(inputs, masks)[0]
        gen_logits = self.generator_linear(gen_bert_out)

        ########## Sample ##########
        rationale_z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        z_add_special_tokens = torch.max(rationale_z[:, :, 1], special_masks)  # cls和sep恒为1
        return rationale_z[:, :, 1], z_add_special_tokens

    def detach_gen_pred(self, inputs, masks,z_add_special_tokens):
        masks_ = masks.unsqueeze(-1)
        z_detach=z_add_special_tokens.detach()
        pred_output = self.pred_encoder.forward_predictor(inputs, masks, rationale_mask=z_detach)[0]
        if self.lay:
            pred_output = self.layernorm2(pred_output)
        # pred_output = self.pred_encoder(inputs, rationale_z[:,:,1].int())[0]
        cls_outputs = pred_output * masks_ + (1. -
                                              masks_) * (-1e6)
        # maxpool_z=rationale_z.clone().detach()
        # cls_outputs=cls_outputs*maxpool_z[:,:,1].unsqueeze(-1)+(1.-maxpool_z[:,:,1].unsqueeze(-1))*(-1e6)
        # # # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.pred_fc(self.dropout(cls_outputs))

        return cls_logits

    def pred_forward_logit(self, inputs, masks,z_add_special_tokens):
        masks_ = masks.unsqueeze(-1)
        pred_output = self.pred_encoder.forward_predictor(inputs, masks, rationale_mask=z_add_special_tokens)[0]
        if self.lay:
            pred_output = self.layernorm2(pred_output)
        # pred_output = self.pred_encoder(inputs, rationale_z[:,:,1].int())[0]
        cls_outputs = pred_output * masks_ + (1. -
                                              masks_) * (-1e6)
        # maxpool_z=rationale_z.clone().detach()
        # cls_outputs=cls_outputs*maxpool_z[:,:,1].unsqueeze(-1)+(1.-maxpool_z[:,:,1].unsqueeze(-1))*(-1e6)
        # # # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.pred_fc(self.dropout(cls_outputs))

        return cls_logits

    def train_one_step(self, inputs, masks,special_masks):
        masks_ = masks.unsqueeze(-1)


        pred_output = self.pred_encoder(inputs, masks)[0]
        if self.lay:
            pred_output = self.layernorm2(pred_output)
        # pred_output = self.pred_encoder(inputs, rationale_z[:,:,1].int())[0]
        cls_outputs = pred_output * masks_ + (1. -
                                              masks_) * (-1e6)
        # maxpool_z=rationale_z.clone().detach()
        # cls_outputs=cls_outputs*maxpool_z[:,:,1].unsqueeze(-1)+(1.-maxpool_z[:,:,1].unsqueeze(-1))*(-1e6)
        # # # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.pred_fc(self.dropout(cls_outputs))
        return cls_logits




class Bert_classfier(nn.Module):
    def __init__(self, args):
        super(Bert_classfier, self).__init__()
        self.lay=False
        self.args = args
        if args.cell_type=='bert':
            self.pred_encoder = Rationale_Bert.from_pretrained('bert-base-uncased')
        elif args.cell_type=='electra':
            self.pred_encoder = Rationale_Electra.from_pretrained("google/electra-small-discriminator")
        elif args.cell_type=='roberta':
            self.pred_encoder = Rationale_Roberta.from_pretrained("roberta-base")
        else:
            print('not defined model type')
        self.z_dim = 2
        self.pred_fc=nn.Linear(args.hidden_dim, 2)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm2 = nn.LayerNorm(args.hidden_dim)
        freeze_para=['embeddings']              #freeze the word embedding

        for name,para in self.pred_encoder.named_parameters():
            for ele in freeze_para:
                if ele in name:
                    para.requires_grad=False
                    print('freeze the predictor embedding')
        if args.freeze_bert==1:                 #固定整个bert
            for name, para in self.pred_encoder.named_parameters():
                para.requires_grad = False

        print('layernorm={}'.format(self.lay))


    def get_rationale(self, inputs, masks,special_masks):

        gen_bert_out = self.gen_encoder(inputs, masks)[0]
        gen_logits = self.generator_linear(gen_bert_out)

        ########## Sample ##########
        rationale_z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        z_add_special_tokens = torch.max(rationale_z[:, :, 1], special_masks)  # cls和sep恒为1
        return rationale_z[:, :, 1], z_add_special_tokens



    def pred_forward_logit(self, inputs, masks,z_add_special_tokens):
        masks_ = masks.unsqueeze(-1)
        pred_output = self.pred_encoder.forward_predictor(inputs, masks, rationale_mask=z_add_special_tokens)[0]
        if self.lay:
            pred_output = self.layernorm2(pred_output)
        # pred_output = self.pred_encoder(inputs, rationale_z[:,:,1].int())[0]
        cls_outputs = pred_output * masks_ + (1. -
                                              masks_) * (-1e6)
        # maxpool_z=rationale_z.clone().detach()
        # cls_outputs=cls_outputs*maxpool_z[:,:,1].unsqueeze(-1)+(1.-maxpool_z[:,:,1].unsqueeze(-1))*(-1e6)
        # # # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.pred_fc(self.dropout(cls_outputs))

        return cls_logits

    def train_one_step(self, inputs, masks,special_masks):
        masks_ = masks.unsqueeze(-1)


        pred_output = self.pred_encoder(inputs, masks)[0]
        if self.lay:
            pred_output = self.layernorm2(pred_output)
        # pred_output = self.pred_encoder(inputs, rationale_z[:,:,1].int())[0]
        cls_outputs = pred_output * masks_ + (1. -
                                              masks_) * (-1e6)
        # maxpool_z=rationale_z.clone().detach()
        # cls_outputs=cls_outputs*maxpool_z[:,:,1].unsqueeze(-1)+(1.-maxpool_z[:,:,1].unsqueeze(-1))*(-1e6)
        # # # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.pred_fc(self.dropout(cls_outputs))
        return cls_logits








class Rationale_Electra(ppb.ElectraModel):
    def __init__(self,config):
        super(Rationale_Electra, self).__init__(config)

    def forward_predictor(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            rationale_mask=None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        hidden_states = hidden_states * rationale_mask.unsqueeze(-1)

        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)

        hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return hidden_states

    def get_rationale_word_embedding(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            rationale_mask=None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        # hidden_states = hidden_states * rationale_mask.unsqueeze(-1)
        #
        # if hasattr(self, "embeddings_project"):
        #     hidden_states = self.embeddings_project(hidden_states)
        #
        # hidden_states = self.encoder(
        #     hidden_states,
        #     attention_mask=extended_attention_mask,
        #     head_mask=head_mask,
        #     encoder_hidden_states=encoder_hidden_states,
        #     encoder_attention_mask=encoder_extended_attention_mask,
        #     past_key_values=past_key_values,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        return hidden_states


    def pred_use_embed(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            rationale_mask=None,
            rationale_embed=None
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # hidden_states = self.embeddings(
        #     input_ids=input_ids,
        #     position_ids=position_ids,
        #     token_type_ids=token_type_ids,
        #     inputs_embeds=inputs_embeds,
        #     past_key_values_length=past_key_values_length,
        # )

        hidden_states = rationale_embed * rationale_mask.unsqueeze(-1)

        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)

        hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return hidden_states

class Rationale_Bert(ppb.BertModel):
    r"""
            forward_predictor:人工将embedding输出后mask了
            后面两个函数用来求利普希茨常数
            """
    def __init__(self,config, add_pooling_layer=True):
        super(Rationale_Bert, self).__init__(config, add_pooling_layer=True)

    def forward_predictor(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            rationale_mask=None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        这是我写的predictor
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        embedding_output = embedding_output * rationale_mask.unsqueeze(-1)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


    def get_rationale_word_embedding(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            rationale_mask=None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        # hidden_states = hidden_states * rationale_mask.unsqueeze(-1)
        #
        # if hasattr(self, "embeddings_project"):
        #     hidden_states = self.embeddings_project(hidden_states)
        #
        # hidden_states = self.encoder(
        #     hidden_states,
        #     attention_mask=extended_attention_mask,
        #     head_mask=head_mask,
        #     encoder_hidden_states=encoder_hidden_states,
        #     encoder_attention_mask=encoder_extended_attention_mask,
        #     past_key_values=past_key_values,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        return hidden_states


    def pred_use_embed(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            rationale_mask=None,
            rationale_embed=None
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # hidden_states = self.embeddings(
        #     input_ids=input_ids,
        #     position_ids=position_ids,
        #     token_type_ids=token_type_ids,
        #     inputs_embeds=inputs_embeds,
        #     past_key_values_length=past_key_values_length,
        # )

        hidden_states = rationale_embed * rationale_mask.unsqueeze(-1)

        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)

        hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return hidden_states

class Rationale_Roberta(ppb.RobertaModel):
    def __init__(self,config, add_pooling_layer=True):
        super(Rationale_Roberta, self).__init__(config, add_pooling_layer=True)

    def forward_predictor(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        rationale_mask=None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        r"""
        人工将embedding输出后mask了
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        #人工添加
        embedding_output = embedding_output * rationale_mask.unsqueeze(-1)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

# Rationale_Electra.from_pretrained("google/electra-small-discriminator")




