import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask #added model_eval_multitask


TQDM_DISABLE=True

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5
SIZE_LAYER_1 = 384
SIZE_LAYER_2 = 576


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        #sentiment 
        self.py_sequential = torch.nn.Sequential(torch.nn.Linear(BERT_HIDDEN_SIZE, SIZE_LAYER_1), nn.ReLU(), torch.nn.Linear(SIZE_LAYER_1, N_SENTIMENT_CLASSES))
        self.paraphrase_classifier = torch.nn.Sequential(torch.nn.Linear(BERT_HIDDEN_SIZE * 2, BERT_HIDDEN_SIZE), nn.ReLU(), torch.nn.Linear(BERT_HIDDEN_SIZE, 1))
        #self.similarity = torch.nn.Linear(BERT_HIDDEN_SIZE * 2, 1) # read paper
        self.similarity_projection_layer1 = torch.nn.Sequential(torch.nn.Linear(BERT_HIDDEN_SIZE, SIZE_LAYER_2), nn.ReLU(), torch.nn.Linear(SIZE_LAYER_2, SIZE_LAYER_1))
        self.similarity_projection_layer2 = torch.nn.Sequential(torch.nn.Linear(BERT_HIDDEN_SIZE, SIZE_LAYER_2), nn.ReLU(), torch.nn.Linear(SIZE_LAYER_2, SIZE_LAYER_1))



    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        outputs = self.bert(input_ids, attention_mask) # pretrained bert embeddings
        embeddings = self.dropout(outputs['pooler_output'])# BERT embeddings

        return embeddings
        

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        embeddings = self.forward(input_ids, attention_mask)
        #make sure sizes align 
        #intermediete = self.sentiment_classifier_layer2(embeddings)
        logits = self.py_sequential(embeddings)
        return logits



    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        embeddings1 = self.forward(input_ids_1, attention_mask_1)
        embeddings2 = self.forward(input_ids_2, attention_mask_2)
        logit = self.paraphrase_classifier(torch.cat((embeddings1, embeddings2), dim=-1))
        return logit
        


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO cosine similarity as an extension here
        embeddings1 = self.forward(input_ids_1, attention_mask_1)
        embeddings2 = self.forward(input_ids_2, attention_mask_2)
        # put two embeddings into two layers
        #add pytorch sequential twize for both embeddings
        embeddings1 = self.similarity_projection_layer1(embeddings1)
        embeddings2 = self.similarity_projection_layer2(embeddings2)
        sim_score = F.cosine_similarity(embeddings1, embeddings2)
        sim_score = (sim_score + 1) * 2.5
        #sim_score = torch.tensor(sim_score, requires_grad=True)
        return sim_score
    



def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


## Currently only trains on sst dataset
# modify the train_multitask
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    #SST DATASET
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
                                    
                                    
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)
                                    

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    # lr = args.lr
    
    # extension 1: layer-wise learning rate decay
    lr = args.layer_learning_rate[0]
    lr_group = [lr * pow(args.layer_learning_rate_decay, 11 - i) for i in range(12)]
    groups = [(f'layers.{i}.', lr * pow(args.layer_learning_rate_decay, 11 - i)) for i in range(12)]
    parameters = []
    
    layer_names = []
    for idx, (name, param) in enumerate(model.named_parameters()):
        layer_names.append(name)
        
    parameters = []
    
    next_num = 1

    # store params & learning rates
    for idx, name in enumerate(layer_names):
        
        # display info
        
        if str(next_num) in name:
            next_num += 1
        
        #print(f'{idx}: lr = {lr_group[next_num - 1]:.6f}, {name}')

        # append layer parameters
        parameters += [{'params': [p for n, p in model.named_parameters() if n == name],
                        'lr':     lr_group[next_num - 1]}]
    
    
    # extension 1 done
    
    optimizer = AdamW(parameters)
    best_dev_acc = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch1, batch2, batch3 in tqdm(zip(sst_train_dataloader, para_train_dataloader, sts_train_dataloader), desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids_sst, b_mask_sst, b_labels_sst = (batch1['token_ids'],
                                       batch1['attention_mask'], batch1['labels'])

            b_ids_sst = b_ids_sst.to(device)
            b_mask_sst = b_mask_sst.to(device)
            b_labels_sst = b_labels_sst.to(device)

            optimizer.zero_grad()
            logits_sst = model.predict_sentiment(b_ids_sst, b_mask_sst)
            loss1 = F.cross_entropy(logits_sst, b_labels_sst.view(-1), reduction='sum') / args.batch_size
            
            b_ids_para, b_ids2_para, b_mask_para, b_mask2_para, b_labels_para = (batch2['token_ids_1'], batch2['token_ids_2'],
                                       batch2['attention_mask_1'], batch2['attention_mask_2'], batch2['labels'])

            b_ids_para = b_ids_para.to(device)
            b_ids2_para = b_ids2_para.to(device)
            b_mask_para = b_mask_para.to(device)
            b_mask2_para = b_mask2_para.to(device)
            b_labels_para = b_labels_para.to(device)

            optimizer.zero_grad()
            logit_para = model.predict_paraphrase(b_ids_para, b_mask_para, b_ids2_para, b_mask2_para)
            loss2 = F.binary_cross_entropy(F.sigmoid(logit_para.view(-1)), b_labels_para.view(-1).float(), reduction='sum') / args.batch_size
            
            b_ids_sts, b_ids2_sts, b_mask_sts, b_mask2_sts, b_labels_sts = (batch3['token_ids_1'], batch3['token_ids_2'],
                                       batch3['attention_mask_1'], batch3['attention_mask_2'], batch3['labels'])

            b_ids_sts = b_ids_sts.to(device)
            b_ids2_sts = b_ids2_sts.to(device)
            b_mask_sts = b_mask_sts.to(device)
            b_mask2_sts = b_mask2_sts.to(device)
            b_labels_sts = b_labels_sts.to(device)
            
            optimizer.zero_grad()
            #logit = model.predict_similarity(b_ids, b_mask, b_ids2, b_mask2)
            sim_score = model.predict_similarity(b_ids_sts, b_mask_sts, b_ids2_sts, b_mask2_sts)
            #cos_score_trans = nn.Identity()
            loss_MSE = nn.MSELoss()
            #sim_score = cos_score_trans(sim_score)
            #sim_score = sim_score.to(device)
            loss3 = loss_MSE(sim_score, b_labels_sts.view(-1).float()) / args.batch_size
            #loss = loss.to(device)
            #loss = F.cross_entropy(logit.view(-1), b_labels.view(-1).type(torch.FloatTensor), reduction='sum') / args.batch_size
            
            #contrastive learning
            
            loss = loss1 + loss2 + loss3
            
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_para_acc, _, _, train_sent_acc, _, _, train_sts_corr, _, _ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        dev_para_acc, _, _, dev_sent_acc, _, _, dev_sts_corr, _, _  = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)
        if (dev_para_acc+dev_sent_acc+dev_sts_corr)/3 > best_dev_acc:
            best_dev_acc = (dev_para_acc+dev_sent_acc+dev_sts_corr)/3
            save_model(model, optimizer, args, config, args.filepath)
        
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}")
        print(f"Epoch {epoch}:  train acc :: {train_para_acc :.3f}, dev acc :: {dev_para_acc :.3f}")
        print(f"Epoch {epoch}:  train acc :: {train_sts_corr :.3f}, dev acc :: {dev_sts_corr :.3f}")
        print(f"Epoch {epoch}:  train acc :: {train_sent_acc :.3f}, dev acc :: {dev_sent_acc :.3f}")



def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=16)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-3)
    
    # parameters for extension 1: layer-wise learning rate decay
    parser.add_argument("--layer_learning_rate",
                        type=float,
                        nargs='+',
                        default=[1.5e-5]*12,
                        help="learning rate in each group")
    parser.add_argument("--layer_learning_rate_decay",
                        type=float,
                        default=0.95)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
