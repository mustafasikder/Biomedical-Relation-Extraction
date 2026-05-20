import argparse
import json
import logging
import os
import random

import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, BertModel, BertPreTrainedModel, get_linear_schedule_with_warmup

from relation_experiment import CLS, SEP, generate_relation_data, compute_f1, save_model
from shared.const import task_ner_labels, task_rel_labels

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_WORDS = 128
nlp = spacy.load('en_core_web_sm')


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        return F.relu(self.linear(torch.bmm(adj, x)))


class GNNBertForRelation(BertPreTrainedModel):
    def __init__(self, config, num_rel_labels=2, gnn_hidden_dim=256):
        super().__init__(config)
        self.num_labels = num_rel_labels
        H = config.hidden_size
        self.bert       = BertModel(config)
        self.dropout    = nn.Dropout(config.hidden_dropout_prob)
        self.gcn1       = GCNLayer(H, gnn_hidden_dim)
        self.gcn2       = GCNLayer(gnn_hidden_dim, H)
        self.layer_norm = nn.LayerNorm(H * 2)
        self.classifier = nn.Linear(H * 2, num_rel_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, sub_word_idx=None, obj_word_idx=None,
                word_to_subtok=None, adj_matrix=None):
        seq      = self.dropout(self.bert(input_ids, token_type_ids, attention_mask)[0])
        counts   = word_to_subtok.sum(-1, keepdim=True).clamp(min=1e-9)
        word_emb = torch.bmm(word_to_subtok, seq) / counts
        word_emb = self.gcn2(self.gcn1(word_emb, adj_matrix), adj_matrix)
        idx      = torch.arange(input_ids.size(0), device=input_ids.device)
        rep      = self.layer_norm(torch.cat([word_emb[idx, sub_word_idx], word_emb[idx, obj_word_idx]], dim=1))
        logits   = self.classifier(self.dropout(rep))
        if labels is not None:
            return CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))
        return logits


class GNNInputFeatures:
    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                 sub_word_idx, obj_word_idx, word_to_subtok_sparse, sentence_key, max_seq_length):
        self.input_ids             = input_ids
        self.input_mask            = input_mask
        self.segment_ids           = segment_ids
        self.label_id              = label_id
        self.sub_word_idx          = sub_word_idx
        self.obj_word_idx          = obj_word_idx
        self.word_to_subtok_sparse = word_to_subtok_sparse
        self.sentence_key          = sentence_key
        self.max_seq_length        = max_seq_length


class GNNDataset(torch.utils.data.Dataset):
    def __init__(self, features, adj_cache):
        self.features  = features
        self.adj_cache = adj_cache

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        f   = self.features[idx]
        w2s = np.zeros((MAX_WORDS, f.max_seq_length), dtype=np.float32)
        for wi, pos in f.word_to_subtok_sparse:
            if wi < MAX_WORDS and pos < f.max_seq_length:
                w2s[wi, pos] = 1.0
        return (
            torch.tensor(f.input_ids,    dtype=torch.long),
            torch.tensor(f.input_mask,   dtype=torch.long),
            torch.tensor(f.segment_ids,  dtype=torch.long),
            torch.tensor(f.label_id,     dtype=torch.long),
            torch.tensor(f.sub_word_idx, dtype=torch.long),
            torch.tensor(f.obj_word_idx, dtype=torch.long),
            torch.tensor(w2s,            dtype=torch.float),
            torch.tensor(self.adj_cache[f.sentence_key], dtype=torch.float),
        )


def add_marker_tokens(tokenizer, ner_labels):
    tokens = ['<SUBJ_START>', '<SUBJ_END>', '<OBJ_START>', '<OBJ_END>']
    for l in ner_labels:
        tokens += [f'<SUBJ_START={l}>', f'<SUBJ_END={l}>', f'<OBJ_START={l}>', f'<OBJ_END={l}>']
    tokenizer.add_tokens(tokens)


def build_adj_cache(examples):
    from spacy.tokens import Doc
    unique = list({tuple(ex['token']) for ex in examples})
    adj_cache = {}
    for words in unique:
        n   = min(len(words), MAX_WORDS)
        adj = np.eye(MAX_WORDS, dtype=np.float32)
        try:
            doc = Doc(nlp.vocab, words=list(words[:n]))
            for _, proc in nlp.pipeline:
                doc = proc(doc)
            for tok in doc:
                i, j = tok.i, tok.head.i
                if i != j and i < MAX_WORDS and j < MAX_WORDS:
                    adj[i, j] = adj[j, i] = 1.0
        except Exception:
            pass
        deg        = adj.sum(1)
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        adj_cache[words] = adj * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    return adj_cache


def convert_examples_to_features(examples, label2id, max_seq_length, tokenizer, special_tokens):
    adj_cache = build_adj_cache(examples)
    features  = []

    def special(w):
        if w not in special_tokens:
            special_tokens[w] = f'<{w}>'
        return special_tokens[w]

    for ex in examples:
        tokens = [CLS]
        SS = special(f"SUBJ_START={ex['subj_type']}"); SE = special(f"SUBJ_END={ex['subj_type']}")
        OS = special(f"OBJ_START={ex['obj_type']}");  OE = special(f"OBJ_END={ex['obj_type']}")
        word_map = {}

        for i, word in enumerate(ex['token']):
            if i == ex['subj_start']: tokens.append(SS)
            if i == ex['obj_start']:  tokens.append(OS)
            positions = []
            for sub in tokenizer.tokenize(word):
                positions.append(len(tokens)); tokens.append(sub)
            if positions: word_map[i] = positions
            if i == ex['subj_end']: tokens.append(SE)
            if i == ex['obj_end']:  tokens.append(OE)

        tokens.append(SEP)
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]

        segment_ids = [0] * len(tokens)
        input_ids   = tokenizer.convert_tokens_to_ids(tokens)
        input_mask  = [1] * len(input_ids)
        pad         = max_seq_length - len(input_ids)
        input_ids   += [0] * pad; input_mask += [0] * pad; segment_ids += [0] * pad

        w2s_sparse = [(wi, pos) for wi, positions in word_map.items()
                      if wi < MAX_WORDS for pos in positions if pos < max_seq_length]

        features.append(GNNInputFeatures(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
            label_id=label2id[ex['relation']],
            sub_word_idx=min(ex['subj_start'], MAX_WORDS - 1),
            obj_word_idx=min(ex['obj_start'],  MAX_WORDS - 1),
            word_to_subtok_sparse=w2s_sparse,
            sentence_key=tuple(ex['token']),
            max_seq_length=max_seq_length,
        ))
    return features, adj_cache


def evaluate(model, device, loader, label_ids, num_labels, e2e_ngold=None):
    model.eval()
    all_logits = []
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        ids, mask, seg, lids, sub_w, obj_w, w2s, adj = batch
        with torch.no_grad():
            logits = model(ids, seg, mask, sub_word_idx=sub_w, obj_word_idx=obj_w,
                           word_to_subtok=w2s, adj_matrix=adj)
        all_logits.append(logits.detach().cpu().numpy())
    preds = np.argmax(np.concatenate(all_logits), axis=1)
    return preds, compute_f1(preds, label_ids.numpy(), e2e_ngold)


def main(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    n_gpu  = torch.cuda.device_count()
    os.makedirs(args.output_dir, exist_ok=True)

    label_list = ['no_relation'] + task_rel_labels[args.task]
    label2id   = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    config    = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    add_marker_tokens(tokenizer, task_ner_labels[args.task])
    special_tokens = {}

    def load_split(split):
        _, examples, nrel = generate_relation_data(os.path.join(args.data_dir, f'{split}.json'))
        feats, adj_cache  = convert_examples_to_features(examples, label2id, args.max_seq_length,
                                                          tokenizer, special_tokens)
        label_ids = torch.tensor([f.label_id for f in feats], dtype=torch.long)
        return feats, adj_cache, label_ids, nrel

    train_feats, train_adj, _,              train_nrel = load_split('train')
    eval_feats,  eval_adj,  eval_label_ids, eval_nrel  = load_split('dev')
    test_feats,  test_adj,  test_label_ids, test_nrel  = load_split('test')

    train_loader = DataLoader(GNNDataset(train_feats, train_adj), batch_size=args.train_batch_size, shuffle=True)
    eval_loader  = DataLoader(GNNDataset(eval_feats,  eval_adj),  batch_size=args.eval_batch_size)
    test_loader  = DataLoader(GNNDataset(test_feats,  test_adj),  batch_size=args.eval_batch_size)

    model = GNNBertForRelation.from_pretrained(args.model, config=config,
                                                num_rel_labels=num_labels,
                                                gnn_hidden_dim=args.gnn_hidden_dim,
                                                ignore_mismatched_sizes=True)
    old_size = model.bert.get_input_embeddings().weight.size(0)
    model.bert.resize_token_embeddings(len(tokenizer))
    emb = model.bert.get_input_embeddings()
    with torch.no_grad():
        emb.weight[old_size:] = emb.weight[:old_size].mean(0).unsqueeze(0).expand(len(tokenizer) - old_size, -1)

    model.to(device)
    if n_gpu > 1: model = nn.DataParallel(model)

    total_steps = len(train_loader) * args.num_train_epochs
    no_decay    = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    params = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    best_result = None
    eval_step   = max(1, len(train_loader) // args.eval_per_epoch)
    global_step = running_loss = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(device) for t in batch)
            ids, mask, seg, lids, sub_w, obj_w, w2s, adj = batch
            loss = model(ids, seg, mask, lids, sub_word_idx=sub_w, obj_word_idx=obj_w,
                         word_to_subtok=w2s, adj_matrix=adj)
            if n_gpu > 1: loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            running_loss += loss.item(); global_step += 1
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

            if (step + 1) % eval_step == 0:
                logger.info("Epoch %d step %d loss=%.4f", epoch, step + 1, running_loss / global_step)
                _, result = evaluate(model, device, eval_loader, eval_label_ids, num_labels, eval_nrel)
                model.train()
                if best_result is None or result['f1'] > best_result['f1']:
                    best_result = result
                    logger.info("New best dev f1=%.4f", result['f1'])
                    save_model(args.output_dir, model, tokenizer)

    test_model = GNNBertForRelation.from_pretrained(args.output_dir, num_rel_labels=num_labels,
                                                     gnn_hidden_dim=args.gnn_hidden_dim,
                                                     ignore_mismatched_sizes=True)
    test_model.to(device)
    _, test_result = evaluate(test_model, device, test_loader, test_label_ids, num_labels, test_nrel)
    logger.info("Test: %s", json.dumps(test_result))

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump({'args': vars(args), 'dev': best_result, 'test': test_result}, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',             required=True, choices=['chemprot', 'ddi', 'aimed'])
    parser.add_argument('--data_dir',         required=True)
    parser.add_argument('--output_dir',       required=True)
    parser.add_argument('--model',            required=True)
    parser.add_argument('--gnn_hidden_dim',   type=int,   default=256)
    parser.add_argument('--train_batch_size', type=int,   default=16)
    parser.add_argument('--eval_batch_size',  type=int,   default=32)
    parser.add_argument('--learning_rate',    type=float, default=2e-5)
    parser.add_argument('--num_train_epochs', type=int,   default=5)
    parser.add_argument('--eval_per_epoch',   type=int,   default=2)
    parser.add_argument('--max_seq_length',   type=int,   default=256)
    parser.add_argument('--seed',             type=int,   default=42)
    parser.add_argument('--no_cuda',          action='store_true')
    args = parser.parse_args()
    main(args)
