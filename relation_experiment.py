import argparse
import json
import logging
import os
import random
import time

import numpy as np
import spacy
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel

from shared.const import task_rel_labels, task_ner_labels
from shared.data_structures import Dataset

nlp = spacy.load("en_core_web_sm")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

CLS = "[CLS]"
SEP = "[SEP]"
NO_VERB_TOKEN = "<NO_VERB>"


class InputFeatures:
    def __init__(self, input_ids, input_mask, segment_ids, label_id, sub_idx, obj_idx, verb_idx=None):
        self.input_ids   = input_ids
        self.input_mask  = input_mask
        self.segment_ids = segment_ids
        self.label_id    = label_id
        self.sub_idx     = sub_idx
        self.obj_idx     = obj_idx
        self.verb_idx    = verb_idx


def find_main_verb(tokens, subj_start, subj_end):
    from spacy.tokens import Doc
    doc = Doc(nlp.vocab, words=tokens)
    for _, proc in nlp.pipeline:
        doc = proc(doc)
    roots = [t.i for t in doc if t.dep_ == "ROOT" and t.pos_ in ("VERB", "AUX")]
    if not roots:
        return None
    if len(roots) == 1:
        return roots[0]
    center = (subj_start + subj_end) // 2
    return min(roots, key=lambda v: abs(v - center))


def add_marker_tokens(tokenizer, ner_labels, use_verb_token=False):
    new_tokens = ['<SUBJ_START>', '<SUBJ_END>', '<OBJ_START>', '<OBJ_END>']
    for label in ner_labels:
        new_tokens += [f'<SUBJ_START={label}>', f'<SUBJ_END={label}>', f'<OBJ_START={label}>', f'<OBJ_END={label}>']
    if use_verb_token:
        new_tokens += ['<VERB>', '</VERB>', NO_VERB_TOKEN]
    tokenizer.add_tokens(new_tokens)


def convert_examples_to_features(examples, label2id, max_seq_length, tokenizer, special_tokens, use_verb_token=False):
    def get_special_token(w):
        if w not in special_tokens:
            special_tokens[w] = f"<{w}>"
        return special_tokens[w]

    features = []
    for ex in examples:
        tokens  = [CLS]
        SUBJ_S  = get_special_token(f"SUBJ_START={ex['subj_type']}")
        SUBJ_E  = get_special_token(f"SUBJ_END={ex['subj_type']}")
        OBJ_S   = get_special_token(f"OBJ_START={ex['obj_type']}")
        OBJ_E   = get_special_token(f"OBJ_END={ex['obj_type']}")

        verb_orig = find_main_verb(ex['token'], ex['subj_start'], ex['subj_end']) if use_verb_token else None
        sub_idx = obj_idx = verb_idx = None

        for i, token in enumerate(ex['token']):
            if i == ex['subj_start']:
                sub_idx = len(tokens); tokens.append(SUBJ_S)
            if i == ex['obj_start']:
                obj_idx = len(tokens); tokens.append(OBJ_S)
            if use_verb_token and verb_orig is not None and i == verb_orig:
                tokens.append(get_special_token("VERB"))
                verb_idx = len(tokens)
            tokens += tokenizer.tokenize(token)
            if use_verb_token and verb_orig is not None and i == verb_orig:
                tokens.append(get_special_token("/VERB"))
            if i == ex['subj_end']:
                tokens.append(SUBJ_E)
            if i == ex['obj_end']:
                tokens.append(OBJ_E)

        if use_verb_token and verb_idx is None:
            verb_idx = len(tokens); tokens.append(NO_VERB_TOKEN)

        tokens.append(SEP)

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
            if sub_idx is None or sub_idx >= max_seq_length: sub_idx = 1
            if obj_idx is None or obj_idx >= max_seq_length: obj_idx = 1
            if use_verb_token and (verb_idx is None or verb_idx >= max_seq_length): verb_idx = 1

        segment_ids = [0] * len(tokens)
        input_ids   = tokenizer.convert_tokens_to_ids(tokens)
        input_mask  = [1] * len(input_ids)
        pad = max_seq_length - len(input_ids)
        input_ids   += [0] * pad
        input_mask  += [0] * pad
        segment_ids += [0] * pad

        features.append(InputFeatures(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
            label_id=label2id[ex['relation']],
            sub_idx=sub_idx, obj_idx=obj_idx,
            verb_idx=verb_idx if use_verb_token else None,
        ))
    return features


class BertForRelation(BertPreTrainedModel):
    def __init__(self, config, num_rel_labels, use_verb_token=False):
        super().__init__(config)
        self.num_labels    = num_rel_labels
        self.use_verb_token = use_verb_token
        self.bert          = BertModel(config)
        self.dropout       = nn.Dropout(config.hidden_dropout_prob)
        input_dim          = config.hidden_size * (3 if use_verb_token else 2)
        self.layer_norm    = nn.LayerNorm(input_dim)
        self.classifier    = nn.Linear(input_dim, num_rel_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, sub_idx=None, obj_idx=None, verb_idx=None):
        seq = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        idx = torch.arange(input_ids.size(0), device=input_ids.device)
        reps = [seq[idx, sub_idx], seq[idx, obj_idx]]
        if self.use_verb_token and verb_idx is not None:
            reps.append(seq[idx, verb_idx])
        rep    = self.layer_norm(torch.cat(reps, dim=1))
        logits = self.classifier(self.dropout(rep))
        if labels is not None:
            return CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))
        return logits


def generate_relation_data(path):
    data = Dataset(path)
    samples = []
    nrel = 0
    for doc in data:
        for sent in doc:
            nrel += len(sent.relations)
            gold = {rel.pair: rel.label for rel in sent.relations}
            for x in range(len(sent.ner)):
                for y in range(len(sent.ner)):
                    if x == y: continue
                    sub, obj = sent.ner[x], sent.ner[y]
                    samples.append({
                        'relation':   gold.get((sub.span, obj.span), 'no_relation'),
                        'subj_start': sub.span.start_sent, 'subj_end': sub.span.end_sent,
                        'subj_type':  sub.label,
                        'obj_start':  obj.span.start_sent, 'obj_end':  obj.span.end_sent,
                        'obj_type':   obj.label,
                        'token':      sent.text,
                    })
    return data, samples, nrel


def compute_f1(preds, labels, e2e_ngold):
    n_gold = n_pred = n_correct = 0
    for p, l in zip(preds, labels):
        if p != 0: n_pred += 1
        if l != 0: n_gold += 1
        if p != 0 and l != 0 and p == l: n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    prec = n_correct / n_pred
    rec  = n_correct / e2e_ngold
    f1   = 2 * prec * rec / (prec + rec)
    return {'precision': prec, 'recall': rec, 'f1': f1, 'n_correct': n_correct, 'n_pred': n_pred}


def evaluate(model, device, loader, label_ids, num_labels, e2e_ngold=None):
    model.eval()
    all_logits = []
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        ids, mask, seg, lids, sub_idx, obj_idx = batch[:6]
        verb_idx = batch[6] if len(batch) == 7 else None
        with torch.no_grad():
            logits = model(ids, seg, mask, sub_idx=sub_idx, obj_idx=obj_idx, verb_idx=verb_idx)
        all_logits.append(logits.detach().cpu().numpy())
    preds = np.argmax(np.concatenate(all_logits), axis=1)
    return preds, compute_f1(preds, label_ids.numpy(), e2e_ngold)


def make_dataset(features, use_verb):
    tensors = [
        torch.tensor([f.input_ids   for f in features], dtype=torch.long),
        torch.tensor([f.input_mask  for f in features], dtype=torch.long),
        torch.tensor([f.segment_ids for f in features], dtype=torch.long),
        torch.tensor([f.label_id    for f in features], dtype=torch.long),
        torch.tensor([f.sub_idx     for f in features], dtype=torch.long),
        torch.tensor([f.obj_idx     for f in features], dtype=torch.long),
    ]
    if use_verb:
        tensors.append(torch.tensor([f.verb_idx for f in features], dtype=torch.long))
    return TensorDataset(*tensors)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def save_model(output_dir, model, tokenizer):
    os.makedirs(output_dir, exist_ok=True)
    m = model.module if hasattr(model, 'module') else model
    m.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    n_gpu  = torch.cuda.device_count()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    label_list = ['no_relation'] + task_rel_labels[args.task]
    label2id   = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    add_marker_tokens(tokenizer, task_ner_labels[args.task], args.use_verb_token)
    special_tokens = {}

    def load_split(split):
        _, examples, nrel = generate_relation_data(os.path.join(args.data_dir, f'{split}.json'))
        feats = convert_examples_to_features(examples, label2id, args.max_seq_length,
                                             tokenizer, special_tokens, args.use_verb_token)
        label_ids = torch.tensor([f.label_id for f in feats], dtype=torch.long)
        return feats, label_ids, nrel

    train_feats, _,              train_nrel = load_split('train')
    eval_feats,  eval_label_ids, eval_nrel  = load_split('dev')
    test_feats,  test_label_ids, test_nrel  = load_split('test')

    train_loader = DataLoader(make_dataset(train_feats, args.use_verb_token), batch_size=args.train_batch_size, shuffle=True)
    eval_loader  = DataLoader(make_dataset(eval_feats,  args.use_verb_token), batch_size=args.eval_batch_size)
    test_loader  = DataLoader(make_dataset(test_feats,  args.use_verb_token), batch_size=args.eval_batch_size)

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model)
    model  = BertForRelation.from_pretrained(args.model, config=config,
                                              num_rel_labels=num_labels,
                                              use_verb_token=args.use_verb_token)

    old_size = model.bert.get_input_embeddings().weight.size(0)
    model.bert.resize_token_embeddings(len(tokenizer))
    emb = model.bert.get_input_embeddings()
    with torch.no_grad():
        mean = emb.weight[:old_size].mean(0)
        emb.weight[old_size:] = mean.unsqueeze(0).expand(len(tokenizer) - old_size, -1)

    model.to(device)
    if n_gpu > 1: model = nn.DataParallel(model)

    total_steps = len(train_loader) * args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
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
            batch    = tuple(t.to(device) for t in batch)
            ids, mask, seg, lids, sub_idx, obj_idx = batch[:6]
            verb_idx = batch[6] if len(batch) == 7 else None
            loss     = model(ids, seg, mask, lids, sub_idx=sub_idx, obj_idx=obj_idx, verb_idx=verb_idx)
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

    test_model = BertForRelation.from_pretrained(args.output_dir, num_rel_labels=num_labels,
                                                  use_verb_token=args.use_verb_token,
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
    parser.add_argument('--use_verb_token',   action='store_true')
    parser.add_argument('--train_batch_size', type=int,   default=16)
    parser.add_argument('--eval_batch_size',  type=int,   default=32)
    parser.add_argument('--learning_rate',    type=float, default=2e-5)
    parser.add_argument('--num_train_epochs', type=int,   default=5)
    parser.add_argument('--eval_per_epoch',   type=int,   default=10)
    parser.add_argument('--max_seq_length',   type=int,   default=256)
    parser.add_argument('--seed',             type=int,   default=42)
    parser.add_argument('--no_cuda',          action='store_true')
    args = parser.parse_args()
    main(args)
