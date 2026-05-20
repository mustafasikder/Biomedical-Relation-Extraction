import json
import numpy as np


def fields_to_batches(d, keys_to_ignore=[]):
    keys = [k for k in d if k not in keys_to_ignore]
    length = len(d[keys[0]])
    return [{k: d[k][i] for k in keys} for i in range(length)]


class Dataset:
    def __init__(self, json_file):
        self.documents = [Document(json.loads(line)) for line in open(json_file)]

    def __getitem__(self, ix):
        return self.documents[ix]

    def __len__(self):
        return len(self.documents)


class Document:
    def __init__(self, js):
        self._doc_key = js["doc_key"]
        entries = fields_to_batches(js, ["doc_key", "clusters", "section_starts"])
        sentence_lengths = [len(e["sentences"]) for e in entries]
        sentence_starts = np.roll(np.cumsum(sentence_lengths), 1)
        sentence_starts[0] = 0
        self.sentences = [Sentence(e, s, i) for i, (e, s) in enumerate(zip(entries, sentence_starts))]

    def __getitem__(self, ix):
        return self.sentences[ix]

    def __len__(self):
        return len(self.sentences)


class Sentence:
    def __init__(self, entry, sentence_start, sentence_ix):
        self.sentence_start = sentence_start
        self.sentence_ix    = sentence_ix
        self.text           = entry["sentences"]
        self.ner            = [NER(n, self.text, sentence_start) for n in entry.get("ner", [])]
        self.relations      = [Relation(r, self.text, sentence_start) for r in entry.get("relations", [])]

    def __len__(self):
        return len(self.text)


class Span:
    def __init__(self, start, end, text, sentence_start):
        self.start_doc  = start
        self.end_doc    = end
        self.start_sent = start - sentence_start
        self.end_sent   = end - sentence_start
        self.span_doc   = (self.start_doc, self.end_doc)
        self.span_sent  = (self.start_sent, self.end_sent)
        self.text       = text[self.start_sent:self.end_sent + 1]

    def __eq__(self, other):
        return self.span_doc == other.span_doc and self.span_sent == other.span_sent

    def __hash__(self):
        return hash(self.span_doc + self.span_sent + (" ".join(self.text),))


class NER:
    def __init__(self, ner, text, sentence_start):
        self.span  = Span(ner[0], ner[1], text, sentence_start)
        self.label = ner[2]


class Relation:
    def __init__(self, relation, text, sentence_start):
        self.pair  = (Span(relation[0], relation[1], text, sentence_start),
                      Span(relation[2], relation[3], text, sentence_start))
        self.label = relation[4]
