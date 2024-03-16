import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


''' prompt encoding:
Here are some rules of thumb:

    Use "query: " and "passage: " correspondingly for asymmetric tasks such as passage retrieval in open QA, ad-hoc information retrieval.

    Use "query: " prefix for symmetric tasks such as semantic similarity, paraphrase retrieval.

    Use "query: " prefix if you want to use embeddings as features, such as linear probing classification, clustering.
'''    

# max ctx len is 512 for large-unsupervised


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


'''
# Each input text should start with "query: " or "passage: ".
# For tasks other than retrieval, you can simply use the "query: " prefix.
input_texts = ['query: how much protein should a female eat',
               'query: summit define',
               "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
               "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."]

tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')
model = AutoModel.from_pretrained('intfloat/e5-small-v2')

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
print(embeddings)
'''

'''
# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
'''

import click
from pathlib import Path
import subprocess
import lmdb
import numpy as np
import faiss                   # make faiss available
import io
import json
import torch
from more_itertools import chunked, flatten

model = None
tokenizer = None

def format_text(texts, type_):
    if type_ == 'passage':
        return [f'passage: {t}' for t in texts]
    elif type_ == 'query':
        return [f'query: {t}' for t in texts]
    else:
        raise RuntimeError(f'unkown format {type_}')
    
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def embed_batch(texts, type_):
    global model
    global tokenizer
    if model is None:
        model = AutoModel.from_pretrained('intfloat/e5-small-v2').to(device)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')

    #bs = 1
    #embeddings = []

    # batch across text sizes
    #for batch in chunked(texts, bs):
    with torch.no_grad():
        batch = texts
        input_texts = format_text(batch, type_)
        batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k:x.to(device) for k,x in batch_dict.items()}

        outputs = model(**batch_dict)
        eee = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        #embeddings.append(eee)

        #return [x.float().to('cpu').detach().numpy() for x in flatten(eee)]
        return eee.float().to('cpu').detach().numpy() 



def search_2(index, query, topk=5):
    qemb = np.array(embed_batch([query], 'query'))
    #qemb = np.array(embed_batch([query], 'passage'))
    #print(qemb.shape)
    qembt = qemb
    dist, n_idx = index.search(qembt, topk)
    return n_idx

def file_to_text_splits(fp):
    res = subprocess.run(['pdftotext', str(fp), '/dev/stdout'], text=True, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    txt = res.stdout
    # 512 = max model context
    for chunk in chunked(str(txt), 512):
        yield fp, ''.join(chunk)

def build_index_embeds(files: list[str]) -> list[np.array]:
    bs = 16
    batch = []

    # batch of files
    for f in files:
        for fn, split in file_to_text_splits(f):
            batch.append(split)
            if len(batch) == bs:
                embs = embed_batch(batch, 'passage')
                batch = []
                for emb in embs:
                    yield fn, np.expand_dims(emb, 0), split


def build_index(vecs):
    index = faiss.IndexFlatL2(384)   # build the index
    for v in vecs:
        index.add(v)
    return index



@click.group()
@click.pass_context
def cli(ctx):
    pass


# how much to fill they keys
# this allows 10^32 db-entries
fill_depth = 32
max_dbsize = int(1e12)

@cli.command()
@click.argument('query', type=str)
@click.pass_context
def search(ctx, query):
    #env = lmdb.open('db.lmdb', map_size=max_dbsize)
    global max_dbsize
    env = lmdb.open('db.lmdb', map_size=max_dbsize)
    print(env.stat())

    vecs = []
    with env.begin(buffers=True) as txn:
        cursor = txn.cursor()

        for k, val in iter(cursor):
            b = val
            s = bytes(b).decode('utf-8')
            record = json.loads(s)
            ffp = io.StringIO()
            ffp.write(record['vec'])
            ffp.seek(0)
            v = np.loadtxt(ffp)
            idx = bytes(k).decode('utf-8')
            vecs.append(np.expand_dims(v, 0))

    index = build_index(vecs)
    search_batch = search_2(index, query)

    for neighs in search_batch:
        for nidx, n in enumerate(neighs):
            with env.begin(buffers=True) as txn:
                kvk = str(n).zfill(fill_depth).encode('utf-8')
                b = txn.get(kvk)
                s = bytes(b).decode('utf-8')
                record = json.loads(s)
                filename = record['fname']
                t = record['txt']
                print(f'{nidx}: {n} {filename} {t}')

def stream_files(fp):
    while True:
        line = fp.readline()
        if not line:
            break
        filepath = Path(line.strip())
        yield str(filepath)



def index2(pdf_files):
    global max_dbsize
    env = lmdb.open('db.lmdb', map_size=max_dbsize)
    fst = stream_files(pdf_files)
    vecs = build_index_embeds(fst)

    for i, (f,ve,t) in enumerate(vecs):
        with env.begin(write=True, buffers=True) as txn:
            k = str(i).zfill(fill_depth).encode('utf-8')
            ffp = io.StringIO()
            np.savetxt(ffp, ve)
            ffp.seek(0)
            record = {'fname':str(f), 'vec':str(ffp.read()), 'txt':t}
            vall = json.dumps(record).encode('utf-8')
            txn.put(k,vall)
            print(f)


@cli.command()
@click.argument('pdf_files', type=click.File('r'))
@click.pass_context
def index(ctx, pdf_files):
    index2(pdf_files)

if __name__ == '__main__':
    cli()


# needs click transformers torch
# and pdftotext cli command
