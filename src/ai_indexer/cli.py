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

model = None
tokenizer = None

def format_text(texts, type_):
    if type_ == 'passage':
        return [f'passage: {t}' for t in texts]
    elif type_ == 'query':
        return [f'query: {t}' for t in texts]
    else:
        raise RuntimeError(f'unkown format {type_}')
    
device = 'cpu'

def embed(texts, type_):
    global model
    global tokenizer
    if model is None:
        model = AutoModel.from_pretrained('intfloat/e5-small-v2').to(device)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')


    input_texts = format_text(texts, type_)
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    return embeddings



def search_2(index, query, topk=3):
    qemb = embed([query], 'query').to('cpu')
    #print(qemb.shape)
    qembt = qemb.float().detach().numpy()
    dist, n_idx = index.search(qembt, topk)
    return n_idx


def build_index_embeds(files: list[str]) -> list[np.array]:
    batch = []
    embs = []
    bs = 5

    for filepath in files:
        #print(filepath)
        #continue
        res = subprocess.run(['pdftotext', str(filepath), '/dev/stdout'], text=True, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

        txt = res.stdout
        batch.append(txt)
        if len(batch) > bs:
            embs.extend(embed(batch, 'passage').to('cpu'))
            batch = []

    embs.extend(embed(batch, 'passage'))
    batch = []
    vecs = []

    for emb in embs:
        vecs.append(emb.float().unsqueeze(0).detach().numpy())

    return vecs

def build_index(vecs):
    index = faiss.IndexFlatL2(384)   # build the index
    #vecs = build_index_embeds(files)

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

@cli.command()
@click.argument('query', type=str)
@click.pass_context
def search(ctx, query):
    env = lmdb.open('db.lmdb')
    '''
    with np.load('vec_db.npz') as data:
        vecs = np.array(data['arr_0'])
    '''

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
                print(f'{nidx}: {n} {filename}')


@cli.command()
@click.argument('pdf_files', type=click.File('r'))
@click.pass_context
def index(ctx, pdf_files):
    files = []
    while True:
        line = pdf_files.readline()
        if not line:
            break
        filepath = Path(line.strip())
        files.append(str(filepath))

    env = lmdb.open('db.lmdb')

    #if not Path('vec_db.npz').exists():
    vecs = build_index_embeds(files)
    #np.savez_compressed('vec_db.npz', np.stack(vecs))

    for i, (f,ve) in enumerate(zip(files, vecs)):
        with env.begin(write=True, buffers=True) as txn:
            k = str(i).zfill(fill_depth).encode('utf-8')
            ffp = io.StringIO()
            np.savetxt(ffp, ve)
            ffp.seek(0)
            record = {'fname':str(f), 'vec':str(ffp.read())}
            vall = json.dumps(record).encode('utf-8')
            txn.put(k,vall)
            print(i)

    #vecs = np.load('vec_db.npz')
    #index = build_index(vecs)

if __name__ == '__main__':
    cli()


# needs click transformers torch
# and pdftotext cli command
