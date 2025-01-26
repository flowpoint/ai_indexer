import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

import click
from pathlib import Path
import subprocess
import lmdb
import numpy as np
import faiss
from faiss import write_index, read_index
import io
import json
import sys
from time import time
from more_itertools import chunked, flatten, batched

model_type = None
model_name = None
# the index here is the faiss index and the db together
index_folder = Path('./index')
db_path = index_folder / Path('db.lmdb')
faiss_file = index_folder / Path('faiss.index')

verbosity = 1

model = None
tokenizer = None
faiss_index = None
embedding_size = None

# how much to fill they keys
# this allows 10^32 db-entries
fill_depth = 32
max_dbsize = int(1e12)

device = None

# e5-small-v2 formatting/pooling

# max ctx len is 512 for large-unsupervised
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
# ----

# mistral-7b formatting/pooling

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a web search query, retrieve relevant passages that answer the query'
def format_text(texts, type_):
    if type_ == 'passage':
        return [f'{t}' for t in texts]
    elif type_ == 'query':
        return [str(get_detailed_instruct(task, t)) for t in texts]
    else:
        raise RuntimeError(f'unkown format {type_}')

# ----

def format_text2(texts, type_):
    if type_ == 'passage':
        return [f'passage: {t}' for t in texts]
    elif type_ == 'query':
        return [f'query: {t}' for t in texts]
    else:
        raise RuntimeError(f'unkown format {type_}')
   

def set_model(model_type_):
    global model_type
    global embedding_size
    global model_name

    model_type = model_type_

    if model_type == 'big':
        model_name = 'intfloat/e5-mistral-7b-instruct'
        embedding_size = 4096
    elif model_type == 'small':
        model_name = 'intfloat/e5-small-v2'
        embedding_size = 384
    else:
        raise RuntimeError(f'unknown model type {model_type}')


def load_model():
    global model
    global tokenizer
    global embedding_size

    if model is None:
        if device == 'cuda':
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, load_in_4bit=True)#.to(device)
            # compiling had issues
            #model = torch.compile(model)
        else:
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
            model = torch.compile(model)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

def embed_batch(texts, type_):
    load_model()
    with torch.no_grad():
        batch = texts
        input_texts = format_text(batch, type_)
        batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k:x.to(device) for k,x in batch_dict.items()}

        outputs = model(**batch_dict)

        if model_type == 'big':
            eee = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        elif model_type == 'small':
            eee = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        return eee.float().to('cpu').detach().numpy() 


def search_faiss(index, query, topk=5):
    qemb = np.array(embed_batch([query], 'query'))
    dist, n_idx = index.search(qemb, topk)
    return n_idx

def file_to_text_splits(file_: str):
    file = Path(file_)
    try:
        if file.name.endswith('pdf'):
            res = subprocess.run(['pdftotext', str(file), '/dev/stdout'], 
                                 text=True, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            txt = res.stdout
        elif file.name.endswith('.txt') or file.name.endswith('.md'):
            with file.open('r') as fp:
                txt = fp.read()
        # 512 = max model context
        for chunk in chunked(str(txt), 512):
            yield str(file), ''.join(chunk)
    except Exception as e:
        eprint(f'error {e} with file {str(file)}')

def eprint(*args, **kwargs):
    if verbosity >= 1:
        print(*args, file=sys.stderr, **kwargs)


def build_index_embeds(files: list[str], bs=128) -> list[np.array]:
    # batch of files
    for batch in batched(flatten(map(file_to_text_splits, files)), bs):
        try:
            #for batch in batched(file_to_text_splits(Path(f)), bs):
            unpadded_len = len(batch)
            if unpadded_len == bs:
                pbatch = batch
            else:
                # pad the batch
                pbatch = list(batch) + ['', ''] * (bs-unpadded_len)

            filenames, splits = zip(*batch)
            embs = embed_batch(splits, 'passage')
            for emb, filename, split, _ in zip(embs, filenames, splits, range(unpadded_len)):
                yield filename, np.expand_dims(emb, 0), split
        except Exception as e:
            eprint(f'error {e} embedding batch {str(batch)}')
            raise(e)


def build_index(vecs):
    index = faiss.index_factory(embedding_size, 'SQ8')
    index.train(np.concatenate(vecs[::5]))
    for v in vecs:
        index.add(v)
    return index


def read_vectors_from_db(env):
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
            vec = np.expand_dims(v, 0)
            yield vec

def build_and_write_index(env, faiss_file):
    vecs = list(read_vectors_from_db(env))
    faiss_index = build_index(vecs)
    write_index(faiss_index, str(faiss_file))
    return faiss_index

def search_query(query, topk):
    global max_dbsize
    global faiss_index

    env = lmdb.open(str(db_path), map_size=max_dbsize)

    if faiss_index is not None:
        # skip because index is loaded
        pass
    elif not faiss_file.exists():
        raise RuntimeError('error: faiss index not found, consider rebuilding')
    else:
        faiss_index = read_index(str(faiss_file))
        if faiss_index.ntotal < env.stat()['entries']:
            raise RuntimeError('error: faiss index too small. consider rebuilding')

    assert faiss_index.ntotal == env.stat()['entries'], 'the number of faiss vectors and phrases does not match'

    search_batch = search_faiss(faiss_index, query, topk)

    st = time()
    results = []
    for neighs in search_batch:
        for nidx, n in enumerate(neighs):
            with env.begin(buffers=True) as txn:
                kvk = str(n).zfill(fill_depth).encode('utf-8')
                b = txn.get(kvk)
                s = bytes(b).decode('utf-8')
                record = json.loads(s)
                filename = record['fname']
                t = record['txt']
                results.append([int(nidx), int(n), str(filename), str(t)])

    et = time()
    eprint(f'knn search took roughly: {et-st} seconds')

    return results


def stream_files(fp):
    while True:
        line = fp.readline()
        if not line:
            break
        filepath = Path(line.strip())
        yield str(filepath)



def index_files(pdf_files, batch_size):
    global max_dbsize
    env = lmdb.open(str(db_path), map_size=max_dbsize)
    fst = stream_files(pdf_files)
    vecs = build_index_embeds(fst, batch_size)
    eprint('start indexing files')

    for i, (f,ve,t) in enumerate(vecs):
        with env.begin(write=True, buffers=True) as txn:
            k = str(i).zfill(fill_depth).encode('utf-8')
            ffp = io.StringIO()
            np.savetxt(ffp, ve)
            ffp.seek(0)
            record = {'fname':str(f), 'vec':str(ffp.read()), 'txt':t}
            vall = json.dumps(record).encode('utf-8')
            txn.put(k,vall)
            # print file, to show it has been indexed to db
            if verbosity >= 1:
                print(str(f))

    eprint('indexing files done')
    eprint('assembling search index')
    global faiss_index
    faiss_index = build_and_write_index(env, faiss_file)



CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option('-d', '--device', 'device_', type=click.Choice(['auto','cpu','cuda']), default='auto')
@click.option('-m', '--model_type', 'model_type_', type=click.Choice(['small','big']), default='big')
def cli(ctx, device_, model_type_):
    ''' a simple semantic search cli tool '''
    global device

    if device_ == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    set_model(model_type_)

@cli.command()
@click.argument('query', type=str)
@click.option('-k', '--topk', type=int, default=5)
@click.pass_context
def search(ctx, query, topk):
    ''' search a semantic document index '''
    global max_dbsize
    env = lmdb.open(str(db_path), map_size=max_dbsize)
    #print(env.stat()['entries'])
    for res in search_query(query, topk):
        nidx, n, filename, t = res
        if verbosity >= 1:
            print(f'{nidx}: {n} {filename} {t}')


@cli.command()
@click.argument('pdf_files', type=click.File('r'))
@click.option('-b', '--batch_size', type=int, default=128)
@click.pass_context
def index(ctx, pdf_files, batch_size):
    ''' build a semantic document index '''
    index_files(pdf_files, batch_size)


if __name__ == '__main__':
    cli()


# huggingface usage docs as hint
''' prompt encoding:
Here are some rules of thumb:

    Use "query: " and "passage: " correspondingly for asymmetric tasks such as passage retrieval in open QA, ad-hoc information retrieval.

    Use "query: " prefix for symmetric tasks such as semantic similarity, paraphrase retrieval.

    Use "query: " prefix if you want to use embeddings as features, such as linear probing classification, clustering.

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

# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
'''
