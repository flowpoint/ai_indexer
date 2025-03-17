import tokenizers
#from transformers import AutoTokenizer

from tqdm import tqdm
import click
import zarr
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
vectors_path = index_folder / Path('vectors.zarr')
faiss_file = index_folder / Path('faiss.index')

compile_model = False

verbosity = 1
dtype = 'float16'

model = None
tokenizer = None
faiss_index = None
embedding_size = None

# how much to fill they keys
# this allows 10^32 db-entries
fill_depth = 32
max_dbsize = int(1e12)

device = None

def average_pool_np(last_hidden_states: np.ndarray,
                 attention_mask: np.ndarray) -> np.ndarray:

    #assert last_hidden_states.shape[1:] == [512,384]
    attention_mask2 = ~np.tile(attention_mask[...,None], [1,1,last_hidden_states.shape[-1]]).astype(np.bool_)

    last_hidden = np.ma.array(last_hidden_states, mask=attention_mask2, fill_value=0.0).filled()
    return last_hidden.sum(axis=1) / attention_mask.sum(axis=1)[...,None]


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

    embedding_size = 384
    '''
    if model_type == 'big':
        model_name = 'intfloat/e5-mistral-7b-instruct'
        embedding_size = 4096
    elif model_type == 'small':
        model_name = 'intfloat/e5-small-v2'
        embedding_size = 384
    else:
        raise RuntimeError(f'unknown model type {model_type}')
    '''


def load_model():
    global model
    global tokenizer
    global embedding_size

    if model is None:
        import onnxruntime

        #model_name = 'intfloat/e5-small-v2'
        model_name = 'onnx_model_export/intfloat_e5_small_v2'
        onnx_file = 'onnx_model_export/intfloat_e5_small_v2/model.onnx'
        ##tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')
        #tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer = tokenizers.Tokenizer.from_file('onnx_model_export/intfloat_e5_small_v2/tokenizer.json')
        tokenizer.enable_truncation(512)
        tokenizer.enable_padding()
        #model = onnxruntime.InferenceSession(onnx_file)
        model = onnxruntime.InferenceSession(onnx_file, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
        #model = onnxruntime.InferenceSession(onnx_file, providers=['CUDAExecutionProvider'])


    #if tokenizer is None:
        #tokenizer = AutoTokenizer.from_pretrained(model_name)
        #tokenizer = tokenizers.Tokenizer.from_file('tok1/tokenizers.json')

def embed_batch(texts, type_):
    load_model()
    batch = texts
    input_texts = format_text(batch, type_)
    #batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='np')
    #batch_dict = dict(tokenizer(input_texts, return_tensors='np'))
    #batch_dict = dict(tokenizer.encode(input_texts, return_tensors='np'))
    batch_enc = tokenizer.encode_batch(input_texts)[0]#, return_tensors='np'))
    batch_dict = {"input_ids":np.array([batch_enc.ids]), "attention_mask":np.array([batch_enc.attention_mask])}
    #batch_dict = dict(tokenizer(input_texts)), return_tensors='np'))
    #batch_dict.pop('token_type_ids')
    outputs = model.run(None, batch_dict)
    #outputs = model(**batch_dict)

    eee = average_pool_np(outputs[0], batch_dict['attention_mask'])
    return eee


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
    index.train(vecs[::5])
    index.add(vecs)
    return index


def build_and_write_index(env, faiss_file):
    vecs = zarr.open(str(vectors_path))
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

    db_ent = env.stat()['entries']
    assert faiss_index.ntotal == env.stat()['entries'], f'the number of faiss vectors and phrases does not match: {faiss_index.ntotal} {db_ent}'

    search_batch = search_faiss(faiss_index, query, topk)

    st = time()
    results = []
    for neighs in search_batch:
        for nidx, n in enumerate(neighs):
            with env.begin(buffers=True) as txn:
                if n == -1:
                    continue
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

    vec_db = zarr.create_array(store=str(vectors_path), shape=[batch_size, embedding_size], shards=[batch_size, embedding_size],
                               dtype=dtype, chunks=[batch_size, embedding_size], overwrite=True)

    buf = np.empty([batch_size, embedding_size])

    for i, (f,vec,t) in tqdm(enumerate(vecs)):
        if i % batch_size == 0:
            txn = env.begin(write=True, buffers=True)
            i0 = i

        if i >= vec_db.shape[0]:
            vec_db.resize((vec_db.shape[0]+batch_size, vec_db.shape[1]))
        k = str(i).zfill(fill_depth).encode('utf-8')
        record = {'fname':str(f), 'txt':t}
        vall = json.dumps(record).encode('utf-8')

        buf[i%batch_size] = vec
        txn.put(k,vall)
        # print file, to show it has been indexed to db
        if verbosity >= 1:
            print(str(f))

        if i % batch_size == batch_size - 1:
            vec_db[i0:i+1] = buf
            txn.commit()

    txn.commit()

    txn = env.begin(write=True, buffers=True)
    vec_db[i0:i+1] = buf
    vec_db.resize([i+1,embedding_size])
    txn.commit()

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
    device = 'cuda'
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
    global compile_model
    compile_model = True
    index_files(pdf_files, batch_size)

@cli.command()
@click.option('-b', '--batch_size', type=int, default=128)
@click.pass_context
def build_index2(ctx, batch_size):
    ''' build a semantic document index '''
    build_and_write_index(None, faiss_file)


if __name__ == '__main__':
    cli()
