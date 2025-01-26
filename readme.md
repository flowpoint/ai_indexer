# ai_indexer

a small tool for semantic indexing and searching

* only pdfs, txt and .md for now
* requires pdftotext command to be available
* requires lmdb

## install

```
pip install .
```

## usage

index some files.
for each embedded chunk, the filename will be printed to stdout consecutively.
at the end, the index will be "assembled"

```
ls ~/Documents/*.pdf ~/Documents/*.txt | ai_indexer index -
```

search the index.

```
ai_indexer search 'hello'
```
