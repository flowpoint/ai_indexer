# ai_indexer

a small tool for semantic indexing and searching

* only pdfs for now
* requires pdftotext command to be available
* requires lmdb

## install

pip install .

## index:

ls ~/Documents/*.pdf | ai_indexer index -

ai_indexer search 'hello'
