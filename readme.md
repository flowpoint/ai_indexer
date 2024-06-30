# aigrep

requires pdftotext command to be available
requires lmdb

QUERY='codellama'

ls ~/Documents/*.pdf | ai_indexer - $QUERY
