import requests

uil = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
parems = {
    'db': 'pubmed',
    'term': 'cancer',
    'retmode': 'json',
    'retmax': 5
}

response = requests.get(uil, params=parems)
data = response.json()
print(data)
print(f'find {data["esearchresult"]["count"]} articles')

from Bio import Entrez
import os, time, csv
from tpdm import tqdm
import pandas as pd

Entrez.email = "zhanbing2025@gmail.com"
Entrez.api_key = os.environ.get('NCBI_API_KEY')  # set your NCBI API key in environment variable
RATE_SLEEP = 0.34  # NCBI allows up to 3 requests per second with an API key

query = "machine learning [Title] AND 2020:2025 [dp]"
handle = Entrez.esearch(db="pubmed", term=query, retmax=1000)

record = Entrez.read(handle)
pmids = record["IdList"]

handle.close()
print(f'Found {len(pmids)} PMIDs')

def fetch_summaries(pmids_chunk):
    hanlde = Entrez.efetch(db="pubmed", id=",".join(pmids_chunk), rettye="xml")
    recorda = Entrez.read(handle)
    handle.close()
    return records

results = []
batch_size = 100
for i in tqdm(range(0, len(pmids),batch_size)):
    chunk = pmids[i:i+batch_size]
    records = fetch_summaries(chunk)
    for article in records[PubmedArticle]:
        pmid = article['MedlineCitation']['PMID']
        title = article['MedlineCitation']['Article']['ArticleTitle']
        abstract = ''
        if 'abstract' in title and title['Abstract'] and 'AbstractText' in title['Abstract']:
            abstract = ' '.join(title['Abstract']['AbstractText'])
        journal = title.get('Journal', {}).get('Title', '')
        #authors
        authors = []
        for a in title.get('AuthorList', []):
            name_parts = []
            if 'ForeName' in a:
                name_parts.append(a['ForeName'])
            if 'LastName' in a:
                name_parts.append(a['LastName'])
            authors.append(' '.join(name_parts))
        results.append({'pmid': pmid, 'title': title, 'abstract': abstract})

import pandas as pdP