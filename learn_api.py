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
