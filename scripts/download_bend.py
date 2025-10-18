import argparse
from typing import List
import requests
from bs4 import BeautifulSoup
import os

BASE_LINK = 'https://sid.erda.dk/'
LINK = 'https://sid.erda.dk/cgi-sid/ls.py?share_id=f6hdp1zTzh&current_dir=.&flags=f'

def get_soup(link: str):
    source_code = requests.get(link)
    soup = BeautifulSoup(source_code.content, "lxml")
    f = []
    f.extend(soup.find_all('a', {'class' : ['leftpad directoryicon', ]}))
    f.extend(soup.find_all('a', {'title' : 'open'}))
    return f

# download file in link 
def download_file(link: str, destination: str):
    r = requests.get(link, allow_redirects=True)
    open(destination, 'wb').write(r.content)

def rec(
    link: str,
    destination: str = './',
    whitelist: List[str] = [],
):
    f = get_soup(link)
    for child in f:
        if child.get('title') == 'open':
            link = f'{BASE_LINK}{child.get("href")}'
            child_path = child.get("href")[27:]
            if not any(map(lambda x: x in link, whitelist)) and len(whitelist) > 0: continue
            os.makedirs(f'{destination}/{os.path.dirname(child_path)}', exist_ok=True)
            print(f'{destination}/{child_path}')
            download_file(link, f'{destination}/{child_path}')
        else:
            link = f'{BASE_LINK}cgi-sid/{child.get("href")}'
            rec(link, destination, whitelist=whitelist)

def main(args):
    print('DOWNLOADING')
    rec(LINK, destination = '.', whitelist=args.use_cases.split(","))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--use-cases", default="pretrained_models,variant_effects,genomes", type=str, help="Comma-delimited whitelist of substrings for downloading.")
    args = parser.parse_args()
    main(args)
