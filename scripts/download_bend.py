import requests
from bs4 import BeautifulSoup
import os


base_link = 'https://sid.erda.dk/'
link = 'https://sid.erda.dk/cgi-sid/ls.py?share_id=f6hdp1zTzh&current_dir=.&flags=f'

def get_soup(link):
    source_code = requests.get(link)
    soup = BeautifulSoup(source_code.content, 'html')
    f = []
    f.extend(soup.find_all('a', {'class' : ['leftpad directoryicon', ]}))
    f.extend(soup.find_all('a', {'title' : 'open'}))
    return f


# download file in link 
def download_file(link, destination):
    r = requests.get(link, allow_redirects=True)
    open(destination, 'wb').write(r.content)



def rec(link, destination = './'):
    f = get_soup(link)
    for child in f:
        if child.get('title') == 'open':
            link = f'{base_link}{child.get("href")}'
            child_path = child.get("href")[27:]
            os.makedirs(f'{destination}/{os.path.dirname(child_path)}', exist_ok=True)
            download_file(link, f'{destination}/{child_path}')
        else:
            link = f'{base_link}cgi-sid/{child.get("href")}'
            rec(link, destination)