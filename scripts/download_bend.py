"""Download the BEND data and baseline models from the public ERDA share."""

import os
from pathlib import Path, PurePosixPath
import tempfile
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup


SHARE_ID = 'aNQa0Oz2lY'
ROOT_URL = f'https://sid.erda.dk/cgi-sid/ls.py?share_id={SHARE_ID}&current_dir=.&flags=f'
TIMEOUT = (15, 120)


def download_file(session, url, destination):
    """Stream into a temporary file so failed downloads cannot look complete."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with session.get(url, stream=True, timeout=TIMEOUT) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                temp_name = output.name
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        os.replace(temp_name, destination)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def rec(session, url, destination):
    response = session.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'lxml')
    directories = soup.find_all('a', class_='directoryicon')
    files = soup.find_all('a', title='open')
    if not directories and not files:
        print(f'No files or directories found at {url}; skipping. Check share permissions if this folder should contain files.', flush=True)
        return
    for child in files:
        file_url = urljoin(response.url, child['href'])
        path = unquote(urlparse(file_url).path)
        prefix = f'/share_redirect/{SHARE_ID}/'
        if not path.startswith(prefix):
            raise ValueError(f'Unexpected file link: {file_url}')
        relative = PurePosixPath(path[len(prefix):])
        if not relative.parts or '..' in relative.parts or relative.is_absolute():
            raise ValueError(f'Unsafe file path: {path}')
        # The share's data license must not overwrite the repository's code license.
        if relative == PurePosixPath('LICENSE'):
            relative = PurePosixPath('BEND_DATA_LICENSE')
        target = destination.joinpath(*relative.parts)
        print(target, flush=True)
        download_file(session, file_url, target)
    for child in directories:
        rec(session, urljoin(response.url, child['href']), destination)


def main():
    with requests.Session() as session:
        rec(session, ROOT_URL, Path('.'))


if __name__ == '__main__':
    main()
