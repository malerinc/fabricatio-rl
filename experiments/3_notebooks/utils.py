import requests

from tqdm.notebook import tqdm
from os import remove
from os.path import exists
from pyunpack import Archive
from pathlib import Path

def retrieve_if_necessary(data_path):
    if not exists(data_path):
        print("Downloading...")
        download_file(
            'https://tu-dortmund.sciebo.de/s/pCqd1dAAOtUUl8f/download',
            'temp.7z', 1000000)
        print('Finished downloading file from remote.')
        tgt_path = Path(data_path).resolve().parent
        print("Extracting...")
        Archive('temp.7z').extractall(tgt_path)
        print("Finished extracting file.")
        remove('temp.7z')
        print("Removed temporary file.")
        print("Done")
    else:
        print("Required files already in place!")


def download_file(url, filename, chunk_size):
    """
    Downloads a file reachable at the download 'url' and saves it
    under 'filename'.

    This function opens a session, such that the download process
    incurrs less communication overhead.
    """
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        total_length = r.headers.get('content-length')
        print(total_length, type(total_length))
        pbar = tqdm(total=int(int(total_length) / chunk_size))
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk:
                f.write(chunk)
                pbar.update(1)
    return filename

