import urllib.request
import os

import sys


def reporthook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(total_size)), read_so_far, total_size)
        sys.stderr.write(s)
        if read_so_far >= total_size:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (read_so_far,))


def download_file(file_path, url_path):
    if not os.path.exists(file_path):
        print('file does not exist, downloading from internet')
        urllib.request.urlretrieve(url=url_path, filename=file_path,
                                   reporthook=reporthook)
