import os
import tensorflow as tf
import zipfile
from six.moves import urllib
from tempfile import gettempdir

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    local_filename = os.path.join(gettempdir(), filename)
    if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(url + filename,
                                                       local_filename)
    print("retreived..")
    statinfo = os.stat(local_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + local_filename +
                        '. Can you get to it with a browser?')
    return local_filename

# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        pay = f.read(f.namelist()[0])
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

def main():
    local_filename = maybe_download('text8.zip', 31344016)
    data = read_data(local_filename)
    print(data)
    print(local_filename)

if __name__ == "__main__":
    main()
