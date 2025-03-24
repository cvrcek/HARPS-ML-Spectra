import os
import sys
import pdb

import requests
import cgi
import json
import pandas

import getpass


def getDispositionFilename( response ):
    """Get the filename from the Content-Disposition in the response's http header"""
    contentdisposition = response.headers.get('Content-Disposition')
    if contentdisposition == None:
        return None
    value, params = cgi.parse_header(contentdisposition)
    filename = params["filename"]
    return filename

def writeFile( response ):
    """Write on disk the retrieved file"""
    if response.status_code == 200:
        # The ESO filename can be found in the response header
        filename = getDispositionFilename( response )
        # Let's write on disk the downloaded FITS spectrum using the ESO filename:
        with open(filename, 'wb') as f:
            f.write(response.content)
        return filename

df = pandas.read_csv('harps_metadata_and_labels.csv')

if 1:
    n_samples = 500
else:
    n_samples = len(df['dp_id'])

for i in range(n_samples):
    if os.path.isfile('./' + df['dp_id'][i] + '.fits'):
        print('file already downloaded!')
        continue
    file_url = 'https://dataportal.eso.org/dataportal_new/file/' + df['dp_id'][i]
    response = requests.get(file_url)
    filename = writeFile(response)
    if filename:
        print("Saved file: %s" % (filename))
    else:
        print("Could not get file (status: %d)" % (response.status_code))
