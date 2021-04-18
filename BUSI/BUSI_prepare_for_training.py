import zipfile

with zipfile.ZipFile('BUSI/BUSI_train.zip', 'r') as zip_ref:
    zip_ref.extractall('BUSI')