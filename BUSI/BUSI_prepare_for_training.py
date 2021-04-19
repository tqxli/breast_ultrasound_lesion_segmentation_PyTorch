import zipfile
from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1_JTLKi8bxfv-DeCvJWiZrsdxi2eCRF9r',
                                    dest_path='./BUSI/BUSI_train.zip',
                                    unzip=True)
                                    
with zipfile.ZipFile('BUSI/BUSI_train.zip', 'r') as zip_ref:
    zip_ref.extractall('BUSI')