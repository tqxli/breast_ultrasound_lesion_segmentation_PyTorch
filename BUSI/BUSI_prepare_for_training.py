import zipfile
from google_drive_downloader import GoogleDriveDownloader as gdd
import os

gdd.download_file_from_google_drive(file_id='16TpYrPxPK8_nhLmyNax6MLn7aYamMGJt',
                                    dest_path='./BUSI/BUSI_train_complete.zip',
                                    unzip=True)
                                    
with zipfile.ZipFile('BUSI/BUSI_train_complete.zip', 'r') as zip_ref:
    zip_ref.extractall('BUSI')

os.rename('BUSI/BUSI_train_complete', 'BUSI/BUSI_train')