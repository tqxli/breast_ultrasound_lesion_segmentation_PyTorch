import zipfile
from google_drive_downloader import GoogleDriveDownloader as gdd
import os

# gdd.download_file_from_google_drive(file_id='16TpYrPxPK8_nhLmyNax6MLn7aYamMGJt',
#                                     dest_path='./BUSI/BUSI_train_complete.zip',
#                                     unzip=True)
                                    
# with zipfile.ZipFile('BUSI/BUSI_train_complete.zip', 'r') as zip_ref:
#     zip_ref.extractall('BUSI')

# os.rename('BUSI/BUSI_train_complete', 'BUSI/BUSI_train')

gdd.download_file_from_google_drive(file_id='1cBVd68Gs35lewYLrJzBFj6NhvU4nuja-',
                                    dest_path='./BUSI/BUSI_train_single_mass.zip',
                                    unzip=True)
                                    
#with zipfile.ZipFile('BUSI/BUSI_train_single_mass.zip', 'r') as zip_ref:
#    zip_ref.extractall('BUSI')

os.rename('BUSI/BUSI_train_single_mass', 'BUSI/BUSI_train')