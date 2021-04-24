import zipfile
from google_drive_downloader import GoogleDriveDownloader as gdd
import os

# gdd.download_file_from_google_drive(file_id='16TpYrPxPK8_nhLmyNax6MLn7aYamMGJt',
#                                     dest_path='./BUSI/BUSI_train_complete.zip',
#                                     unzip=True)

# os.rename('BUSI/BUSI_train_complete', 'BUSI/BUSI_train')

gdd.download_file_from_google_drive(file_id='1xF5a5LE3PbJbT-Qv0sJiUa307h-kStKf',
                                    dest_path='./BUSI/BUSI_new.zip',
                                    unzip=True)

os.rename('BUSI/BUSI_new', 'BUSI/BUSI_train')