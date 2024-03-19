import glob
import os

def read_files_in_directory(directory_path):
    file_paths = glob.glob(os.path.join(directory_path, '*'))
    _Dump = []
    for file_path in file_paths:
      _Dump.append(file_path)
    return  _Dump