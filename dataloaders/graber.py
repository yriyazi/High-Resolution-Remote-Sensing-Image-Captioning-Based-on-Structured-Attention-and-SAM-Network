import os
import utils
def crawl_directories(root_dir,type = utils.dataframe[utils.DatasetInUse]):
    """
    Function to crawl through directories and retrieve lists of .npz and .jpg files.

    Args:
        root_dir (str): The root directory to start crawling from.
        type (str): The type of file to filter for. Defaults to the dataset type defined in utils.

    Returns:
        tuple: A tuple containing two lists - npz_files and jpg_files.

    """
    npz_files = []
    jpg_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".npz"):
                npz_files.append(os.path.join(dirpath, filename))

            if filename.endswith(type):
                file = os.path.join(dirpath, filename).split('.')[0]
                jpg_files.append(file+type)

        
                
    return npz_files,jpg_files