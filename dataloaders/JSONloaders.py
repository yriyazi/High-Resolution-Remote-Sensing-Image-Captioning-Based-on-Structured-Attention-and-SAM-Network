import json

class JSONloaders():
    """
    Class for loading and manipulating JSON data.

    Args:
        file_path (str): The path to the JSON file.

    Attributes:
        data (dict): The loaded JSON data.
        list_names (list): A list of image filenames.

    """
    def __init__(self,
                 file_path:str) -> None:
        """
        Initializes an instance of JSONloaders.

        Opens and loads the JSON file specified by file_path.
        Extracts the image filenames and stores them in list_names.

        Args:
            file_path (str): The path to the JSON file.

        """
        with open(file_path, 'r') as file:
            self.data = json.load(file)
        self.list_names = self.__name_grabber()
        
    def item(self,
             file_name:str):
        """
        Retrieves the file name and associated sentences for a given image file name.

        Args:
            file_name (str): The image file name to retrieve data for.

        Returns:
            tuple: A tuple containing the file name and a list of sentences associated with the image.

        """
        id = self.list_names.index(file_name)
        return self._item(id)
    
    def _item(self,
              id:int):
        """
        Retrieves the file name and associated sentences for a given image ID.

        Args:
            id (int): The image ID to retrieve data for.

        Returns:
            tuple: A tuple containing the file name and a list of sentences associated with the image.

        """
        image_id = id
        file_name = self.data['images'][image_id]['filename']
        
        sentenses = self.__Sentence_graber(self.data['images'][image_id]['sentences'])

        return file_name,sentenses
    
    def __Sentence_graber(Self,root):
        """
        Retrieves the sentences from the given root object.

        Args:
            root: The root object containing the sentences.

        Returns:
            list: A list of sentences.

        """
        sentenses = None
        for l in root:
            if not sentenses:
                sentenses = []
                # print(l)
            sentenses.append(l['raw'])
        return sentenses
    
    def __name_grabber(self,):
        """
        Retrieves the image filenames from the loaded JSON data.

        Returns:
            list: A list of image filenames.

        """
        names = []
        for data in self.data['images']:
            names.append(data['filename'])
        return names
    
    def _all_Sentences(self,):
        """
        Retrieves all sentences from the loaded JSON data.

        Returns:
            list: A list of all sentences.

        """
        sentenses = None
        for l in self.data['images']:
            if not sentenses:
                sentenses = []
            sentenses.extend(self.__Sentence_graber(l['sentences']))
        
        return sentenses
    
    def splitter(self):
        """
        Splits the image filenames into train, validation, and test sets based on the 'split' value in the JSON data.

        Returns:
            tuple: A tuple containing three lists - train, validation, and test image filenames.

        """
        train = []
        validation = []
        test = []
        for i in self.data['images']:
            if i['split']=='train':
                train.append(i['filename'])
            elif i['split']=='val' :
                validation.append(i['filename'])
            elif i['split']=='test' :
                test.append(i['filename'])
                
        return train,validation,test
            