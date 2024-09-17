from typing import List
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import BaseNode

import json
import pandas as pd


def dataloader(path: str = '../json_data/', extension : str = '.txt') -> List[BaseNode]:
    node_parser = SimpleDirectoryReader(input_dir=path, required_exts=[extension])
    documents = node_parser.load_data()
    return documents


def json_to_txt(json_path: str = 'json_data/transcripts_2024.json', output_path: str = 'data/', stock_ls : list = []) -> None:
    
    """
    Loads json file and saves to text file

    Args:
        json_path (str): direct path to json file
        outptu_path (str): directory to save text files
        stock_ls (list): list of stock symbols to filter on
    """
    
    #load json
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    #flatten list
    new_data_ls = []

    for sub_ls in data:
        for i in sub_ls:
            new_data_ls.append(i)
            
    #filter on stock_ls
    df=pd.DataFrame(new_data_ls)
    
    subset = df[df['symbol'].isin(stock_ls)].drop_duplicates(subset=['symbol', 'date']).to_dict('records')
    
    for i in subset:
        file_name = output_path + str(i['year']) + '_' + str(i['quarter']) + '_' + i['symbol']+'.txt'
        with open(file_name, 'w') as file:
            file.write(i['content'])


if __name__ == '__main__':
    import ipdb;ipdb.set_trace()