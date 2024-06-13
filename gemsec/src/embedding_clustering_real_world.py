"""Running the model."""

from param_parser import parameter_parser
from print_and_read import graph_reader
from model import GEMSECWithRegularization, GEMSEC
from model import DeepWalkWithRegularization, DeepWalk
from time import time 

def create_and_run_model(args):
    """
    Function to read the graph, create an embedding and train it.
    """
    graph = graph_reader(args.input)
    args.model = "GEMSECWithRegularization"
    if args.model == "GEMSECWithRegularization":
        model = GEMSECWithRegularization(args, graph)
    elif args.model == "GEMSEC":
        model = GEMSEC(args, graph)
    elif args.model == "DeepWalkWithRegularization":
        model = DeepWalkWithRegularization(args, graph)
    else:
        model = DeepWalk(args, graph)
    model.train()

if __name__ == "__main__":
    
    folders = ['dolphin', 'football', 'karate', 'mexican', 'polbooks', 'railway', 'strike']
    
    for folder in folders: 
        pre = time()
        input_path = 'dataset/real-world dataset/{}/{}.csv'.format(folder, folder)
        output_path = 'dataset/real-world dataset/{}/{}-gemsec-regu.json'.format(folder,folder)
        
        args = parameter_parser()
        args.input = input_path
        args.assignment_output = output_path 
        create_and_run_model(args)
        now = time()
        
        print('elapsed time {} {}'.format(id, now - pre))