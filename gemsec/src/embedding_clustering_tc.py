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
    args.model = "GEMSEC"
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
    
    for id in range(1,11):
        pre = time()
        input_path = 'dataset/TC1-all_including-GT/TC1-{}/1-{}.csv'.format(id, id)
        output_path = 'dataset/TC1-all_including-GT/TC1-{}/1-{}-gemsec.json'.format(id, id)
        
        args = parameter_parser()
        args.input = input_path
        args.assignment_output = output_path 
        create_and_run_model(args)
        now = time()
        
        print('elapsed time {} {}'.format(id, now - pre))