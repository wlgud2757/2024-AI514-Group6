"""Running the model."""

from param_parser import parameter_parser
from print_and_read import graph_reader
from model import GEMSECWithRegularization, GEMSEC
from model import DeepWalkWithRegularization, DeepWalk

#     
# python /home2/s20235025/bigdata/GEMSEC/src/embedding_clustering.py --input /home2/s20235025/bigdata/GEMSEC/data/TC/1-1.csv --embedding-output /home2/s20235025/bigdata/TC1-all_including-GT/TC1-1/1-1-gemsec.dat


def create_and_run_model(args):
    """
    Function to read the graph, create an embedding and train it.
    """
    graph = graph_reader(args.input)
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
    args = parameter_parser()
    create_and_run_model(args)
