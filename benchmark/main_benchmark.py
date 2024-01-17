from script.prepare_data import prepare_file as pf
from script.prepare_data import prepare_graph

class Main:
    

    def __init__(self) -> None:
        pass

    
    def main(self)->None:
        prepare_graph.main()
        prepare_data = pf.PrepareData('benchmark/data/KG_edgelist_mask.txt')
        prepare_data.main()
        return None



if __name__ == "__main__":
    print("Starting now .....")
    launch = Main()
    launch.main()
    print("END")
