from pandas import read_csv, DataFrame, concat
from typing import Union
from pathlib import Path


class KgModifier:
    """
    A class to modify knowledge graphs by removing specific relations.

    Attributes:
    - kg (DataFrame): The knowledge graph as a pandas DataFrame.
    - relations (list): List of relations to be removed from the knowledge graph.
    - path (Path): The path to save the modified knowledge graph.
    """

    def __init__(self, kg_path: str, output_path: str, relations: Union[str, list]):
        """
        Initializes the KgModifier class.

        Parameters:
        - kg_path (str): The file path to the input knowledge graph.
        - output_path (str): The file path to save the modified knowledge graph.
        - relations (Union[str, list]): The relation(s) to be removed from the knowledge graph.
        """
        self.kg = read_csv(kg_path, sep=",", low_memory=False)
        self.relations = self.convert_single_relation_to_list(relation=relations)
        self.path = Path(output_path)

    def get_kg(self) -> DataFrame:
        """
        Retrieves the knowledge graph DataFrame.

        Returns:
        - DataFrame: The knowledge graph DataFrame.
        """
        return self.kg

    def get_relations(self) -> list:
        """
        Retrieves the list of relations to be removed.

        Returns:
        - list: The list of relations to be removed.
        """
        return self.relations

    def convert_single_relation_to_list(self, relation: Union[str, list]) -> list:
        """
        Converts a single relation to a list if it's not already a list.

        Parameters:
        - relation (Union[str, list]): The relation or list of relations.

        Returns:
        - list: The list of relations.
        """
        if isinstance(relation, str):
            relation = [relation]
        return relation

    def remove_relations_from_kg(self, column: str):
        """
        Removes specified relations from the knowledge graph.

        Parameters:
        - column (str): The column name containing relations.

        Returns:
        - DataFrame: The modified knowledge graph DataFrame excluding specified relations.
        """
        relation_masks = [self.kg[column].str.contains(rel) for rel in self.relations]
        # Combine the masks with logical OR to get a mask for all relations to remove
        mask_to_remove = concat(relation_masks, axis=1).any(axis=1)
        # Invert the mask to get a mask for data excluding the specific relation(s)
        mask_to_keep = ~mask_to_remove
        # Create DataFrames based on the masks
        df_excluding_relations = self.kg[mask_to_keep]
        return df_excluding_relations

    def save_new_kg(self, kg_df: DataFrame):
        """
        Saves the modified knowledge graph DataFrame to a CSV file.

        Parameters:
        - kg_df (DataFrame): The modified knowledge graph DataFrame.
        """
        kg_df.to_csv(self.path, sep=",", index=False)

    def main(self):
        """
        Main method to execute the removal of specified relations from the knowledge graph.
        """
        new_kg = self.remove_relations_from_kg("relation")
        self.save_new_kg(new_kg)


if __name__ == "__main__":
    list_disease_rel = ["contraindication", "indication", "off-label use", "disease_phenotype_negative",
                        "disease_phenotype_positive", "disease_protein", "disease_disease", "exposure_disease"]
    list_drugs_rel = ["contraindication", "drug_drug", "indication", "off-label use", "drug_protein", "drug_effect"]

    kg_path = "/home/thomas/Documents/projects/kge_project/benchmark/data/kg_giant_orphanet.csv"

    remove_drugs_rel = KgModifier(
        kg_path=kg_path,
        output_path="alignement/data/split_kg/disease_kg.csv",
        relations=list_drugs_rel)
    kg = remove_drugs_rel.get_kg()
    remove_drugs_rel.main()

    remove_diseases_rel = KgModifier(
        kg_path=kg_path,
        output_path="alignement/data/split_kg/drugs_kg.csv",
        relations=list_disease_rel)

    remove_diseases_rel.main()
