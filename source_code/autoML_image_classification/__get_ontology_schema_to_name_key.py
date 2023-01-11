def __get_ontology_schema_to_name_path(ontology_normalized, invert=False):
    """ Recursively iterates through an ontology to create a dictionary where {key=schema_id : value=name_path}; name_path = parent///answer///parent///answer....///parent///answer
    Args:
        ontology_normalized   : Required (dict) : Ontology as a dictionary from ontology.normalized (where type(ontology) = labelbox.schema.ontology.Ontology.normalized)
        invert                : Optional (bool) : If True, will invert the dictionary to be {key=name_path : value=schema_id}
    Returns:
        Dictionary where {key=schema_id : value=name_path}
    """
    def __map_layer(feature_dict={}, node_layer= [], parent_name_path=""):
        """ Recursive function that does the following for each node in a node_layer:
                1. Creates a name_path given the parent_name_path
                2. Adds schema_id : name_path to your working dictionary
                3. If there's another layer for a given node, recursively calls itself, passing it its own name key as it's childrens' parent_name_path
        Args:
            feature_dict              :     Dictionary where {key=schema_id : value=name_path}
            node_layer                :     A list of classifications, tools, or option dictionaries
            parent_name_path           :     A concatenated list of parent node names separated with "///" creating a unique mapping key
        Returns:
            feature_dict
        """
        if node_layer:
            for node in node_layer:
                if "tool" in node.keys():
                    node_name = node["name"]
                    next_layer = node["classifications"]
                elif "instructions" in node.keys():
                    node_name = node["instructions"]
                    next_layer = node["options"]
                else:
                    node_name = node["label"]
                    next_layer = node["options"] if 'options' in node.keys() else []
                if parent_name_path:
                    name_path = parent_name_path + "///" + node_name
                else:
                    name_path = node_name
                feature_dict.update({node['featureSchemaId'] : name_path})
                if next_layer:
                    feature_dict = __map_layer(feature_dict, next_layer, name_path)
        return feature_dict
    ontology_schema_to_name_path = __map_layer(feature_dict={}, node_layer=ontology_normalized["tools"]) if ontology_normalized["tools"] else {}
    if ontology_normalized["classifications"]:
        ontology_schema_to_name_path = __map_layer(feature_dict=ontology_schema_to_name_path, node_layer=ontology_normalized["classifications"])
    if invert:
        return {v: k for k, v in ontology_schema_to_name_path.items()} 
    else:
        return ontology_schema_to_name_path