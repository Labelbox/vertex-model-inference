import requests
import uuid
import ndjson
import time
from labelbox.data.serialization import LBV1Converter, NDJsonConverter
from labelbox.data.metrics.group import get_label_pairs
from labelbox.data.metrics import feature_miou_metric, feature_confusion_matrix_metric
from google.cloud import storage, aiplatform

def compute_metrics(labels, predictions, options):
    """ Computes metrics and adds metric values to predictions to-be-uploaded to Labelbox
    Args:
        labels      :       List of NDJSON ground truth labels from a model run
        predictions :       List of NDJSON prediction labels from a ETL'ed prediction job
        options     :       A dictionary where you can lookup the name of an option given the schemaId
    Returns:
        The same prediction list passed in with metrics attached, ready to-be-uploaded to a model run
    Nested Function:
        add_name_to_annotation()
    """
    predictions_with_metrics = []
    pairs = get_label_pairs(labels, predictions, filter_mismatch=True)
    for (ground_truth, prediction) in pairs.values():
        metrics = []
        for annotation in prediction.annotations:
            add_name_to_annotation(annotation, options)
        for annotation in ground_truth.annotations:
            add_name_to_annotation(annotation, options)
        metrics.extend(feature_confusion_matrix_metric(ground_truth.annotations, prediction.annotations))
        metrics.extend(feature_miou_metric(ground_truth.annotations, prediction.annotations))
        prediction.annotations.extend(metrics)
        predictions_with_metrics.append(prediction)
    return predictions_with_metrics

def add_name_to_annotation(annotation, options):
    """ Computes metrics and adds metric values to predictions to-be-uploaded to Labelbox
    Args:
        annotation      :       Annotation from a Labelbox NDJSON
        options         :       A dictionary where you can lookup the name of an option given the schemaId
    Returns:
        The same annotation with the name of the feature added in
    """    
    classification_name_lookup = {v['feature_schema_id']: k for k, v in options.items()}
    annotation.name = " "
    annotation.value.answer.name = classification_name_lookup[annotation.value.answer.feature_schema_id].replace(' ', '-')      

def export_model_run_labels(lb_client, model_run_id, media_type):
    """ Exports ground truth annotations from a model run
    Args:
        lb_client           :       Labelbox Client object
        model_run_id        :       Labelbox model run ID to pull data rows and ground truth labels from
        media_type          :       String that is either 'text' or 'image'
    Returns:
        NDJSON list of ground truth annotations from the model run
    """        
    query_str = """
        mutation exportModelRunAnnotationsPyApi($modelRunId: ID!) {
            exportModelRunAnnotations(data: {modelRunId: $modelRunId}) {
                downloadUrl createdAt status
            }
        }
        """
    url = lb_client.execute(query_str, {'modelRunId': model_run_id}, experimental=True)['exportModelRunAnnotations']['downloadUrl']
    counter = 1
    while url is None:
        counter += 1
        if counter > 10:
            raise Exception(f"Unsuccessfully got downloadUrl after {counter} attempts.")
        time.sleep(10)
        url = lb_client.execute(query_str, {'modelRunId': model_run_id}, experimental=True)['exportModelRunAnnotations']['downloadUrl']
    response = requests.get(url)
    response.raise_for_status()
    contents = ndjson.loads(response.content)
    for row in contents:
        row['media_type'] = media_type
    return LBV1Converter.deserialize(contents)

def process_predictions(batch_prediction_job, name_path_to_schema):
    """
    Args:
    Returns:    
    Nested Functions:
        build_radio_ndjson()
    """
    annotation_data = []
    for batch in batch_prediction_job.iter_outputs():
        for prediction_data in ndjson.loads(batch.download_as_string()):
            if 'error' in prediction_data:
                continue
            prediction = prediction_data['prediction']
            # only way to get data row id is to lookup from the content uri
            data_row_id = prediction_data['instance']['content'].split("/")[-1].replace(".jpg", "")
            annotation_data.append(build_radio_ndjson(prediction, name_path_to_schema, data_row_id))
    return annotation_data

def build_radio_ndjson(prediction, name_path_to_schema, data_row_id):
    """
    Args:
    Returns:
    """    
    print(f"prediction: {prediction}")
    confidences = prediction['confidences']
    print(f"confidences : {confidences}")
    argmax = confidences.index(max(confidences))
    print(f"argmax : {argmax}")
    pred_name_path = prediction['displayNames'][argmax]
    print(f"pred_name_path : {pred_name_path}")
    option_schema_id = name_path_to_schema[pred_name_path]['feature_schema_id']
    parent_schema_id = name_path_to_schema[pred_name_path]['parent_schema_id']
    return {
        "uuid": str(uuid.uuid4()),
        "answer": {
            'schemaId': option_schema_id
        },
        'dataRow': {
            "id": data_row_id
        },
        "schemaId": parent_schema_id
    }

def model_ontology_name_path_to_schema(lb_client, model_id, divider="///"):
    ontology = _get_model_ontology(lb_client, model_id)
    return _get_ontology_schema_to_name_path(ontology.normalized, invert=True, divider=divider)

def _get_model_ontology(lb_client, model_id):
    ontology_id = lb_client.execute(
    """query modelOntologyPyApi($modelId: ID!){
        model(where: {id: $modelId}) {ontologyId}}
    """, {'modelId': model_id})['model']['ontologyId']
    return lb_client.get_ontology(ontology_id)

def _get_ontology_schema_to_name_path(ontology_normalized, invert=False, divider="///"):
    """ Recursively iterates through an ontology to create a dictionary where {key=schema_id : value=name_path}; name_path = parent///answer///parent///answer....///parent///answer
    Args:
        ontology_normalized   : Required (dict) : Ontology as a dictionary from ontology.normalized (where type(ontology) = labelbox.schema.ontology.Ontology.normalized)
        invert                : Optional (bool) : If True, will invert the dictionary to be {key=name_path : value=schema_id}
    Returns:
        Dictionary where {key=schema_id : value=name_path}
    """
    def _map_layer(feature_dict={}, node_layer= [], parent_name_path="", parent_schema_id=""):
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
                name_path = parent_name_path + str(divider) + node_name if parent_name_path else node_name
                if not invert:
                    feature_dict.update({node['featureSchemaId'] : {"name_path" : name_path, "parent_name_path" : parent_name_path, "parent_schema_id" : parent_schema_id}})
                else:
                    feature_dict.update({name_path : {"feature_schema_id" : node['featureSchemaId'], "parent_name_path" : parent_name_path, "parent_schema_id" : parent_schema_id}})
                if next_layer:
                    feature_dict = _map_layer(feature_dict, next_layer, name_path, node['featureSchemaId'])
        return feature_dict
    ontology_schema_to_name_path = _map_layer(feature_dict={}, node_layer=ontology_normalized["tools"]) if ontology_normalized["tools"] else {}
    if ontology_normalized["classifications"]:
        ontology_schema_to_name_path = _map_layer(feature_dict=ontology_schema_to_name_path, node_layer=ontology_normalized["classifications"])
    if invert:
        return {v: k for k, v in ontology_schema_to_name_path.items()} 
    else:
        return ontology_schema_to_name_path

def batch_predict(etl_file, model, job_name, model_type):
    """ Creates a batch prediction job given a Vertex Model and ETL file
    Args:
        etl_file        :       File generated from ETL function 
        job_name        :       Name given to the batch prediction job
        model_type      :       Used in creating the destination URI
    Returns:
        Google Vertex Batch Prediction Job
    Nested Functions:
      parse_url()
      build_inference_fule()
    """
    bucket_name, key = parse_uri(etl_file)
    source_uri = build_inference_file(bucket_name, key)
    nowgmt = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    destination = f"gs://{bucket_name}/inference/{model_type}/{nowgmt}/"
    batch_prediction_job = model.batch_predict(
        job_display_name=job_name,
        instances_format='jsonl',
        machine_type='n1-standard-4',
        gcs_source=[source_uri],
        gcs_destination_prefix=destination,
        sync=False)
    batch_prediction_job.wait_for_resource_creation()
    while batch_prediction_job.state == aiplatform.compat.types.job_state.JobState.JOB_STATE_RUNNING:
        time.sleep(30)
    batch_prediction_job.wait()
    return batch_prediction_job

def build_inference_file(bucket_name : str, key: str) -> str:
    """ 
    Args:
        bucket_name         :        GCS bucket where the predictions will be saved
        key                 :        GCS key
    Returns:
        Inference file URL
    """        
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    # Create a blob object from the filename_path
    blob = bucket.blob(key)
    contents = ndjson.loads(blob.download_as_string())
    prediction_inputs = []
    for line in contents:
        prediction_inputs.append({
            "content": line['imageGcsUri'],
            "mimeType": "text/plain",
        })
    nowgmt = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    blob = bucket.blob(f"inference_file/bounding-box/{nowgmt}.jsonl")
    blob.upload_from_string(data=ndjson.dumps(prediction_inputs), content_type="application/jsonl")
    return f"gs://{bucket.name}/{blob.name}"

def parse_uri(etl_file):
    """ Given an etl_file URI will return the bucket name and gcs key
    Args:
        etl_file            :       String URL representing the 
    Returns:
        Google storage bucket name and gcs key
    """     
    parts = etl_file.replace("gs://", "").split("/")
    bucket_name, key = parts[0], "/".join(parts[1:])
    return bucket_name, key
