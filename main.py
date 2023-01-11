def inference_function(request):
    """ Generates and uploads predictions to a given model run
    Args (passed from previous functions):
        lb_api_key          : Required (str) : Labelbox API Key
        lb_model_run_id     : Required (str) : Labelbox Model Run ID
        model_name          : Required (str) : Desired Vertex Model Name
        model_type          : Required (str) : Labelbox Model Type (whatever was selected from the 'Train Model' button in the UI)
        etl_file            : Optional (str) : URL to the ETL'ed data row / ground truth data from a Labelbox Model Run - required if training, created if inferring
    """

    import json
    import uuid
    from labelbox import Client
    from labelbox.data.serialization import NDJsonConverter
    from google.cloud import aiplatform    
    from source_code.config import get_lb_client, get_gcs_client, create_gcs_key

    # Receive data from trigger
    request_bytes = request.get_data()
    request_json = json.loads(request_bytes)
    
    # Parse data from trigger    
    lb_api_key = request_json['lb_api_key']    
    model_name = request_json['model_name'] 
    google_project = request_json["google_project"]
    gcs_region = request_json['gcs_region']
    lb_model_run_id = request_json['lb_model_run_id'] 
    model_type = request_json['model_type']
    
    lb_client = get_lb_client(lb_api_key)
    model_run = lb_client.get_model_run(lb_model_run_id)
    
    try:
        # Determine if this is model training execution of a model inference execution (information is a nested string in model_type)
        print(f"Begin data inference job") 
        gcs_bucket = request_json['gcs_bucket']
        print(f"Get or creating a GCS bucket with name {gcs_bucket} to store ETL file")
        gcs_key = create_gcs_key(lb_model_run_id)
        gcs_client = get_gcs_client(google_project)
        try:
            bucket = gcs_client.get_bucket(gcs_bucket)
        except: 
            print(f"Bucket does not exsit, will create one with name {gcs_bucket}")
            bucket = gcs_client.create_bucket(gcs_bucket, location=gcs_region)            
        model_display_name = model_type[15:]
        # Assumes that the model ETL type is in the model name, separated by a "--"
        etl_type = model_display_name.split("--")[0]
        print(f"Model Name: {model_display_name}\nModel Type: {etl_type}")
        model = aiplatform.Model.list(filter=f'display_name={model_display_name}')[0]
        if etl_type == "autoML_image_classification":
            from source_code.autoML_image_classification.etl import etl_job, upload_ndjson_data
            from source_code.autoML_image_classification.inference import batch_predict, model_ontology_name_path_to_schema, process_predictions                
        elif etl_type == "custom_image_classification":
            from source_code.custom_image_classification.etl import etl_job, upload_ndjson_data  
            from source_code.custom_image_classification.inference import batch_predict, model_ontology_name_path_to_schema, process_predictions            
        print(f"Creating ETL File for Inference Job")
        json_data = etl_job(lb_client, lb_model_run_id, bucket, how="inference")
        etl_file = upload_ndjson_data(json_data, bucket, gcs_key)
        print(f"ETL file generated. Batch predict time")
        prediction_job = batch_predict(etl_file, model, lb_model_run_id, "radio")
        print('Predictions generated. Converting predictions into Labelbox format.')
        # Dictionary where {key=name_path : value=schema_id}
        ontology_name_path_to_schema = model_ontology_name_path_to_schema(lb_client, model_run.model_id, divider="_")
        print(f"Converting model predictions to Labelbox format")
        predictions_ndjson = process_predictions(prediction_job, ontology_name_path_to_schema)
        print(f"Uploading predictions to model run")
        upload_task = model_run.add_predictions(f'inference-import-{uuid.uuid4()}', predictions_ndjson)
        print(upload_task.statuses)
        model_run.update_status("COMPLETE") 
        print('Inference job complete.')

    except Exception as e:
        print(e)
        model_run.update_status("FAILED") 
    
    return "Inference Job"

def model_run(request):
    """ Reroutes the webhook trigger to the ETL function
    Args:
        modelId             : Required (str) : Labelbox Model ID
        modelRunId          : Required (str) : Labelbox Model Run ID
        modelType           : Required (str) : Labelbox Model Type (whatever was selected from the 'Train Model' button in the UI)
    Environment Variables:
        gcs_bucket          : Required (str) : GCS Bucket to save Vertex ETL file to
        gcs_region          : Required (str) : GCS Region where the cloud function, GCS bucket, and model training instance are located
        lb_api_key          : Required (str) : Labelbox API Key
        model_name          : Required (str) : Desired Vertex Model Name
        inference_url       : Required (str) : URL to trigger Inference Cloud Function
        project_id          : Optional (str) : Project ID to run inference on (required if modelType is Inference-based)

    """
    import requests
    import json
    from source_code.config import env_vars, get_lb_client

    # Receive data from trigger   
    request_bytes = request.get_data()
    request_json = json.loads(request_bytes)
    
    # Parse data from trigger    
    lb_model_id = request_json['modelId']
    lb_model_run_id = request_json['modelRunId']
    model_type = request_json['modelType']
    
    try:
        # Structure config variable json
        post_dict = {
            "model_type" : model_type,
            "lb_model_id" : lb_model_id,
            "lb_model_run_id" : lb_model_run_id,
            "gcs_bucket" : env_vars("gcs_bucket"),
            "gcs_region" : env_vars("gcs_region"),
            "lb_api_key" : env_vars("lb_api_key"),
            "google_project" : env_vars("google_project"),
            "model_name" : env_vars("model_name"),
            "inference_url" : env_vars("inference_url")
        }
        # Update model run status
        lb_client = get_lb_client(post_dict["lb_api_key"])
        lb_client.get_model_run(lb_model_run_id).update_status("EXPORTING_DATA")
        # Determine if this is model training execution of a model inference execution
        print(f"Sending for inference")
        post_url = env_vars("inference_url")
        # Send data to ETL or Inference Function            
        if "" in post_dict.keys():
            print("Missing Environemnt Variables. Check your Environment Variables and try again.")
        else:
            requests.post(post_url, data=json.dumps(post_dict).encode('utf-8'))

    except Exception as e:
        print("Model Run Function Failed. Check your Environment Variables and try again.")
        print(e)

    return "Model Run Function"

def models(request):
    """ Serves a list of model and inference options in the UI
    Args:
        model_options       :           A list of model names you want to appear in the Labelbox UI
    """
    from google.cloud import aiplatform

    # Input list of model inference options here
    models = aiplatform.Model.list()                      
    inference_options = [f"MAL-Inference: {model.display_name}" for model in models]

    return {option : [] for option in inference_options}
