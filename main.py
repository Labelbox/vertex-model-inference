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
        if "mal-inference" in model_type.lower():
            print(f"Unlabeled data inference job") 
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
        else:
            print(f"Post-Train Inference Job")
            etl_file = request_json['etl_file'] 
            # Select model type to determine which functions to import     
            if model_type == "autoML_image_classification":
                from source_code.autoML_image_classification.inference import batch_predict, model_ontology_name_path_to_schema, process_predictions, export_model_run_labels, compute_metrics
            elif model_type == "custom_image_classification":
                from source_code.custom_image_classification.inference import batch_predict, model_ontology_name_path_to_schema, process_predictions, export_model_run_labels, compute_metrics 
            # Grab a Vertex Model using the model_name variable
            model = aiplatform.Model.list(filter=f'display_name={model_name}')[0]      
            # Create a Vertex prediction job  
            prediction_job = batch_predict(etl_file, model, lb_model_run_id, "radio")
            print('Predictions generated. Converting predictions into Labelbox format.')
            # Get a dictionary where {key=name_path (divider="///"): value=feature_schema_id}
            ontology_name_path_to_schema = model_ontology_name_path_to_schema(model_run.model_id, lb_client, divider="_")
            # Turn the results of the prediction job into Labelbox Label objects
            predictions = list(NDJsonConverter.deserialize(process_predictions(prediction_job, ontology_name_path_to_schema)))
            print('Predictions reformatted. Exporting ground truth labels from model run.')
            # Export your model run labels and compute performance metrics
            labels = export_model_run_labels(lb_client, lb_model_run_id, 'image')
            print('Computing metrics.')
            predictions_with_metrics_ndjson = list(NDJsonConverter.serialize(compute_metrics(labels, predictions, ontology_name_path_to_schema)))
            print('Metrics computed. Uploading predictions and metrics to model run.')
            # Upload predictions with metrics to model run
            upload_task = model_run.add_predictions(f'diagnostics-import-{uuid.uuid4()}', predictions_with_metrics_ndjson)
            print(upload_task.statuses)
            model_run.update_status("COMPLETE")  
            print('Inference job complete.')

    except Exception as e:
        print(e)
        model_run.update_status("FAILED") 
    
    return "Inference Job"

def monitor_function(request):
    """ Periodically checks a training job to see if it's completed, canceled, paused or failing
    Args (passed from previous functions):
        lb_api_key          : Required (str) : Labelbox API Key
        lb_model_run_id     : Required (str) : Labelbox Model Run ID
        model_name          : Required (str) : Desired Vertex Model Name
        inference_url       : Required (str) : URL to trigger Inference Cloud Function
        monitor_url         : Required (str) : URL to trigger Monitor Cloud Function
        model_type          : Required (str) : Labelbox Model Type (whatever was selected from the 'Train Model' button in the UI)
    Returns:
        Will either send the model training pipeline to inference or terminate the model training pipeline
    """    
    import requests
    import json
    import time
    from google.cloud import aiplatform
    from source_code.config import get_lb_client
    
    # Receive data from trigger    
    request_bytes = request.get_data()
    request_json = json.loads(request_bytes)    
    
    # Parse data from trigger    
    lb_api_key = request_json['lb_api_key']    
    model_name = request_json["model_name"]
    lb_model_run_id = request_json['lb_model_run_id']    
    inference_url = request_json["inference_url"]
    monitor_url = request_json["monitor_url"]
    model_type = request_json['model_type']
    
    # Configure environment
    lb_client = get_lb_client(lb_api_key)  
    model_run = lb_client.get_model_run(lb_model_run_id)

    try:
        # Select model type to determine which list of training jobs to check
        if model_type == "autoML_image_classification":
            training_job = aiplatform.AutoMLImageTrainingJob.list(filter=f'display_name={model_name}')[0]
        elif model_type == "custom_image_classification":
            training_job = aiplatform.CustomTrainingJob.list(filter=f'display_name={model_name}')[0]
        # Will wait 5 minutes before checking model run status  
        time.sleep(300)
        # Check the training job state
        job_state = str(training_job.state)
        completed_states = [
            "PipelineState.PIPELINE_STATE_FAILED",
            "PipelineState.PIPELINE_STATE_CANCELLED",
            "PipelineState.PIPELINE_STATE_PAUSED",
            "PipelineState.PIPELINE_STATE_CANCELLING"
        ]
        print(f'Current Job State: {job_state}')    
        # Trigger different functions depending on the training job state
        if job_state == "PipelineState.PIPELINE_STATE_SUCCEEDED":
            print('Training compete, sent to inference.')
            requests.post(inference_url, data=request_bytes)
        elif job_state in completed_states:
            print("Training failed, terminating deployment.")
            model_run.update_status("FAILED")
        else:
            print('Training incomplete, will check again in 5 minutes.')
            requests.post(monitor_url, data=request_bytes)

    except:
        model_run.update_status("FAILED")            

    return "Monitor Job"

def train_function(request):
    """ Initiates the training job in Vertex
    Args (passed from previous functions):
        lb_api_key          : Required (str) : Labelbox API Key
        lb_model_run_id     : Required (str) : Labelbox Model Run ID
        etl_file            : Required (str) : URL to the ETL'ed data row / ground truth data from a Labelbox Model Run    
        model_name          : Required (str) : Desired Vertex Model Name
        monitor_url         : Required (str) : URL to trigger Monitor Cloud Function
        model_type          : Required (str) : Labelbox Model Type (whatever was selected from the 'Train Model' button in the UI)
    Returns:
        Creates a vertex dataset and launches a Vertex training job, triggers the monitor function
    """   
    import json
    import requests
    from source_code.config import get_lb_client    
    
    # Receive data from trigger
    request_bytes = request.get_data()
    request_json = json.loads(request_bytes)

    print(request_json)
    
    # Parse data from trigger
    etl_file = request_json['etl_file']
    lb_api_key = request_json['lb_api_key']
    lb_model_run_id = request_json['lb_model_run_id']
    model_name = request_json['model_name']
    monitor_url = request_json['monitor_url']
    model_type = request_json['model_type']
    
    # Configure environment
    lb_client = get_lb_client(lb_api_key)
    model_run = lb_client.get_model_run(lb_model_run_id)   

    try:
        # Select model type to determine which functions to import
        if model_type == "autoML_image_classification":
            from source_code.autoML_image_classification.train import create_vertex_dataset, create_training_job
        elif model_type == "custom_image_classification":
            from source_code.custom_image_classification.train import create_vertex_dataset, create_training_job 
        # Create a Vertex Dataset
        vertex_dataset = create_vertex_dataset(lb_model_run_id, etl_file)
        # Create a Verted Model
        vertex_model, vertex_model_id = create_training_job(f"{model_type}--{model_name}", vertex_dataset, lb_model_run_id)
        # Update Model Run status
        model_run.update_status("TRAINING_MODEL")          
        print('Training launched, sent to monitor function.')                                                              
        print(f"Job Name: {lb_model_run_id}")
        print(f'Vertex Model ID: {vertex_model_id}')
        # Trigger Monitor Function 
        requests.post(monitor_url, data=request_bytes)

    except Exception as e:
        print(e)
        model_run.update_status("FAILED")

    return "Train Job"

def etl_function(request):
    """ Exports data rows and ground truth labels from a model run, generates an ETL file in a storage bucket and launches training
    Args (passed from previous functions):
        lb_api_key          : Required (str) : Labelbox API Key
        lb_model_run_id     : Required (str) : Labelbox Model Run ID
        gcs_bucket          : Required (str) : GCS Bucket to save Vertex ETL file to
        gcs_region          : Required (str) : GCS Region where the cloud function, GCS bucket, and model training instance are located
        model_name          : Required (str) : Desired Vertex Model Name
        train_url           : Required (str) : URL to trigger Train Cloud Function
        monitor_url         : Required (str) : URL to trigger Monitor Cloud Function
        inference_url       : Required (str) : URL to trigger Inference Cloud Function
        model_type          : Required (str) : Labelbox Model Type (whatever was selected from the 'Train Model' button in the UI)
    Returns:
        Google Bucket with an ETL file representing the data rows and ground truth labels from the model run
        Dictionary that gets passed through the other functions
    """
    import json
    import requests    
    from labelbox import Client
    from google.cloud import storage, aiplatform    
    from source_code.config import get_lb_client, get_gcs_client, create_gcs_key
    
    # Receive data from trigger 
    request_bytes = request.get_data()
    request_json = json.loads(request_bytes)
    
    # Read environment variables   
    lb_api_key = request_json["lb_api_key"]
    google_project = request_json["google_project"]
    gcs_bucket = request_json['gcs_bucket']
    train_url = request_json['train_url']
    gcs_region = request_json['gcs_region']
    lb_model_run_id = request_json['lb_model_run_id']
    model_type = request_json["model_type"]
    
    # Configure environment
    lb_client = get_lb_client(lb_api_key)
    gcs_client = get_gcs_client(google_project)   
    model_run = lb_client.get_model_run(lb_model_run_id)
    model_run.update_status("PREPARING_DATA")
    
    try:
        # Select model type to determine which functions to import
        if model_type == "autoML_image_classification":
            from source_code.autoML_image_classification.etl import etl_job, upload_ndjson_data
        elif model_type == "custom_image_classification":
            from source_code.custom_image_classification.etl import etl_job, upload_ndjson_data
        # Get or create a GCS Bucket to store ETL Data   
        gcs_key = create_gcs_key(lb_model_run_id)
        try:
            bucket = gcs_client.get_bucket(gcs_bucket)
        except: 
            print(f"Bucket does not exsit, will create one with name {gcs_bucket}")
            bucket = gcs_client.create_bucket(gcs_bucket, location=gcs_region)
        # Create JSON of ETL Data
        json_data = etl_job(lb_client, lb_model_run_id, bucket)
        # Upload JSON to GCS Bucket
        etl_file = upload_ndjson_data(json_data, bucket, gcs_key)
        # Send data to Train Function 
        request_json['etl_file'] = etl_file
        post_bytes = json.dumps(request_json).encode('utf-8')
        requests.post(train_url, data=post_bytes)
        print(f"ETL Complete. Training Job Initiated.")

    except Exception as e:
        print("ETL Function Failed. Check your configuration and try again.")
        print(e)
        model_run.update_status("FAILED")

    return "ETL Job"

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
        etl_url             : Required (str) : URL to trigger ETL Cloud Function
        train_url           : Required (str) : URL to trigger Train Cloud Function
        monitor_url         : Required (str) : URL to trigger Monitor Cloud Function
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
            "train_url" : env_vars("train_url"),
            "monitor_url" : env_vars("monitor_url"),
            "inference_url" : env_vars("inference_url")
        }
        # Update model run status
        lb_client = get_lb_client(post_dict["lb_api_key"])
        lb_client.get_model_run(lb_model_run_id).update_status("EXPORTING_DATA")
        # Determine if this is model training execution of a model inference execution
        if "mal-inference" in model_type.lower():
            print(f"Sending for inference")
            post_url = env_vars("inference_url")
        else:
            print(f"Sending for ETL")
            post_url = env_vars("etl_url")
        # Send data to ETL or Inference Function            
        if "" in post_dict.keys():
            print("Missing Environemnt Variables. Check your Environment Variables and try again.")
        else:
            requests.post(post_url, data=json.dumps(post_dict).encode('utf-8'))

    except Exception as e:
        print("Model Run Function Failed. Check your Environment Variables and try again.")
        print(e)

    return "Rerouting to ETL"

def models(request):
    """ Serves a list of model and inference options in the UI
    Args:
        model_options       :           A list of model names you want to appear in the Labelbox UI
    """
    from google.cloud import aiplatform

    ## Input list of model training options here
    training_options = [ 
        "autoML_image_classification"
        # "custom_image_classification"
    ]

    # Input list of model inference options here
    models = aiplatform.Model.list()                      
    inference_options = [f"MAL-Inference: {model.display_name}" for model in models]

    all_options = training_options + inference_options

    return {option : [] for option in all_options}
