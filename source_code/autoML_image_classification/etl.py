import os
import json
import requests
from io import BytesIO
from typing import Tuple, Optional, Callable, Dict, Any, List
from PIL.Image import Image, open as load_image, DecompressionBombError
from concurrent.futures import ThreadPoolExecutor, as_completed
from labelbox import Client
from labelbox.data.annotation_types import Label, Radio
from google.cloud import storage
from google.api_core.retry import Retry
from source_code.config import get_labels_for_model_run
from source_code.errors import MissingEnvironmentVariableException, InvalidDataRowException, InvalidLabelException

def etl_job(lb_client: Client, object_id: str, bucket: storage.Bucket, how="training"):
    """ Creates a json file that is used for input into a vertex ai training job
    Args:    
        lb_client           : Required (labelbox.Client) : Labelbox Client object
        object_id           : Required (str) : Labelbox Model ID or Project ID to pull labels from
        bucket              : Required (google.cloud.storage.Bucket) : Cloud storage bucket used to upload image data to
        how                 : Optional (str) : Determines where to pull the ontology from - either "training" or "inference"
    Retuns:
        stringified ndjson
    Nested Functions:
        get_labels_for_model_run()
        process_in_threadpool()
        process_label()
        process_data_row()
    """
    if how == "training":
        labels = get_labels_for_model_run(lb_client, object_id, media_type='image', strip_subclasses=True)
        vertex_labels = process_in_threadpool(process_label, labels, bucket)
    else:
        model_run = lb_client.get_model_run(object_id)
        data_rows = [
            {"data_row_id" : label['DataRow ID'] , "url" : label['Labeled Data']} for label in model_run.export_labels(download=True)
        ]
        vertex_labels = process_in_threadpool(process_data_row, data_rows, bucket)
    return "\n".join([json.dumps(label) for label in vertex_labels])

def process_in_threadpool(process_fn: Callable[..., Dict[str, Any]], items: List[Label], *args, max_workers = 8) -> List[Dict[str, Any]]:
    """ Function for running etl processing in parallel
    Args:
        process_fn          : Required (function) : Function to execute in parallel
        items               : Required (list) : List of items to run in parallel
        args                : Optional (**) : Args that are passed through to the process_fn
        max_workers         : Optional (int) : How many threads should be used
    Returns:
        A list of results from the process_fn
    """
    print('Processing Labels')
    vertex_labels = []
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        training_data_futures = (exc.submit(process_fn, item, *args) for item in items)
        filter_count = {'labels' : 0,'data_rows' : 0}
        for future in as_completed(training_data_futures):
            try:
                vertex_labels.append(future.result())
            except InvalidDataRowException:
                filter_count['data_rows'] += 1
            except InvalidLabelException:
                filter_count['labels'] += 1
    print('Label Processing Complete')                
    return vertex_labels

def process_data_row(data_row: dict, bucket: storage.Bucket, downsample_factor = 2.) -> Dict[str, Any]:
    """ Function for converting a Labelbox Data Row object into a vertex json label for radio classification
    Args:
        data_row            : Required (dict) : Data row as-dict from project.export_queued_data_rows()
        bucket              : Required (storage.Bucket) : GCS Storage Bucket object
        downsample_factor   : Optional (float) Amount to downsample images if assets are too large
    Returns:
        Dict representing a vertex label for a prediction job, the label itself is a placeholder
    Nested Functions:
        get_image_bytes()     
        upload_image_to_gcs()            
    """
    row_data_url = data_row['url']
    image_bytes, _ = get_image_bytes(row_data_url, downsample_factor)
    classification = {'displayName': 'no_label'}
    gcs_uri = upload_image_to_gcs(image_bytes, row_data_url, bucket)
    return {
        'imageGcsUri': gcs_uri,
        'classificationAnnotation': classification,
        'dataItemResourceLabels': {
            "aiplatform.googleapis.com/ml_use": "test",
            "dataRowId": data_row['data_row_id']
        }
    }

def process_label(label: Label, bucket: storage.Bucket, downsample_factor = 2.) -> Dict[str, Any]:
    """ Function for converting a labelbox Label object into a vertex json label for radio classification.
    Args:
        label: the label to convert
        bucket: cloud storage bucket to write image data to
        downsample_factor: how much to scale the images by before running inference
    Returns:
        Dict representing a vertex label
    Nested Functions:
        get_image_bytes()     
        upload_image_to_gcs()
    """
    PARTITION_MAPPING = {
        'training': 'train',
        'test': 'test',
        'validation': 'validation'
    }    
    classifications = []
    image_bytes, _ = get_image_bytes(label.data.url, downsample_factor)
    for annotation in label.annotations:
        if isinstance(annotation.value, Radio):
            classifications.append({
                "displayName":
                    f"{annotation.name}_{annotation.value.answer.name}"
            })
    if len(classifications) > 1:
        raise InvalidLabelException(
            "Skipping example. Must provide <= 1 classification per image.")
    elif len(classifications) == 0:
        classification = {'displayName': 'no_label'}
    else:
        classification = classifications[0]
    gcs_uri = upload_image_to_gcs(image_bytes, label.data.uid, bucket)
    return {
        'imageGcsUri': gcs_uri,
        'classificationAnnotation': classification,
        'dataItemResourceLabels': {
            "aiplatform.googleapis.com/ml_use": PARTITION_MAPPING[label.extra.get("Data Split")],
            "dataRowId": label.data.uid
        }
    }

def get_image_bytes(image_url: str, downsample_factor = 1.) -> Optional[Tuple[Image, Tuple[int,int]]]:
    """ Fetch image bytes from a url and optionally resize the image.
    Args:
        image_url: A url that references an image
        downsample_factor: How much to scale the image by.
            The new dimensions will be (width * 1/downsample_factor, height * 1/downsample_factor)
    Returns:
        The resized PIL Image
    Nested Functions:
        _download_image()
        image_to_bytes()
    """
    try:
        with _download_image(image_url) as image:
            w,h = image.size
            with image.resize((int(w *  1./downsample_factor), int(h * 1./downsample_factor))) as resized_image:
                image_bytes = image_to_bytes(resized_image)
                return image_bytes, (w,h)
    except DecompressionBombError:
        raise InvalidDataRowException(f"Image too large : `{image_url}`.")
    except:        
        raise InvalidDataRowException(f"Unable to fetch image : `{image_url}`.")

@Retry()
def _download_image(image_url: str) -> Image:
    """ Downloads as a PIL Image object
    Args:
        image_url       :       String of a URL to-be-downloaded as a PIL Image object
    Returns:
        Image as a PIL Image object
    """
    return load_image(BytesIO(requests.get(image_url).content))

def image_to_bytes(im: Image) -> BytesIO:
    """ Converts a PIL Image object into Bytes
    Args:
        im              :       PIL Image object
    Returns:   
        Image converted into bytes
    """
    im_bytes = BytesIO()
    im.save(im_bytes, format="jpeg")
    im_bytes.seek(0)
    return im_bytes

@Retry()
def upload_image_to_gcs(image_bytes: BytesIO, data_row_id: str, bucket: storage.Bucket, dims: Optional[Tuple[int,int]] = None) -> str:
    """ Uploads images to gcs. Vertex will not work unless the input data is a gcs_uri in a regional bucket hosted in us-central1.
    Args:
        image_bytes     :       Image as bytes
        data_row_id     :       The id of the image being processed
        bucket          :       Cloud storage bucket object
        dims            :       Optional image dimensions to encode in the filename (used later for reverse etl)
    Returns:
        gcs uri for an image
    """
    if dims is not None:
        # Dims is currently used during inference to scale the prediction
        # When we have media attributes we should use that instead.
        w, h = dims
        suffix = f"_{int(w)}_{int(h)}"
    else:
        suffix = ""
    gcs_key = f"training/images/{data_row_id}{suffix}.jpg"
    blob = bucket.blob(gcs_key)
    blob.upload_from_file(image_bytes, content_type="image/jpg")
    return f"gs://{bucket.name}/{blob.name}"

@Retry()
def upload_ndjson_data(stringified_json : str, bucket: storage.Bucket, gcs_key : str) -> str:
    """
    Uploads ndjson to gcs
    Args:
        stringified_json: ndjson string to write to gcs
        bucket: Cloud storage bucket object
        gcs_key: cloud storage key (basically the file name)
    """
    print('Uploading Converted Labels')
    blob = bucket.blob(gcs_key)
    blob.upload_from_string(stringified_json)
    print('Upload Complete')
    return f"gs://{bucket.name}/{blob.name}"
