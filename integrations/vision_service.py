import requests
import json
from datetime import datetime
import uuid
import os
from dotenv import load_dotenv
import urllib3

# Suppress SSL certificate warnings for development
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

VISION_API_BASE_URL = "https://vision-api.visin.eu/api"

def get_auth_headers():
    """Get authorization headers for API requests."""
    token = os.getenv('VISIN_TOKEN')
    if token:
        return {'Authorization': f'Bearer {token}'}
    return {}

def get_training_by_uuid(training_uuid):
    """
    Get training ID by UUID.
    
    Args:
        training_uuid (str): UUID of the training
    
    Returns:
        str or None: Training ID (_id) if found, None if not found or failed
    """
    url = f"{VISION_API_BASE_URL}/trainings/uuid/{training_uuid}"
    
    try:
        response = requests.get(url, headers=get_auth_headers(), timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            training_id = data.get("data", {}).get("_id")
            return training_id
        else:
            print(f"Training not found for UUID {training_uuid}: {data.get('message')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed when getting training by UUID: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse response JSON: {e}")
        return None


def create_training(uuid, name, model, dataset, description=None, status="running", tags=None, config_id=None):
    """
    Create a new training run in the vision service.
    
    Args:
        uuid (str): Unique identifier for the training
        name (str): Name of the training run
        model (str): Model type (e.g., 'clft', 'clfcn')
        dataset (str): Dataset name (e.g., 'zod', 'waymo')
        description (str, optional): Description of the training
        status (str, optional): Initial status, defaults to 'pending'
        tags (list, optional): List of tags for the training
        config_id (str, optional): ID of the associated config
    
    Returns:
        str or None: Training ID if successful, None if failed
    """
    url = f"{VISION_API_BASE_URL}/trainings"
    
    payload = {
        "uuid": uuid,
        "status": status,
        "name": name,
        "model": model,
        "dataset": dataset
    }
    
    if description:
        payload["description"] = description
    
    if config_id:
        payload["configId"] = config_id
    
    if tags:
        payload["tags"] = tags
    
    try:
        response = requests.post(url, json=payload, headers=get_auth_headers(), timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            training_id = data.get("data", {}).get("_id")
            return training_id
        else:
            print(f"Failed to create training: {data.get('message')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed when creating training: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse response JSON: {e}")
        return None

def send_epoch_results_from_file(training_id, epoch_num, results_file_path):
    """
    Send epoch results from a logged JSON file to the vision service.
    
    Args:
        training_id (str): ID of the training run
        epoch_num (int): Epoch number
        results_file_path (str): Path to the JSON file containing epoch results
    
    Returns:
        bool: True if successful, False if failed
    """
    try:
        # Read the logged JSON file
        with open(results_file_path, 'r') as f:
            payload = json.load(f)
        
        # Update the training_uuid to match the training_id
        payload["training_uuid"] = training_id
        payload["trainingId"] = training_id
        
        url = f"{VISION_API_BASE_URL}/epochs/upload"
        
        response = requests.post(url, json=payload, headers=get_auth_headers(), timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return True
        else:
            print(f"Failed to send epoch results: {data.get('message')}")
            return False
    except FileNotFoundError:
        print(f"Results file not found: {results_file_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"Failed to parse results JSON file: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Request failed when sending epoch results: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error when sending epoch results: {e}")
        return False

def create_config(name, config_data, config_uuid=None):
    """
    Create a new configuration in the vision service.
    
    Args:
        name (str): Name of the configuration
        config_data (dict): The configuration data
        config_uuid (str, optional): UUID for the config, auto-generated if not provided
    
    Returns:
        str or None: Config ID if successful, None if failed
    """
    if config_uuid is None:
        config_uuid = str(uuid.uuid4())
    
    # Use the upload endpoint for config data
    url = f"{VISION_API_BASE_URL}/configs/upload"
    
    # Extract summary from config
    summary = config_data.get('Summary', f"Training config for {name}")
    
    # Extract config name from the name parameter (remove extra parts)
    config_name = name.replace(' Config', '').split(' - ')[-1]  # Extract just the config part
    
    payload = {
        "config_data": config_data,
        "Summary": summary,
        "config_name": config_name
    }
    
    try:
        print(f"Creating config with config_name={config_name}, Summary={summary}")
        response = requests.post(url, json=payload, headers=get_auth_headers(), timeout=30, verify=False)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            config_id = data.get("data", {}).get("_id")
            return config_id
        else:
            print(f"Failed to create config: {data.get('message')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed when creating config: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse response JSON: {e}")
        return None

def send_test_results_from_file(results_file_path):
    """
    Send test results from a logged JSON file to the vision service.
    
    Args:
        results_file_path (str): Path to the JSON file containing test results
    
    Returns:
        bool: True if successful, False if failed
    """
    try:
        # Read the logged JSON file
        with open(results_file_path, 'r') as f:
            payload = json.load(f)
        
        url = f"{VISION_API_BASE_URL}/test-results/upload"
        
        response = requests.post(url, json=payload, headers=get_auth_headers(), timeout=30, verify=False)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            print(f"Successfully uploaded test results to vision service")
            return True
        else:
            print(f"Failed to upload test results: {data.get('message')}")
            return False
    except FileNotFoundError:
        print(f"Test results file not found: {results_file_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"Failed to parse test results JSON file: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Request failed when uploading test results: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error when uploading test results: {e}")
        return False

def send_benchmark_results_from_file(results_file_path, training_uuid=None, epoch_uuid=None, epoch=None):
    """
    Send benchmark results from a logged JSON file to the vision service.
    
    Args:
        results_file_path (str): Path to the JSON file containing benchmark results
        training_uuid (str, optional): UUID of the training run this benchmark belongs to
        epoch_uuid (str, optional): UUID of the epoch this benchmark belongs to
        epoch (int, optional): Epoch number this benchmark belongs to
    
    Returns:
        bool: True if successful, False if failed
    """
    try:
        # Read the logged JSON file
        with open(results_file_path, 'r') as f:
            data = json.load(f)
        
        # Convert timestamp to ISO format if needed
        timestamp = data.get('timestamp')
        if timestamp and 'T' not in timestamp:
            # Convert '2023-01-01 00:00:00' to '2023-01-01T00:00:00.000Z'
            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            timestamp = dt.isoformat() + '.000Z'
        
        payload = {
            'timestamp': timestamp,
            'system_info': data.get('system_info', {}),
            'results': data.get('results', [])
        }
        
        # Add optional fields if provided
        if training_uuid:
            payload['training_uuid'] = training_uuid
        if epoch_uuid:
            payload['epoch_uuid'] = epoch_uuid
        if epoch is not None:
            payload['epoch'] = epoch
        
        url = f"{VISION_API_BASE_URL}/benchmarks"
        
        response = requests.post(url, json=payload, headers=get_auth_headers(), timeout=30, verify=False)
        response.raise_for_status()
        response_data = response.json()
        if response_data.get("success"):
            print(f"Successfully uploaded benchmark results to vision service")
            return True
        else:
            print(f"Failed to upload benchmark results: {response_data.get('message')}")
            return False
    except FileNotFoundError:
        print(f"Benchmark results file not found: {results_file_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"Failed to parse benchmark results JSON file: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Request failed when uploading benchmark results: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error when uploading benchmark results: {e}")
        return False

def upload_visualization(epoch_uuid, file_path, viz_type, metadata=None):
    """
    Upload a visualization file to the vision service.
    
    This function:
    1. Requests a signed upload URL from the API
    2. Uploads the file directly to MinIO
    3. Creates the visualization record in the database
    
    Args:
        epoch_uuid (str): UUID of the epoch this visualization belongs to
        file_path (str): Path to the visualization file
        viz_type (str): Type of visualization (e.g., 'segment', 'overlay', 'compare', 'correct_only')
        metadata (dict, optional): Additional metadata for the visualization
    
    Returns:
        dict or None: Visualization data if successful, None if failed
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    
    # Determine mimetype based on extension
    ext = os.path.splitext(file_path)[1].lower()
    mimetype_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg'
    }
    mimetype = mimetype_map.get(ext, 'image/png')
    
    try:
        # Step 1: Get signed upload URL
        viz_uuid = str(uuid.uuid4())
        upload_url_request = {
            "epoch_uuid": epoch_uuid,
            "filename": filename,
            "type": viz_type,
            "mimetype": mimetype
        }
        
        url = f"{VISION_API_BASE_URL}/visualizations/upload-url"
        response = requests.post(url, json=upload_url_request, headers=get_auth_headers(), timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("success"):
            print(f"Failed to get upload URL: {data.get('message')}")
            return None
        
        upload_url = data["data"]["uploadUrl"]
        viz_uuid = data["data"]["visualization_uuid"]
        minio_file_id = data["data"]["minioFileId"]
        
        # Step 2: Upload file to MinIO
        with open(file_path, 'rb') as f:
            upload_response = requests.put(
                upload_url, 
                data=f,
                headers={'Content-Type': mimetype},
                timeout=30
            )
            upload_response.raise_for_status()
        
        # Step 3: Create visualization record
        create_request = {
            "epoch_uuid": epoch_uuid,
            "visualization_uuid": viz_uuid,
            "filename": filename,
            "type": viz_type,
            "minioFileId": minio_file_id,
            "mimetype": mimetype,
            "size": file_size
        }
        
        if metadata:
            create_request["metadata"] = metadata
        
        url = f"{VISION_API_BASE_URL}/visualizations"
        response = requests.post(url, json=create_request, headers=get_auth_headers(), timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            print(f"Successfully uploaded {viz_type} visualization: {filename}")
            return data.get("data")
        else:
            print(f"Failed to create visualization record: {data.get('message')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed when uploading visualization: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return None
    except Exception as e:
        print(f"Unexpected error when uploading visualization: {e}")
        return None
