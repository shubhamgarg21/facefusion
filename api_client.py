import requests
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

class FaceFusionClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def face_swap(self, source_image_path: str, target_file_path: str, **options):
        """
        Perform face swapping
        """
        url = f"{self.base_url}/face-swap"
        
        with open(source_image_path, 'rb') as source_file, \
             open(target_file_path, 'rb') as target_file:
            
            files = {
                'source_image': source_file,
                'target_file': target_file
            }
            
            data = options
            
            response = requests.post(url, files=files, data=data)
            return response.json()
    
    def face_enhance(self, target_file_path: str, source_image_path: Optional[str] = None, **options):
        """
        Perform face enhancement
        source_image_path is optional - if provided, only enhances faces similar to reference
        """
        url = f"{self.base_url}/face-enhance"
        
        files = {
            'target_file': open(target_file_path, 'rb')
        }
        
        # Add source file if provided
        if source_image_path:
            files['source_image'] = open(source_image_path, 'rb')
        
        try:
            data = options
            response = requests.post(url, files=files, data=data)
            return response.json()
        finally:
            # Close all opened files
            for file_obj in files.values():
                file_obj.close()
    
    def face_enhance_with_reference(self, source_image_path: str, target_file_path: str, **options):
        """
        Perform face enhancement with reference image (convenience method)
        """
        return self.face_enhance(target_file_path, source_image_path, **options)
    
    def face_enhance_all(self, target_file_path: str, **options):
        """
        Perform face enhancement on all faces (convenience method)
        """
        return self.face_enhance(target_file_path, None, **options)
    
    def frame_enhance(self, target_file_path: str, **options):
        """
        Perform frame enhancement (no source image needed)
        """
        url = f"{self.base_url}/frame-enhance"
        
        with open(target_file_path, 'rb') as target_file:
            files = {
                'target_file': target_file
            }
            
            data = options
            
            response = requests.post(url, files=files, data=data)
            return response.json()
    
    def frame_colorize(self, target_file_path: str, **options):
        """
        Perform frame colorization (no source image needed)
        """
        url = f"{self.base_url}/frame-colorize"
        
        with open(target_file_path, 'rb') as target_file:
            files = {
                'target_file': target_file
            }
            
            data = options
            
            response = requests.post(url, files=files, data=data)
            return response.json()
    
    def face_debug(self, target_file_path: str, source_image_path: Optional[str] = None, **options):
        """
        Perform face debugging
        source_image_path is optional - if provided, only debugs faces similar to reference
        """
        url = f"{self.base_url}/face-debug"
        
        files = {
            'target_file': open(target_file_path, 'rb')
        }
        
        # Add source file if provided
        if source_image_path:
            files['source_image'] = open(source_image_path, 'rb')
        
        try:
            data = options
            response = requests.post(url, files=files, data=data)
            return response.json()
        finally:
            # Close all opened files
            for file_obj in files.values():
                file_obj.close()
    
    def batch_face_swap(self, source_image_path: str, target_file_paths: List[str], **options):
        """
        Perform batch face swapping on multiple target files
        """
        url = f"{self.base_url}/batch-face-swap"
        
        files = {'source_image': open(source_image_path, 'rb')}
        
        # Add multiple target files
        for i, target_path in enumerate(target_file_paths):
            files[f'target_files'] = open(target_path, 'rb')
        
        try:
            data = options
            response = requests.post(url, files=files, data=data)
            return response.json()
        finally:
            # Close all opened files
            for file_obj in files.values():
                file_obj.close()
    
    def get_job_status(self, job_id: str):
        """
        Get job status
        """
        url = f"{self.base_url}/jobs/{job_id}/status"
        response = requests.get(url)
        return response.json()
    
    def list_all_jobs(self):
        """
        List all jobs
        """
        url = f"{self.base_url}/jobs"
        response = requests.get(url)
        return response.json()
    
    def delete_job(self, job_id: str):
        """
        Delete a job and its files
        """
        url = f"{self.base_url}/jobs/{job_id}"
        response = requests.delete(url)
        return response.json()

    def download_result(self, job_id: str, output_path: str):
        """
        Download result file
        """
        url = f"{self.base_url}/jobs/{job_id}/download"
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
        return False

    def wait_for_completion(self, job_id: str, timeout: int = 300):
        """
        Wait for job completion
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)
            
            if status['status'] == 'completed':
                return True
            elif status['status'] == 'failed':
                print(f"Job failed: {status.get('error_message', 'Unknown error')}")
                return False
            
            time.sleep(2)
        
        print("Job timed out")
        return False
    
    # Model and information endpoints
    def get_health(self):
        """
        Get API health status
        """
        url = f"{self.base_url}/health"
        response = requests.get(url)
        return response.json()
    
    def get_processors(self):
        """
        Get available processors
        """
        url = f"{self.base_url}/processors"
        response = requests.get(url)
        return response.json()
    
    def get_face_detector_models(self):
        """
        Get available face detector models
        """
        url = f"{self.base_url}/models/face-detector"
        response = requests.get(url)
        return response.json()
    
    def get_face_swapper_models(self):
        """
        Get available face swapper models
        """
        url = f"{self.base_url}/models/face-swapper"
        response = requests.get(url)
        return response.json()
    
    def get_face_enhancer_models(self):
        """
        Get available face enhancer models
        """
        url = f"{self.base_url}/models/face-enhancer"
        response = requests.get(url)
        return response.json()
    
    def get_frame_enhancer_models(self):
        """
        Get available frame enhancer models
        """
        url = f"{self.base_url}/models/frame-enhancer"
        response = requests.get(url)
        return response.json()
    
    def get_frame_colorizer_models(self):
        """
        Get available frame colorizer models
        """
        url = f"{self.base_url}/models/frame-colorizer"
        response = requests.get(url)
        return response.json()
    
    def get_frame_colorizer_sizes(self):
        """
        Get available frame colorizer sizes
        """
        url = f"{self.base_url}/models/frame-colorizer-sizes"
        response = requests.get(url)
        return response.json()
    
    def get_face_debugger_items(self):
        """
        Get available face debugger items
        """
        url = f"{self.base_url}/models/face-debugger-items"
        response = requests.get(url)
        return response.json()
    
    def get_available_models(self):
        """
        Get all available models (legacy method, maintained for compatibility)
        """
        return self.get_face_swapper_models()
    
    def get_all_models(self):
        """
        Get all available models for all processors
        """
        return {
            'face_detector': self.get_face_detector_models(),
            'face_swapper': self.get_face_swapper_models(),
            'face_enhancer': self.get_face_enhancer_models(),
            'frame_enhancer': self.get_frame_enhancer_models(),
            'frame_colorizer': self.get_frame_colorizer_models(),
            'frame_colorizer_sizes': self.get_frame_colorizer_sizes(),
            'face_debugger_items': self.get_face_debugger_items()
        }
    
    # Convenience methods for common workflows
    def enhance_face_and_download(self, target_file_path: str, output_path: str, 
                                 source_image_path: Optional[str] = None, **options):
        """
        Convenience method: enhance face and download result
        source_image_path is optional - if provided, only enhances faces similar to reference
        """
        result = self.face_enhance(target_file_path, source_image_path, **options)
        job_id = result.get('job_id')
        
        if job_id and self.wait_for_completion(job_id):
            return self.download_result(job_id, output_path)
        return False
    
    def enhance_frame_and_download(self, target_file_path: str, output_path: str, **options):
        """
        Convenience method: enhance frame and download result
        """
        result = self.frame_enhance(target_file_path, **options)
        job_id = result.get('job_id')
        
        if job_id and self.wait_for_completion(job_id):
            return self.download_result(job_id, output_path)
        return False
    
    def colorize_frame_and_download(self, target_file_path: str, output_path: str, **options):
        """
        Convenience method: colorize frame and download result
        """
        result = self.frame_colorize(target_file_path, **options)
        job_id = result.get('job_id')
        
        if job_id and self.wait_for_completion(job_id):
            return self.download_result(job_id, output_path)
        return False
    
    def debug_face_and_download(self, target_file_path: str, output_path: str, 
                               source_image_path: Optional[str] = None, **options):
        """
        Convenience method: debug face and download result
        source_image_path is optional - if provided, only debugs faces similar to reference
        """
        result = self.face_debug(target_file_path, source_image_path, **options)
        job_id = result.get('job_id')
        
        if job_id and self.wait_for_completion(job_id):
            return self.download_result(job_id, output_path)
        return False


# Example usage with all processors
if __name__ == "__main__":
    client = FaceFusionClient()
    
    print("=== FaceFusion Client Examples ===")
    
    # Test API health
    try:
        health = client.get_health()
        print(f"API Health: {health}")
        
        # Get all available processors and models
        processors = client.get_processors()
        print(f"Available processors: {processors}")
        
        all_models = client.get_all_models()
        print("Available models:")
        for processor_type, models in all_models.items():
            print(f"  {processor_type}: {models}")
        
    except Exception as e:
        print(f"Error connecting to API: {e}")
        exit(1)
    
    # Face swap
    print("\n=== Face Swap Example ===")
    result = client.face_swap("", "")
    client.wait_for_completion(result["job_id"])
    client.download_result(result["job_id"], "")
    print("Image downloaded to downloaded_image.png")

    # Face enhance the output image
    result = client.face_enhance("")
    client.wait_for_completion(result["job_id"])
    client.download_result(result["job_id"], "")
    print("Image enhanced to downloaded_image_enhanced.png")


    # Example 1: Face Enhancement
    print("\n=== Face Enhancement Example ===")
    try:
        result = client.face_enhance(
            source_image_path="source.jpg",
            target_file_path="target.jpg",
            face_enhancer_model="gfpgan_1.4",
            face_enhancer_blend=85,
            face_enhancer_weight=0.9,
            output_image_quality=95
        )
        
        job_id = result.get('job_id')
        if job_id:
            print(f"Face enhancement job started: {job_id}")
            
            if client.wait_for_completion(job_id):
                print("Job completed successfully!")
                if client.download_result(job_id, "enhanced_result.jpg"):
                    print("Enhanced result downloaded to enhanced_result.jpg")
        
    except FileNotFoundError:
        print("Note: source.jpg and target.jpg files needed for this example")
    except Exception as e:
        print(f"Face enhancement error: {e}")
    
    # Example 2: Frame Enhancement (no source needed)
    print("\n=== Frame Enhancement Example ===")
    try:
        result = client.frame_enhance(
            target_file_path="low_res_video.mp4",
            frame_enhancer_model="real_esrgan_x4",
            frame_enhancer_blend=90,
            output_video_quality=85
        )
        
        job_id = result.get('job_id')
        if job_id:
            print(f"Frame enhancement job started: {job_id}")
            # Note: In practice, you'd wait for completion and download
        
    except FileNotFoundError:
        print("Note: low_res_video.mp4 file needed for this example")
    except Exception as e:
        print(f"Frame enhancement error: {e}")
    
    # Example 3: Frame Colorization
    print("\n=== Frame Colorization Example ===")
    try:
        result = client.frame_colorize(
            target_file_path="bw_video.mp4",
            frame_colorizer_model="ddcolor",
            frame_colorizer_size="512x512",
            frame_colorizer_blend=95
        )
        
        job_id = result.get('job_id')
        if job_id:
            print(f"Frame colorization job started: {job_id}")
        
    except FileNotFoundError:
        print("Note: bw_video.mp4 file needed for this example")
    except Exception as e:
        print(f"Frame colorization error: {e}")
    
    # Example 4: Face Debugging
    print("\n=== Face Debugging Example ===")
    try:
        debugger_items = client.get_face_debugger_items()
        print(f"Available debug items: {debugger_items}")
        
        result = client.face_debug(
            target_file_path="target.jpg",
            face_debugger_items="bounding-box,face-landmark-5/68,face-mask,age,gender"
        )
        
        job_id = result.get('job_id')
        if job_id:
            print(f"Face debugging job started: {job_id}")
        
    except FileNotFoundError:
        print("Note: target.jpg file needed for this example")
    except Exception as e:
        print(f"Face debugging error: {e}")
    
    # Example 5: Using convenience methods
    print("\n=== Convenience Method Example ===")
    try:
        success = client.enhance_face_and_download(
            target_file_path="target.jpg",
            output_path="convenience_result.jpg",
            source_image_path="source.jpg",
            face_enhancer_model="gfpgan_1.4",
            face_enhancer_blend=80
        )
        
        if success:
            print("Face enhancement completed and downloaded using convenience method!")
        
    except FileNotFoundError:
        print("Note: source.jpg and target.jpg files needed for this example")
    except Exception as e:
        print(f"Convenience method error: {e}")
    
    # List all jobs
    try:
        jobs = client.list_all_jobs()
        print(f"\nAll jobs: {json.dumps(jobs, indent=2)}")
    except Exception as e:
        print(f"Error listing jobs: {e}") 