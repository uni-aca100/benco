from IMDLBenCo.registry import MODELS
import torch.nn as nn
import torch
import json
import urllib.request
import urllib.error
import time
from multiprocessing import shared_memory
import numpy as np

SHM_NAME = "benco-img-shm"

@MODELS.register_module()
class MyModel(nn.Module):
    def __init__(
        self,
        MyModel_Customized_param: int,
        pre_trained_weights: str,
        model_endpoint: str = "http://localhost:8000/pred/",
        request_retry_timeout: int = 90,
        request_retries: int = 2,
        request_retry_initial_delay: int = 30,
        request_retry_delay_factor: int = 2,
    ) -> None:
        """
        The parameters of the `__init__` function will be automatically converted into the parameters expected by the argparser in the training and testing scripts by the framework according to their annotated types and variable names. 
        
        In other words, you can directly pass in parameters with the same names and types from the `run.sh` script to initialize the model.
        """
        super().__init__()
        self.inOutShm = None

        self.count_forward_calls = 0
        self.count_dummy_outputs = 0
        
        self.MyModel_Customized_param = MyModel_Customized_param
        self.pre_trained_weights = pre_trained_weights

        self.model_endpoint = model_endpoint
        self.req_timeout = request_retry_timeout
        self.req_retries = request_retries
        self.req_retry_initial_delay = request_retry_initial_delay
        self.req_retry_delay_factor = request_retry_delay_factor
        self.loss_func_a = nn.BCEWithLogitsLoss()

    def _init_shared_memory(self, required_bytes):
        """
        initializes or resizes a shared memory segment for a square image of given size.
        """
        if self.inOutShm is not None and self.inOutShm.size >= required_bytes:
            return  # No need to reinitialize if the existing shared memory is sufficient

        # not enough space, we need to reinitialize
        if self.inOutShm is not None:
            self.inOutShm.close()
            self.inOutShm = None

        # remove the existing shared memory segment if it exists
        try:
            temp_shm = shared_memory.SharedMemory(name=SHM_NAME)
            temp_shm.close()
            temp_shm.unlink()
        except FileNotFoundError:
            pass 

        self.inOutShm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=required_bytes)

    def _get_dummy_output(self, img: torch.Tensor):
        """
        Returns a dummy output in case of error to avoid crashing the framework.
        """
        self.count_dummy_outputs += 1
        print(f"Returning dummy output on forward call {self.count_forward_calls}")
        return {
            "pred_mask": torch.zeros_like(img).to(img.device),
            "pred_label": torch.tensor(0.0, dtype=img.dtype).to(img.device)
        }

    def _pred(self, img: torch.Tensor, mask_dtype: np.dtype, label_dtype: torch.dtype):
        # convert tensor to numpy array reformating the shape to [B, H, W, C] with the same dtype as the input tensor
        raw_img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
        # Revert from the default ImageNet normalization to the range [0, 255] RGB 8-bit format
        # decouple the normalization from the model (heterogeneous models could need the original image like mllm models)
        raw_img = raw_img * np.array([0.229, 0.224, 0.225], dtype=raw_img.dtype).reshape(1, 1, 1, 3)
        raw_img = raw_img + np.array([0.485, 0.456, 0.406], dtype=raw_img.dtype).reshape(1, 1, 1, 3)
        raw_img = (raw_img * 255.0).clip(0, 255).astype(np.uint8)

        # Calculate the mask size in bytes (B, 1, H, W) with the same dtype as the input mask tensor
        mask_nbytes = np.dtype(mask_dtype).itemsize * img.shape[0] * 1 * img.shape[2] * img.shape[3]
        self._init_shared_memory(max(raw_img.nbytes, mask_nbytes))

        try:
            shm_array = np.ndarray(raw_img.shape, dtype=raw_img.dtype, buffer=self.inOutShm.buf)
            np.copyto(shm_array, raw_img)

        except Exception as e:
            print(f"Error while copying to shared memory: {e}")
            # return a dummy output in case of error to avoid crashing the framework
            return self._get_dummy_output(img)

        # 4. Invia la notifica HTTP
        json_data = json.dumps({
            "shm_name": SHM_NAME,
            "img_shape": list(raw_img.shape),
            "img_dtype": str(raw_img.dtype),
            "mask_dtype": str(mask_dtype),
        }).encode("utf-8")
        req = urllib.request.Request(
            self.model_endpoint, data=json_data, headers={"Content-Type": "application/json"}, method="POST"
        )
        
        # total attempts = 1 initial try + configured number of retries
        max_attempts = max(1, self.req_retries + 1)
        req_delay = self.req_retry_initial_delay
        for attempt in range(1, max_attempts + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.req_timeout) as response:
                    res_json = json.loads(response.read().decode("utf-8"))

                    if res_json.get("status") != "success":
                        raise RuntimeError(f"Server returned an error: {res_json.get('message', 'Unknown error')}")
                    else:
                        print(f"Server model request processed successfully, labels: {res_json.get('labels', 'Unknown')}")

                    return self._process_response(res_json, img.device, label_dtype=label_dtype)

            except Exception as e:
                print(f"Unexpected error (attempt {attempt}/{max_attempts}): {e}")

            if attempt < max_attempts:
                print(f"Retrying in {req_delay} seconds...")
                time.sleep(req_delay)
                req_delay *= self.req_retry_delay_factor

        return self._get_dummy_output(img)

    def _process_response(self, res_json, device, label_dtype=torch.int64):
        """
        Process the server response and extract the mask from shared memory.
        {
            "shm_name": str,
            "shape": list,
            "dtype": str,
            "labels": list
        }
        """
        np_mask = np.ndarray(tuple(res_json["shape"]), dtype=res_json["dtype"], buffer=self.inOutShm.buf)
        # convert the mask to a tensor and permute it back to [B, C, H, W] format
        mask_tensor = torch.from_numpy(np_mask).permute(0, 3, 1, 2).to(device)
        #convert the labels list to a tensor
        label_tensor = torch.tensor(list(res_json["labels"]), dtype=label_dtype).to(device).view(-1, 1)  # reshape to [B, 1]

        return {
            "pred_mask": mask_tensor,
            "pred_label": label_tensor
        }

    def forward(self, image, mask, label, *args, **kwargs):
        """
        input:
            image [B, C, H, W], torch.float32 torch.Size([1, 3, 512, 512])
            mask [B, 1, H, W], torch.float64 [0, 1]  torch.Size([1, 1, 512, 512])
            label [B, 1], torch.int64 [0, 1] torch.Size([1])
        """
        self.count_forward_calls += 1
        # ----------Output interface--------------------------------------
        mask_dtype = torch.empty(0, dtype=mask.dtype).numpy().dtype
        output_dict = self._pred(image, mask_dtype, label.dtype)
        return output_dict

    def close(self):
        if self.inOutShm is not None:
            self.inOutShm.close()
            self.inOutShm.unlink()
            self.inOutShm = None
    
    
if __name__ == "__main__":
    print(MODELS)