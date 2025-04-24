import socketserver as ss
import json

from torch import from_numpy, Tensor

from numpy import array, ndarray, uint8

class RequestHandler(ss.StreamRequestHandler): 

    model = None

    def handle(self):
        """ Handles request to the server. Uses stream like (file-like objects) communication. Reads rfile with client's data and writes result to wfile. 
        
        Accepts json string like: {"image": [0,0,0,0....]}

        Returns json string like: {"class_id": 0, "class_name": "cls_1"}
        """

        data: str = self.rfile.readline(1000).rstrip()
        image_array: ndarray = array(json.loads(data)["image"], dtype=uint8)
        
        image_tensor: Tensor = from_numpy(image_array)

        # Call model for inference
        model_responce: dict[str, int|str] = {"class_id": 0, "class_name": "dog"} #self.model.predict(image_tensor)

        responce: bytes = json.dumps(model_responce).encode() + b'\n'
        self.wfile.write(responce)