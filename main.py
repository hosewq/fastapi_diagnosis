from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union

import spineai as sp

app = FastAPI()

class InputFeatures(BaseModel):
	features: List[Union[str, int]]

@app.post("/api/1.0.0/get_diagnosis")
def get_diagnosis(data: InputFeatures):
	
#data.input_features = {'features': ['Spondylosis.', 'OLF, T9-11 Broad based central extrusion, T11-12 Mild central stenosis, L1-2 Central downward extrusion, moderate central stenosis, L2-3 Mild central stenosis, L3-4 Broad based central~Lt foraminal extrusion w/ LRS & NFS, severe central stenosis, L4-5 Central~Lt foraminal extrusion w/ LRS & NFS, L5-S1', 'OLF, T9-11. Broad based central extrusion, T11-12. Mild central stenosis, L1-2. Central downward extrusion, moderate central stenosis, L2-3. Mild central stenosis, L3-4. Broad based central~Lt foraminal extrusion w/ LRS & NFS, severe central stenosis, L4-5. Central~Lt foraminal extrusion w/ LRS & NFS, L5-S1',7,-1,-1,-1,-1,8,-1,-1,-1,-1,-1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,0,90,90,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0]}

	"""
	Endpoint to get spinal disease diagnosis based on user input.
	"""
	if not data.features:
		raise HTTPException(status_code=400, detail="Invalid input data")

	model_path = '/home/ats9900x/spineai/train/transformer_model.pth'
	result = sp.get_prediction(model_path, data.features)
	pred_result = sp.get_diagnosis(result)
	print(pred_result)

	print()
	print("pred_printed")
	print()
	# Example processing logic (to be replaced with actual ML model or logic)
	response_data = pred_result
	
	return response_data
