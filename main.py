from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
from fastapi.encoders import jsonable_encoder

import json
import spineai as sp
import assistant_api as assistant

app = FastAPI()

class InputFeatures(BaseModel):
	features: List[Union[str, int]]

@app.post("/api/1.0.0/get_diagnosis")
async def get_diagnosis(data: InputFeatures):
	
#data.input_features = {'features': ['Spondylosis.', 'OLF, T9-11 Broad based central extrusion, T11-12 Mild central stenosis, L1-2 Central downward extrusion, moderate central stenosis, L2-3 Mild central stenosis, L3-4 Broad based central~Lt foraminal extrusion w/ LRS & NFS, severe central stenosis, L4-5 Central~Lt foraminal extrusion w/ LRS & NFS, L5-S1', 'OLF, T9-11. Broad based central extrusion, T11-12. Mild central stenosis, L1-2. Central downward extrusion, moderate central stenosis, L2-3. Mild central stenosis, L3-4. Broad based central~Lt foraminal extrusion w/ LRS & NFS, severe central stenosis, L4-5. Central~Lt foraminal extrusion w/ LRS & NFS, L5-S1',7,-1,-1,-1,-1,8,-1,-1,-1,-1,-1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,0,90,90,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0]}

	"""
	Endpoint to get spinal disease diagnosis based on user input.
	"""
	if not data.features:
		raise HTTPException(status_code=400, detail="Invalid input data")

	model_path = '/home/ats9900x/spineai/train/transformer_model.pth'
	result = sp.get_prediction(model_path, data.features)
	pred_result = sp.get_diagnosis(result)
	
	converted_data = {key: float(value) for key, value in pred_result.items()}
	response_data = jsonable_encoder(converted_data)
	
	return response_data



class LLMRequest(BaseModel):
	'''
	json_input examples:
	json_input = {'Central stenosis': 0.40886205, 'HLD': 0.3023262, 'Foraminal stenosis': 0.26957512}
	'''
	json_input: str
	use_thread_id: bool = False

@app.post("/api/1.0.0/get_llm_response"):
async def get_llm_response(req: LLMRequest):
	json_input = json.dumps(req.json_input)

	diagnosis, probability = assistant.process_json(json_input)
	
	assistant_chain = assistant.AssistantChatChain()

	question = diagnosis[0]+"에 대해 한국어로 알려줘."
	response = assistant_chain.invoke(question)

	print("\nAI Response:", response)

	return response

print()
