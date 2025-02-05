import connexion
import six

from swagger_server.models.input_features import InputFeatures  # noqa: E501
from swagger_server import util

import spineai as sp
#from spineai import *

def get_diagnosis(input_features):  # noqa: E501
	"""get diagnosis based on user input
	# noqa: E501
	:param input_features: A JSON object of input feature
	:type input_features: str

	:rtype: List[InputFeatures]
	"""

	#for test
	input_features = {'features': ['Spondylosis.', 'OLF, T9-11 Broad based central extrusion, T11-12 Mild central stenosis, L1-2 Central downward extrusion, moderate central stenosis, L2-3 Mild central stenosis, L3-4 Broad based central~Lt foraminal extrusion w/ LRS & NFS, severe central stenosis, L4-5 Central~Lt foraminal extrusion w/ LRS & NFS, L5-S1', 'OLF, T9-11. Broad based central extrusion, T11-12. Mild central stenosis, L1-2. Central downward extrusion, moderate central stenosis, L2-3. Mild central stenosis, L3-4. Broad based central~Lt foraminal extrusion w/ LRS & NFS, severe central stenosis, L4-5. Central~Lt foraminal extrusion w/ LRS & NFS, L5-S1',7,-1,-1,-1,-1,8,-1,-1,-1,-1,-1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,0,90,90,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0]}

	#real
	#input_features = input_features

	model_path = '/home/ats9900x/spineai/train/transformer_model.pth'
	result = sp.get_prediction(model_path, input_features)
	pred_result = sp.get_diagnosis(result)

	return pred_result

###for test
#input_features = {'features': ['Spondylosis.', 'OLF, T9-11 Broad based central extrusion, T11-12 Mild central stenosis, L1-2 Central downward extrusion, moderate central stenosis, L2-3 Mild central stenosis, L3-4 Broad based central~Lt foraminal extrusion w/ LRS & NFS, severe central stenosis, L4-5 Central~Lt foraminal extrusion w/ LRS & NFS, L5-S1', 'OLF, T9-11. Broad based central extrusion, T11-12. Mild central stenosis, L1-2. Central downward extrusion, moderate central stenosis, L2-3. Mild central stenosis, L3-4. Broad based central~Lt foraminal extrusion w/ LRS & NFS, severe central stenosis, L4-5. Central~Lt foraminal extrusion w/ LRS & NFS, L5-S1',7,-1,-1,-1,-1,8,-1,-1,-1,-1,-1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,0,90,90,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0]}

#print(get_diagnosis(input_features))

#model_path = '/home/ats9900x/spineai/train/transformer_model.pth'
#result = sp.get_prediction(model_path, input_features)
#pred_result = sp.get_diagnosis(result)

#print(pred_result)
