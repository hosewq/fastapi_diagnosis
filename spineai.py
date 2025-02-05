import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import re

from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MultiLabelBinarizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerModel(nn.Module):
	def __init__(self, input_dim, num_heads, hidden_dim, output_dim, num_layers, dropout=0.1):
		super(TransformerModel, self).__init__()
		self.embedding = nn.Linear(input_dim, hidden_dim)

		encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		#encoder_layer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.fc = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		x = self.embedding(x)
		x = x.unsqueeze(1)
		x = self.transformer_encoder(x)
		x = x.mean(dim=1)
		x = self.fc(x)

		return x


class TextEmbeddings:
	def __init__(self, xray, ct, mri):
		self.xray_txt = xray
		self.ct_txt = ct
		self.mri_txt = mri

	def clean_text(text):
		text = re.sub(r'(w/)(\S)', r'\1 \2', text)
		text = re.sub(r"\s*\.\s*", ". ", text)
		text = re.sub(r"\s+", " ", text).strip()

		return text

	@staticmethod
	def load_model_and_tokenizer(model_name):
		tokenizer = AutoTokenizer.from_pretrained(model_name)
		model = AutoModel.from_pretrained(model_name)

		model.to(device)

		return tokenizer, model

	def embed_text(self, texts, tokenizer, model):
		inputs = tokenizer(texts, return_tensors="pt", padding=True, max_length=512, truncation=True).to(device)

		with torch.no_grad():
			outputs = model(**inputs)
			embeddings = outputs.last_hidden_state.mean(dim=1)

		return embeddings

	def process(self, category, readings):
		tokenizer, model = self.model_tokenizer_dict[category]
		embeddings = self.embed_text(readings, tokenizer, model)

		return embeddings

	def tensor_to_dataframe(self, tensor, column_prefix):
		tensor_cpu = tensor.cpu().numpy()
		df = pd.DataFrame(tensor_cpu, columns=[f"{column_prefix}_dim{i}" for i in range(tensor_cpu.shape[1])])
		
		return df

	def text_embeddings(self):
		models = {
			'xray' : '/home/ats9900x/bert/bert_test/clibert-finetuned-xray',
			'ct' : '/home/ats9900x/bert/bert_test/clibert-finetuned-ctscan',
			'mri' : '/home/ats9900x/bert/bert_test/clibert-finetuned-mri'
		}
		
		self.model_tokenizer_dict = {key: self.load_model_and_tokenizer(model_path) for key, model_path in models.items()}

		xray_embed = self.process('xray', self.xray_txt)
		ct_embed = self.process('ct', self.ct_txt)
		mri_embed = self.process('mri', self.mri_txt)

		xray_df = self.tensor_to_dataframe(xray_embed, 'xray')
		ct_df = self.tensor_to_dataframe(ct_embed, 'ct')
		mri_df = self.tensor_to_dataframe(mri_embed, 'mri')

		embed_df = pd.concat([xray_df, ct_df, mri_df], axis=1)

		return embed_df
		

# get_diagnosis_prediction():
# Input: model_path, input_features(json)
# Output: prediction result (diagnosis, probability pairs)
def get_prediction(model_path, input_features):
	#process input
	#input_data = input_features["features"] # 1 dim input list
	input_data = input_features

	#for testing - readings are not included.
	#input_data = ['Spondylosis.', 'OLF, T9-11 Broad based central extrusion, T11-12 Mild central stenosis, L1-2 Central downward extrusion, moderate central stenosis, L2-3 Mild central stenosis, L3-4 Broad based central~Lt foraminal extrusion w/ LRS & NFS, severe central stenosis, L4-5 Central~Lt foraminal extrusion w/ LRS & NFS, L5-S1', 'OLF, T9-11. Broad based central extrusion, T11-12. Mild central stenosis, L1-2. Central downward extrusion, moderate central stenosis, L2-3. Mild central stenosis, L3-4. Broad based central~Lt foraminal extrusion w/ LRS & NFS, severe central stenosis, L4-5. Central~Lt foraminal extrusion w/ LRS & NFS, L5-S1', 7,-1,-1,-1,-1,8,-1,-1,-1,-1,-1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,0,90,90,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0]

	te = TextEmbeddings(input_data[0], input_data[1], input_data[2])
	embed_df = te.text_embeddings()

	print(embed_df.head())

	tmp = embed_df.values.tolist()[0]
	tmp2 = input_data[3:]

	#hyperparameters
	#input_dim = len(input_data)
	tmp_result = tmp + tmp2
	input_dim = len(tmp_result)
	hidden_dim = 128
	num_heads = 8
	output_dim = 11
	num_layers = 3

	model = TransformerModel(input_dim=input_dim, num_heads=num_heads, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
	model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
	model = model.to(device)
	model.eval()

	#input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device).unsqueeze(0)
	input_tensor = torch.tensor(tmp_result, dtype=torch.float32).to(device).unsqueeze(0)

	with torch.no_grad():
		output = model(input_tensor)
		probabilities = torch.sigmoid(output).cpu().numpy()

		# prediction by probability threshold (0.5)
		#predicted_classes = [i for i, prob in enumerate(probabilities[0]) if prob > 0.5] #threshold = 0.5
		#return predicted_classes, probabilities[0]

		# predictions - Select top 3 predictions
		top_3_indices = probabilities[0].argsort()[-3:][::-1]
		predicted_classes = top_3_indices.tolist()

		#predicted_class_names = [list(class_indices.keys())[list(class_indices.values()).index(i)] for i in top_3_indices]

	return predicted_classes, probabilities[0][top_3_indices]


##

def get_class_name():
	# Load dataset
	data_path = '/home/ats9900x/spineai/data/preprocessed_dataset.csv'
	data = pd.read_csv(data_path, header=0)

	#data.drop(data.filter(regex='Xray').columns, axis=1, inplace=True)
	#data.drop(data.filter(regex='CT').columns, axis=1, inplace=True)
	#data.drop(data.filter(regex='MRI').columns, axis=1, inplace=True)

	data = data[data['D1_name'] != 'N']
	data['labels'] = data[['D1_name', 'D2_name', 'D3_name']].apply(lambda x: set(x.dropna()), axis=1)
	data['labels'] = data['labels'].apply(lambda labels: labels - {'N'})

	X = data.drop(columns=['D1_name', 'D2_name', 'D3_name', 'labels'])
	labels = data['labels']

	# One-hot encode the labels using MultiLabelBinarizer
	mlb = MultiLabelBinarizer()
	y = mlb.fit_transform(labels)

	return mlb.classes_.tolist()

def get_diagnosis(predictions):
	pred_result = {}
	diagnosis_dict = {"classes":predictions[0], "prob":predictions[1]}

	print(diagnosis_dict)

	classes = get_class_name()
	print(classes)
	print(type(classes))
	print(classes[0])

	for i in range(len(diagnosis_dict["classes"])):
		tmp = classes[diagnosis_dict["classes"][i]]
		diagnosis_dict["classes"][i] = tmp
		pred_result[diagnosis_dict["classes"][i]] = predictions[1][i]

	return pred_result

##
#input_features = {'features': [0,0,0,0,0,0,0,0,0,0,0,0]}
#model_path = '/home/ats9900x/spineai/train/transformer_model.pth'

#result = get_prediction(model_path, input_features)
#pred_result = get_diagnosis(result)
#print(pred_result)

