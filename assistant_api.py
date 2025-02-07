import time
import openai
import json
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import Runnable
from langchain.callbacks.base import BaseCallbackHandler
from config import cfg

OPENAI_API_KEY = cfg["openai_api_key"]["default"]
ASSISTANT_ID = cfg["openai_assistant_id"]["default"]

client = openai.OpenAI(api_key=OPENAI_API_KEY)

class OpenAIAssistant(Runnable):
	"""
	OpenAI Assistant API를 LangChain의 Runnable로 래핑하는 클래스
	"""

	def __init__(self):
		self.thread_id = None

	def invoke(self, input_text: str):
		if self.thread_id is None:
			self.thread_id = self.create_thread()

		self.send_message(self.thread_id, input_text)
		response = self.run_assistant(self.thread_id)

		return response

	def create_thread(self):
		response = client.beta.threads.create()

		return response.id

	def send_message(self, thread_id, message):
		client.beta.threads.messages.create(thread_id=thread_id,role="user",content=message)

	def run_assistant(self, thread_id):
		run = client.beta.threads.runs.create(thread_id=thread_id,assistant_id=ASSISTANT_ID)

		while True:
			run_status = client.beta.threads.runs.retrieve(thread_id=thread_id,run_id=run.id)
			if run_status.status == "completed":
				break
			time.sleep(1)
		
		messages = client.beta.threads.messages.list(thread_id=thread_id)

		return messages.data[0].content[0].text.value

class AssistantChatChain(Runnable):
	def __init__(self):
		self.assistant = OpenAIAssistant()

	def invoke(self, input_text: str):
		print(f"사용자: {input_text}")
		response = self.assistant.invoke(input_text)
		print(f"AI 응답: {response}")

		return response

def process_json(json_input):
	diagnosis = []
	probability = []

	for key, value in json.loads(json_input).items():
		diagnosis.append(key)
		probability.append(value)

	return diagnosis, probability

'''
if __name__ == "__main__":
	assistant_chain = AssistantChatChain()

	question = "Central stenosis, DDD(degenerative disc disesases), disc herniation에 대해 알려줘."
	response = assistant_chain.invoke(question)
	print("\nFinal AI Response:", response)

	json_input = {'Central stenosis': 0.40886205, 'HLD': 0.3023262, 'Foraminal stenosis': 0.26957512}
	json_input = json.dumps(json_input)

	diagnosis, probability = process_json(json_input)

	# question for diagnosis1
	question = diagnosis[0]+"에 대해 알려줘."
	response = assistant_chain.invoke(question)

	print("\nFinal AI Response:", response)
'''


