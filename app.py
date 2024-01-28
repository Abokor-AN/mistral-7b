from langchain_community.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = CTransformers(
    model='mistralai\Mistral-7B-Instruct-v0.2-GGUF\mistral-7b-instruct-v0.2.Q6_K.gguf', callbacks=[StreamingStdOutCallbackHandler()]
)

template = """Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

response = llm_chain.run("What is AI?")