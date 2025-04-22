import os, shlex
import google.generativeai as genai
import chainlit as cl
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import subprocess
from langchain.schema import HumanMessage, AIMessage
os.chdir("photos")
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("The GEMINI_API_KEY environment variable is not set.")
genai.configure(api_key=api_key)


def run_imagemagick_command(command: str) -> str:
    try:
        command_parts = shlex.split(command)
        if not command_parts[0] in ["convert", "exiftool", "ls"]:
            return "Error: Only 'convert', 'exiftool', and 'ls' commands are allowed."
        result = subprocess.run(
            command_parts,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return f"Error (exit code {result.returncode}):\nstdout: {result.stdout}\nstderr: {result.stderr}"
        return f"Command executed successfully.\nOutput:\n{result.stdout}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


@cl.on_chat_start
async def start():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1, google_api_key=api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a very friendly and helpful AI assistant that can also use tools (ImageMagick and exiftool), but DO NOT do anything beyond what user asks. If you wish to use a command, generate ONLY the command, and do so IMMEDIATELY. Do not include any other text. You can run only 1 command at a time, but after receiving the output, you may run more. YOU MAY ONLY USE convert, exiftool, and ls. IMPORTANT - ONLY 1 COMMAND AT A TIME. RUN ALL COMMANDS BEFORE SAYING ANYTHING ELSE. DO NOT DO ANYTHING BEYOND WHAT USER ASKED"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory, verbose=True)
    cl.user_session.set("llm_chain", llm_chain)
    cl.user_session.set("llm", llm)


@cl.on_message
async def main(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")
    llm = cl.user_session.get("llm")
    memory = llm_chain.memory

    if llm_chain is None or llm is None:
        await cl.Message("Error: LLM Chain or LLM not initialized. Start a new chat.").send()
        return
    
    initial_response = await llm_chain.acall({"input": message.content}, callbacks=[cl.AsyncLangchainCallbackHandler()])
    while True:
        initial_response_text = initial_response["text"]
        if initial_response_text.strip().startswith("convert") or initial_response_text.strip().startswith("exiftool") or initial_response_text.strip().startswith("ls"):
            imagemagick_output = run_imagemagick_command(initial_response_text)
            prompt2 = ChatPromptTemplate.from_messages([
                ("system", "You are a very friendly and helpful AI assistant that can also use tools (ImageMagick and exiftool), but DO NOT do anything beyond what user asks. If you wish to use a command, generate ONLY the command, and do so IMMEDIATELY. Do not include any other text. YOU MAY ONLY USE convert, exiftool, and ls. IMPORTANT - ONLY 1 COMMAND AT A TIME. RUN ALL COMMANDS BEFORE SAYING ANYTHING ELSE. DO NOT DO ANYTHING BEYOND WHAT USER ASKED"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ])
            imagemagick_output = "Here is the output of the command: \n" + imagemagick_output
            memory.chat_memory.add_user_message(imagemagick_output)
            memory.chat_memory.add_ai_message(initial_response_text)
            final_chain = LLMChain(prompt=prompt2, llm=llm, memory=memory, verbose = True)
            final_response = await final_chain.acall({"input": ""}, callbacks=[cl.AsyncLangchainCallbackHandler()])
            initial_response = final_response

        else:
            await cl.Message(content=initial_response_text).send()
            break
