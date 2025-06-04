import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain,LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool,initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

# Set up the streamlit app

st.set_page_config(page_title="TextToMath problem solver and data search assistant")
st.title("Text to Math Problem solver using Google Gemma2")
groq_api_key=st.sidebar.text_input("Enter your groq api key:",type="password")

if not groq_api_key:
    st.info("Please add your Groq api key to continue")
    st.stop()

llm = ChatGroq(model_name="Gemma2-9b-It",api_key=groq_api_key)

# Initializing the tools

wiki_wrapper = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="wikipedia",
    func=wiki_wrapper.run,
    description=" A tool for searching the internet to find the various info on the topics mentioned"
)
# Initialize the math tool
math_chain= LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="calculator",
    func=math_chain.run,
    description="A tool for answering math related questions.Only input mathematical expressions"
)

prompt = """
You are a agent tasked for solving user's mathematical questions.Logically arrive at the solution and 
provide a detailed explanation and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template= PromptTemplate(input_variables=['question'],template=prompt)

# Combine all the tools into chain
chain = LLMChain(llm=llm,prompt=prompt_template)
reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,
    description="Tool for answering logic based and reasoning questions"
)

# Initialize the agents
assistant_agent = initialize_agent(
    tools=[wiki_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,)

if "messages" not in st.session_state:
    st.session_state['messages']=[
        {'role':'assistant','content':"Hi i am a math chatbot who can answer all your math questions "}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# function to generate response

def gen_res(question):
    res=assistant_agent.invoke({'input':question})

# Lets start the interaction
question = st.text_area("Enter your question:",'I have 2 apples,3 grapes and 6 bananas,and i eat 2 grapes and 1 banana ,and buy an apple,what fruits iam left with?')
if st.button("Find my answer"):
    if question:
        with st.spinner("generate response..."):
            st.session_state.messages.append({'role':'user','content':question})
            st.chat_message("user").write(question)
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])

            st.session_state.messages.append({'role':'assistant','content':response})
            st.write('Response::')
            st.success(response)
    else:
        st.warning("please enter the question")