import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader


# streamslit app

st.set_page_config(page_title="Langchain:Summarize text from YT or Website")
st.title("Langchain:Summarize text from YT or Website")
st.subheader('Summarize URL')

#  GROQ key and URL to be summarized
with st.sidebar:
    groq_api_key = st.text_input("GROQ_API_KEY",value="",type="password")


llm = ChatGroq(model="Gemma2-9b-It",api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words.
Content:{text}
"""
prompt = PromptTemplate(input_variables=['text'],template=prompt_template)
generic_url = st.text_input("URL",label_visibility="collapsed")
if st.button("Summarize the content from YT or Website"):
    #  Validate all inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid url.It can be a YT video URL or website URL")
    else:
        try:
            with st.spinner("Waiting..."):
                # Loading the YT/website data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,headers={})
                docs=loader.load()
                #  cahin for summarization
                chain = load_summarize_chain(llm,chain_type='stuff',prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception:{e}")