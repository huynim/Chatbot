import streamlit as st
import asyncio
from pyppeteer import launch

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

st.sidebar.success("Select a demo above.")

async def generate_pdf(url, pdf_path):
    browser = await launch()
    page = await browser.newPage()
    
    await page.goto(url)
    
    await page.pdf({'path': pdf_path, 'format': 'A4'})
    
    await browser.close()

st.text_input("URL", key="url", placeholder="Skriv inn lenken")

if 'generate_pdf_button' not in st.session_state:
    st.session_state.generate_pdf_button = False

def set_button_flag():
    st.session_state.generate_pdf_button = True

st.button("Generate PDF", on_click=set_button_flag)

if st.session_state.generate_pdf_button:
    url = st.session_state.url
    if url:
        st.write("Generating PDF...")
        asyncio.run(generate_pdf(url, 'document.pdf'))
        st.write("PDF generated: document.pdf")
    else:
        st.write("Please enter a URL.")
    # Reset the button flag
    st.session_state.generate_pdf_button = False