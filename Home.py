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
asyncio.get_event_loop().run_until_complete(generate_pdf(st.session_state.url, 'document.pdf'))