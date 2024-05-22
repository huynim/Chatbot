import streamlit as st
import asyncio
from playwright.async_api import async_playwright
import os
from datetime import datetime

st.set_page_config(
    page_title="FSH",
    page_icon="🐟",
)

# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

async def generate_pdf(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        await page.goto(url)
        
        # Wait for a specific element that indicates the page has loaded
        try:
            await page.wait_for_selector("body", timeout=10000)  # Wait for up to 10 seconds
        except Exception as e:
            st.write("Tidsavbrudd mens vi ventet på at siden skulle laste inn.")
        
        # Additional delay to ensure all elements are fully loaded
        await asyncio.sleep(5)
        
        # Generate unique filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"{timestamp}.pdf"
        pdf_path = os.path.join('data', pdf_filename)
        
        await page.pdf(path=pdf_path, format='A4')
        
        await browser.close()
        
        return pdf_path

st.text_input("Lenke til nettsted:", key="url", placeholder="Skriv inn lenken")

if 'generate_pdf_button' not in st.session_state:
    st.session_state.generate_pdf_button = False

def set_button_flag():
    st.session_state.generate_pdf_button = True

st.button("Fortsett", on_click=set_button_flag)

if st.session_state.generate_pdf_button:
    url = st.session_state.url
    if url:
        st.write("Laster ned nettsiden...")
        pdf_path = asyncio.run(generate_pdf(url))
        st.write(f"Nettside lastet ned: {pdf_path}")
    else:
        st.write("Vennligst skriv inn en lenke.")
    # Reset the button flag
    st.session_state.generate_pdf_button = False
