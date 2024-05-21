import streamlit as st
import asyncio
from playwright.async_api import async_playwright
import os
from datetime import datetime

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

st.sidebar.success("Select a demo above.")

# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

async def generate_pdf(url, pdf_path):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        await page.goto(url)
        
        # Wait for a specific element that indicates the page has loaded
        try:
            await page.wait_for_selector("body", timeout=10000)  # Wait for up to 10 seconds
        except Exception as e:
            st.write("Timeout while waiting for the page to load.")
        
        # Additional delay to ensure all elements are fully loaded
        await asyncio.sleep(5)
        
        # Generate unique filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"document_{timestamp}.pdf"
        pdf_path = os.path.join('data', pdf_filename)
        
        await page.pdf(path=pdf_path, format='A4')
        
        await browser.close()
        
        return pdf_path

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
        pdf_path = asyncio.run(generate_pdf(url, 'document.pdf'))
        st.write(f"PDF generated: {pdf_path}")
    else:
        st.write("Please enter a URL.")
    # Reset the button flag
    st.session_state.generate_pdf_button = False