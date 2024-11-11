import requests
from bs4 import BeautifulSoup
import os
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Read the CSV file and find the 'True PN' sheet
input_csv = 'scraped_data_30000.xlsx'  # Replace with your actual input CSV path
df = pd.read_excel(input_csv, sheet_name='True PN')

# Get the first column (starting from row 2)
ids = df.iloc[1:, 0].values  # Skip the first row (header)

# Base URL for the website
base_url = "http://202.189.117.101:8999"

# URL of the login page
login_url = f"{base_url}/gpne/index.php"

# Login form data (consider using environment variables for sensitive data)
login_payload = {
    'action': 'login',
    'userRemember': 'yes',
    'userName': 'amuratbayev',
    'userPass': 'YNJyzeu@"U$5$@@',
}

# Directory to store downloaded images
image_dir = 'SSS_irb2_True_PN'
os.makedirs(image_dir, exist_ok=True)

# Create a session to persist the login (shared across threads)
session = requests.Session()

# Function to log in once and share the session
def login():
    try:
        login_response = session.post(login_url, data=login_payload)
        login_response.raise_for_status()  # Raises an error for HTTP errors
        print("Login successful!")
    except requests.HTTPError as e:
        print(f"HTTP error occurred during login: {e}")
    except Exception as e:
        print(f"An error occurred during login: {e}")

# Function to scrape images for a specific ID
def scrape_images(id_value):
    # Construct the URL for the specific object ID
    data_url = f"{base_url}/gpne/objectInfoPage.php?id={id_value}"

    try:
        # Now request the data page for the specific ID
        data_response = session.get(data_url)
        data_response.raise_for_status()  # Raises an error for HTTP errors

        # Parse the content of the data page
        soup = BeautifulSoup(data_response.text, 'html.parser')

        # Find and download images
        image_anchors = soup.find_all('a', class_='thumb')
        for i, img_anchor in enumerate(image_anchors):
            anchor_title = img_anchor.get('title')
            img_url = img_anchor.get('href')

            if not anchor_title.startswith("SSS_irb"):
                continue

            if img_url.startswith('http'):
                full_img_url = img_url  # Absolute URL
            else:
                full_img_url = f"{base_url}/{img_url.lstrip('../')}"  # Convert to absolute URL

            print(f"Downloading from: {full_img_url}")

            try:
                # Get the image content
                img_response = session.get(full_img_url)
                img_response.raise_for_status()  # Raises an error for HTTP errors

                # Determine the image format from the URL
                img_ext = full_img_url.split('.')[-1]  # Simple extraction; validate if needed
                img_path = os.path.join(image_dir, f'image_{id_value}_{i}.{img_ext}')

                # Save the image to the specified directory
                with open(img_path, 'wb') as f:
                    f.write(img_response.content)

                print(f"Downloaded: {img_path}")

            except requests.HTTPError as e:
                print(f"Failed to download {full_img_url}: {e}")

            # Optional: sleep to avoid overloading the server
            time.sleep(1)

    except requests.HTTPError as e:
        print(f"Failed to retrieve data for ID {id_value}: {e}")
    except Exception as e:
        print(f"An error occurred while processing ID {id_value}: {e}")

# Main execution
if __name__ == "__main__":
    # Login first
    login()

    # Use ThreadPoolExecutor to download images in parallel with 10 threads
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks for each ID to be processed in parallel
        executor.map(scrape_images, ids)

    print("All tasks completed.")
