import requests
from bs4 import BeautifulSoup
import os
import time

# Create a session to persist the login
session = requests.Session()

# Base URL for the website
base_url = "http://202.189.117.101:8999"

# URL of the login page and the page to scrape
login_url = f"{base_url}/gpne/index.php"
data_url = f"{base_url}/gpne/objectInfoPage.php?id=8442"

# Login form data (consider using environment variables for sensitive data)
login_payload = {
    'action': 'login',
    'userRemember': 'yes',
    'userName': 'amuratbayev',
    'userPass': 'YNJyzeu@"U$5$@@',
}

# Directory to store downloaded images
image_dir = 'downloaded_images'
os.makedirs(image_dir, exist_ok=True)

# Variable to store the desired output
desired_data = []

try:
    # Send a POST request to the login URL
    login_response = session.post(login_url, data=login_payload)

    # Check if login was successful
    login_response.raise_for_status()  # Raises an error for HTTP errors
    print("Login successful!")

    # Now request the data page
    data_response = session.get(data_url)
    data_response.raise_for_status()  # Raises an error for HTTP errors

    # Parse the content of the data page
    soup = BeautifulSoup(data_response.text, 'html.parser')



    # Find and download images (same logic as before)
    images = soup.find_all('img')  # Finds all <img> tags
    for i, img in enumerate(images):
        img_url = img.get('src')

        # Check if the URL is relative or absolute
        if img_url.startswith('http'):
            full_img_url = img_url  # Absolute URL
        else:
            full_img_url = f"{base_url}/{img_url.lstrip('../')}"  # Convert to absolute URL

        try:
            # Get the image content
            img_response = session.get(full_img_url)
            img_response.raise_for_status()  # Raises an error for HTTP errors

            # Determine the image format from the URL
            img_ext = full_img_url.split('.')[-1]  # Simple extraction; validate if needed
            img_path = os.path.join(image_dir, f'image_{i}.{img_ext}')

            # Save the image to the specified directory
            with open(img_path, 'wb') as f:
                f.write(img_response.content)

            print(f"Downloaded: {img_path}")

        except requests.HTTPError as e:
            # Log the error and skip this image
            print(f"Failed to download {full_img_url}: {e}")

        # Optional: sleep to avoid overloading the server
        time.sleep(1)

except requests.HTTPError as e:
    print(f"HTTP error occurred during login or data retrieval: {e}")  # Handle HTTP errors
except Exception as e:
    print(f"An error occurred: {e}")  # Handle other errors
