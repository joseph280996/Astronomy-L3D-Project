import requests
from bs4 import BeautifulSoup
import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Create a session to persist the login
session = requests.Session()

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

# Path to the CSV file
csv_file_path = 'inputs.csv'

# Directory to store output
output_file = 'objects.csv'

def fetch_likely_pn(object_id):
    """ Function to fetch 'Likely PN' for a given object_id """
    data_url = f"{base_url}/gpne/objectInfoPage.php?id={object_id}"

    try:
        # Request the data page for the current ID
        data_response = session.get(data_url)
        data_response.raise_for_status()  # Raises an error for HTTP errors

        # Parse the content of the data page
        soup = BeautifulSoup(data_response.text, 'html.parser')

        # Target Row 1, Column 0 specifically
        target_row = soup.find_all('tr')[1]  # Find the second row (index 1)
        target_column = target_row.find_all('td')[0]  # Find the first column (index 0)

        # Extract the "Likely PN" value
        likely_pn_value = target_column.get_text(strip=True)
        return object_id, likely_pn_value

    except Exception as e:
        print(f"Failed to retrieve data for ID {object_id}: {e}")
        return object_id, None

try:
    # Send a POST request to the login URL
    login_response = session.post(login_url, data=login_payload)
    login_response.raise_for_status()  # Raises an error for HTTP errors
    print("Login successful!")

    # Read object IDs from CSV file
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        object_ids = [row[0] for row in reader]  # Extract IDs from column A

    # Use ThreadPoolExecutor to fetch data concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers based on your needs
        futures = [executor.submit(fetch_likely_pn, object_id) for object_id in object_ids]

        # Prepare to save results
        with open(output_file, 'w') as f_output:
            for future in as_completed(futures):
                object_id, likely_pn_value = future.result()
                if likely_pn_value:
                    print(f"Extracted 'Likely PN' for ID {object_id}: {likely_pn_value}")
                    f_output.write(f"{object_id}, {likely_pn_value}\n")

except requests.HTTPError as e:
    print(f"HTTP error occurred during login or data retrieval: {e}")  # Handle HTTP errors
except Exception as e:
    print(f"An error occurred: {e}")  # Handle other errors