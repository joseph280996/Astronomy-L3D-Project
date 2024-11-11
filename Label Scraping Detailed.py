import requests
from bs4 import BeautifulSoup
import csv
import time
import concurrent.futures
from queue import Queue
from requests.exceptions import RequestException

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

# Path to the CSV output file
output_csv = 'scraped_data_30000.csv'

# Number of IDs to scrape
total_ids = 30000

# Max workers for ThreadPool (adjust based on your network and server capacity)
max_workers = 10

# Rate limiting (set to 1 request every 0.5 seconds to avoid overwhelming the server)
rate_limit_time = 0.5

# Create a queue to store rate-limited requests
request_queue = Queue()

def scrape_data(object_id):
    """Scrape data for a given ID and return the results."""
    data_url = f"{base_url}/gpne/objectInfoPage.php?id={object_id}"

    try:
        # Now request the data page
        data_response = session.get(data_url)
        data_response.raise_for_status()  # Raises an error for HTTP errors

        # Parse the content of the data page
        soup = BeautifulSoup(data_response.text, 'html.parser')

        # Extract the table rows
        rows = soup.find_all('tr')

        # Prepare variables for data storage
        label = ""  # For Row 1, Column 0 (Label)
        coordinates = ("", "")  # For Row 7, Column 0 (DRAJ2000 and DDECJ2000)
        name = ""  # For h2 element (Name)

        # Iterate over rows and columns to find the required data
        for row_idx, row in enumerate(rows):
            columns = row.find_all('td')

            # Row 1, Column 0: Label
            if row_idx == 1 and columns:
                label = columns[0].get_text(strip=True)

            # Row 7, Column 0: Coordinates
            if row_idx == 7 and columns:
                coord_text = columns[0].get_text(strip=True)
                coordinates = tuple(coord_text.split())  # Split by space into two values (DRAJ2000 and DDECJ2000)

        # Extract the Name from the h2 element
        h2_element = soup.select('aside#sidebar_object div#infodiv div#headertable_info h2')
        if h2_element:
            name = h2_element[0].get_text(strip=True)
        else:
            name = "N/A"  # If no name is found

        return [object_id, name, coordinates[0], coordinates[1], label]

    except (RequestException, IndexError) as e:
        # Handle HTTP and parsing errors
        print(f"Error processing ID {object_id}: {e}")
        return [object_id, "N/A", "N/A", "N/A", "N/A"]

def rate_limited_scrape(object_id):
    """Rate limited wrapper for scrape_data function."""
    time.sleep(rate_limit_time)  # Rate limiting to avoid overwhelming the server
    return scrape_data(object_id)

def main():
    try:
        # Send a POST request to the login URL
        login_response = session.post(login_url, data=login_payload)

        # Check if login was successful
        login_response.raise_for_status()  # Raises an error for HTTP errors
        print("Login successful!")

        # Open the CSV file for writing
        with open(output_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write headers
            writer.writerow(['ID', 'Name', 'DRAJ2000', 'DDECJ2000', 'Label'])

            # Use ThreadPoolExecutor to scrape data concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks for all IDs (1 to 30,000)
                future_to_id = {executor.submit(rate_limited_scrape, object_id): object_id for object_id in range(1, total_ids + 1)}

                for future in concurrent.futures.as_completed(future_to_id):
                    object_id = future_to_id[future]
                    try:
                        data = future.result()
                        writer.writerow(data)  # Write the result to the CSV
                        print(f"ID {object_id} processed successfully.")
                    except Exception as e:
                        print(f"Error processing ID {object_id}: {e}")

        print(f"Data for {total_ids} IDs saved to {output_csv}")

    except requests.HTTPError as e:
        print(f"HTTP error occurred during login: {e}")  # Handle HTTP errors
    except Exception as e:
        print(f"An error occurred: {e}")  # Handle other errors

if __name__ == '__main__':
    main()
