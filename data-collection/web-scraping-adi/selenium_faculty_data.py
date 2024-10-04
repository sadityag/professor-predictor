import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
from urllib.parse import urljoin

# Step 1: Get list of R1 universities
def get_r1_universities():
    # This is a placeholder. In reality, you'd need to scrape or import this list.
    return [
        "https://www.harvard.edu",
        "https://www.stanford.edu",
        "https://www.mit.edu",
        # Add more R1 university URLs here
    ]

# Step 2: Scrape faculty names from each university
def scrape_faculty_names(driver, university_url):
    faculty_names = set()
    
    try:
        # Navigate to the university's faculty directory or employee listing
        driver.get(urljoin(university_url, "/faculty-directory"))  # Adjust this path as needed
        
        # Wait for the faculty list to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "faculty-list"))
        )
        
        # Find all faculty name elements
        faculty_elements = driver.find_elements(By.CLASS_NAME, "faculty-name")
        
        # Extract names
        for element in faculty_elements:
            faculty_names.add(element.text.strip())
        
    except TimeoutException:
        print(f"Timeout occurred when scraping {university_url}")
    except Exception as e:
        print(f"An error occurred when scraping {university_url}: {str(e)}")
    
    return faculty_names

# Step 3: Compare faculty lists year over year
def compare_faculty_lists(previous_year, current_year):
    new_hires = current_year - previous_year
    return len(new_hires)

# Main scraping function
def scrape_r1_faculty_data():
    universities = get_r1_universities()
    driver = webdriver.Chrome()  # Make sure you have chromedriver installed and in PATH
    
    results = {}
    previous_year_data = {}
    
    for year in range(2023, 2025):  # Adjust year range as needed
        current_year_data = {}
        
        for university_url in universities:
            faculty_names = scrape_faculty_names(driver, university_url)
            current_year_data[university_url] = faculty_names
            
            if year > 2023:  # Skip comparison for the first year
                new_hires = compare_faculty_lists(previous_year_data[university_url], faculty_names)
                results.setdefault(university_url, {})[year] = new_hires
        
        previous_year_data = current_year_data
        
        # Save results to CSV
        with open(f'faculty_hires_{year}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['University', 'New Hires'])
            for university, hires in results.items():
                writer.writerow([university, hires.get(year, 'N/A')])
        
        time.sleep(60)  # Wait between years to avoid overloading servers
    
    driver.quit()

if __name__ == "__main__":
    scrape_r1_faculty_data()