import requests
from bs4 import BeautifulSoup
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
from urllib.parse import urljoin
import nltk
from nltk.corpus import wordnet

# Download required NLTK data
nltk.download('wordnet')

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace('_', ' '))
    return list(synonyms)

# Prevent chrome from opening each time the driver calls it
chrome_options = webdriver.ChromeOptions()
# Comment out the next 3 lines to visualize the scraping sites
# chrome_options.add_argument('--headless')
# chrome_options.add_argument('--no-sandbox')
# chrome_options.add_argument('--disable-dev-shm-usage')

def directory_url_generator():
    wikipedia_url = "https://en.wikipedia.org/wiki/List_of_research_universities_in_the_United_States"
    response = requests.get(wikipedia_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    university_urls = []
    for table in soup.find_all('table', {'class': 'wikitable'}):
        for row in table.find_all('tr')[1:]:  # Skip header row
            cells = row.find_all('td')
            if len(cells) > 1:
                university_name = cells[0].text.strip()
                # a represents the "anchor" tag, and this finds
                university_url = cells[0].find('a')['href'] if cells[0].find('a') else None
                if university_url:
                    university_urls.append(f"https://en.wikipedia.org{university_url}")
    
    directory_urls = []
    driver = webdriver.Chrome(options=chrome_options)  # Ensure you have chromedriver in PATH
    
    for url in university_urls:
        try:
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            # Look for common directory link patterns
            directory_keywords = ['faculty', 'directory', 'staff', 'employees']
            for keyword in directory_keywords:
                links = driver.find_elements(By.PARTIAL_LINK_TEXT, keyword.upper())
                links.extend(driver.find_elements(By.PARTIAL_LINK_TEXT, keyword.capitalize()))
                links.extend(driver.find_elements(By.PARTIAL_LINK_TEXT, keyword.lower()))
                
                if links:
                    directory_url = links[0].get_attribute('href')
                    directory_urls.append(directory_url)
                    print(f"Found directory for {url}: {directory_url}")
                    break
            else:
                print(f"Could not find directory for {url}")
        
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
    
    driver.quit()
    return directory_urls

def scrape_faculty_names(driver, directory_url):
    faculty_names = set()
    faculty_titles = ['professor', 'lecturer', 'instructor', 'faculty', 'researcher', 'assoc-prof', 'associate professor']
    title_synonyms = [syn for title in faculty_titles for syn in get_synonyms(title)]
    
    try:
        driver.get(directory_url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        # Look for common faculty listing patterns
        faculty_elements = driver.find_elements(By.XPATH, "//*[contains(@class, 'faculty') or contains(@class, 'staff')]")
        
        for element in faculty_elements:
            name = None
            title = None
            
            # Look for name and title in various formats
            name_element = element.find_elements(By.XPATH, ".//*[contains(@class, 'name')]")
            title_element = element.find_elements(By.XPATH, ".//*[contains(@class, 'title') or contains(@class, 'position')]")
            
            if name_element:
                name = name_element[0].text.strip()
            if title_element:
                title = title_element[0].text.strip().lower()
            
            # If we couldn't find structured name/title, try to parse the text
            if not name or not title:
                text = element.text.split('\n')
                if len(text) >= 2:
                    name, title = text[0], text[1].lower()
            
            # Check if the title matches our criteria
            if title and any(syn in title for syn in title_synonyms):
                if name:
                    faculty_names.add(name)
    
    except TimeoutException:
        print(f"Timeout occurred when scraping {directory_url}")
    except Exception as e:
        print(f"An error occurred when scraping {directory_url}: {str(e)}")
    
    return faculty_names

# The rest of the code (compare_faculty_lists and scrape_r1_faculty_data) remains largely the same

def scrape_r1_faculty_data():
    directory_urls = directory_url_generator()
    driver = webdriver.Chrome()
    
    results = {}
    previous_year_data = {}
    
    for year in range(2023, 2025):  # Adjust year range as needed
        current_year_data = {}
        
        for directory_url in directory_urls:
            faculty_names = scrape_faculty_names(driver, directory_url)
            current_year_data[directory_url] = faculty_names
            
            if year > 2023:  # Skip comparison for the first year
                new_hires = compare_faculty_lists(previous_year_data[directory_url], faculty_names)
                results.setdefault(directory_url, {})[year] = new_hires
        
        previous_year_data = current_year_data
        
        # Save results to CSV
        with open(f'faculty_hires_{year}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['University', 'New Hires'])
            for university, hires in results.items():
                writer.writerow([university, hires.get(year, 'N/A')])
        
        time.sleep(20)  # Wait between years to avoid overloading servers
    
    driver.quit()

if __name__ == "__main__":
    scrape_r1_faculty_data()