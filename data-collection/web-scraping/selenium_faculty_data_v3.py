import requests
from bs4 import BeautifulSoup
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
from urllib.parse import urljoin, urlparse
import nltk
from nltk.corpus import wordnet
import pandas as pd
from datetime import datetime, timedelta

# Download required NLTK data
nltk.download('wordnet', quiet=True)

# Set test flag
test_flag = True

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace('_', ' '))
    return list(synonyms)

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

def get_r1_university_names():
    wikipedia_url = "https://en.wikipedia.org/wiki/List_of_research_universities_in_the_United_States"
    response = requests.get(wikipedia_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    university_names = []
    for table in soup.find_all('table', {'class': 'wikitable'}):
        for row in table.find_all('tr')[1:]:  # Skip header row
            cells = row.find_all('td')
            if len(cells) > 1:
                university_name = cells[0].text.strip()
                university_names.append(university_name)
    
    return university_names[:10] if test_flag else university_names

def get_university_url(driver, university_name):
    try:
        driver.get(f"https://www.google.com/search?q={university_name}")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "search")))
        
        results = driver.find_elements(By.CSS_SELECTOR, "div.g")
        for result in results:
            link = result.find_element(By.CSS_SELECTOR, "a")
            url = link.get_attribute("href")
            if urlparse(url).netloc.endswith('.edu'):
                return url
    except Exception as e:
        print(f"Error finding URL for {university_name}: {str(e)}")
    return None

def find_directory_url(driver, base_url):
    try:
        driver.get(base_url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        directory_keywords = ['faculty', 'directory', 'staff', 'employees']
        for keyword in directory_keywords:
            links = driver.find_elements(By.PARTIAL_LINK_TEXT, keyword.upper())
            links.extend(driver.find_elements(By.PARTIAL_LINK_TEXT, keyword.capitalize()))
            links.extend(driver.find_elements(By.PARTIAL_LINK_TEXT, keyword.lower()))
            
            if links:
                return links[0].get_attribute('href')
    except Exception as e:
        print(f"Error finding directory for {base_url}: {str(e)}")
    return None

def scrape_faculty_names(driver, directory_url):
    faculty_names = set()
    faculty_titles = ['professor', 'lecturer', 'instructor', 'faculty', 'researcher', 'assoc-prof', 'associate professor']
    title_synonyms = [syn for title in faculty_titles for syn in get_synonyms(title)]
    
    try:
        driver.get(directory_url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        faculty_elements = driver.find_elements(By.XPATH, "//*[contains(@class, 'faculty') or contains(@class, 'staff') or contains(@class, 'employee')]")
        
        for element in faculty_elements:
            name = None
            title = None
            
            name_element = element.find_elements(By.XPATH, ".//*[contains(@class, 'name')]")
            title_element = element.find_elements(By.XPATH, ".//*[contains(@class, 'title') or contains(@class, 'position')]")
            
            if name_element:
                name = name_element[0].text.strip()
            if title_element:
                title = title_element[0].text.strip().lower()
            
            if not name or not title:
                text = element.text.split('\n')
                if len(text) >= 2:
                    name, title = text[0], text[1].lower()
            
            if title and any(syn in title for syn in title_synonyms):
                if name:
                    faculty_names.add(name)
    
    except Exception as e:
        print(f"Error scraping {directory_url}: {str(e)}")
    
    return faculty_names

def compare_faculty_lists(previous_year, current_year):
    return len(current_year - previous_year)

def scrape_r1_faculty_data():
    university_names = get_r1_university_names()
    driver = webdriver.Chrome(options=chrome_options)
    
    results = {}
    all_faculty_data = {}
    
    current_year = datetime.now().year
    years_to_scrape = range(current_year - 4, current_year + 1)  # Scrape last 5 years
    
    for university_name in university_names:
        print(f"Processing {university_name}...")
        university_url = get_university_url(driver, university_name)
        if not university_url:
            print(f"Could not find URL for {university_name}")
            continue
        
        directory_url = find_directory_url(driver, university_url)
        if not directory_url:
            print(f"Could not find directory for {university_name}")
            continue
        
        faculty_data = {}
        for year in years_to_scrape:
            print(f"Scraping data for {year}...")
            faculty_names = scrape_faculty_names(driver, directory_url)
            faculty_data[year] = faculty_names
            
            if year > min(years_to_scrape):
                new_hires = compare_faculty_lists(faculty_data[year-1], faculty_names)
                results.setdefault(university_name, {})[year] = new_hires
            
            all_faculty_data.setdefault(university_name, {})[year] = list(faculty_names)
        
        time.sleep(5)  # Delay between universities
    
    driver.quit()
    
    # Save results to CSV
    with open('new_hires.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['University'] + list(years_to_scrape))
        for university, hires in results.items():
            writer.writerow([university] + [hires.get(year, 'N/A') for year in years_to_scrape])
    
    # Save all faculty names to CSV
    for university, years_data in all_faculty_data.items():
        df = pd.DataFrame(years_data)
        df.to_csv(f'{university.replace(" ", "_")}_faculty.csv', index=False)
    
    print("Scraping completed. Results saved to new_hires.csv and individual university CSV files.")

if __name__ == "__main__":
    scrape_r1_faculty_data()