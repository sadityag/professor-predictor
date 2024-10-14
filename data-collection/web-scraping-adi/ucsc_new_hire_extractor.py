import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time

def get_soup(url):
    response = requests.get(url)
    response.raise_for_status()
    return BeautifulSoup(response.content, 'html.parser')

def explore_disciplines(url):
    driver = webdriver.Chrome()
    driver.get(url)
    
    try:
        # Wait for the "Explore the disciplines" section to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "explorethedisciplines"))
        )
        
        # Find all clickable elements in the "Explore the disciplines" section
        discipline_elements = driver.find_elements(By.CSS_SELECTOR, "#explorethedisciplines ~ div a")
        
        discipline_links = []
        for element in discipline_elements:
            discipline_links.append((element.text, element.get_attribute('href')))
        
        print(f"Found {len(discipline_links)} discipline links:")
        for name, link in discipline_links:
            print(f"- {name}: {link}")
            
        # Explore each discipline link
        for name, link in discipline_links:
            print(f"\nExploring {name}...")
            driver.get(link)
            time.sleep(2)  # Wait for page to load
            
            # Look for faculty or department links
            faculty_links = driver.find_elements(By.XPATH, "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'faculty') or contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'department')]")
            
            if faculty_links:
                print(f"Found potential faculty/department links in {name}:")
                for faculty_link in faculty_links[:5]:  # Print first 5 links
                    print(f"- {faculty_link.text}: {faculty_link.get_attribute('href')}")
            else:
                print(f"No obvious faculty/department links found in {name}")
    
    except TimeoutException:
        print("Timed out waiting for page to load")
    finally:
        driver.quit()

def main():
    url = "https://www.ucsc.edu/academics/"
    explore_disciplines(url)

if __name__ == "__main__":
    main()