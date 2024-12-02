from urllib.parse import urljoin

def find_directory_url(driver, base_url):
    try:
        driver.get(base_url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        patterns = [
            r"faculty.*directory",
            r"employee.*directory",
            r"campus.*directory",
            r"staff.*directory",
            r"directory"
        ]
        
        for pattern in patterns:
            links = driver.find_elements(By.XPATH, f"//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{pattern}')]")
            if links:
                relative_url = links[0].get_attribute('href')
                return urljoin(base_url, relative_url)
        
        logging.warning(f"No directory link found for {base_url}")
        return None
    except TimeoutException:
        logging.error(f"Timeout occurred while searching for directory on {base_url}")
        return None
    except Exception as e:
        logging.error(f"Error finding directory URL for {base_url}: {str(e)}", exc_info=True)
        return None