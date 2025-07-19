import os
import re
import time
import random
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import mysql.connector
from mysql.connector import Error
from urllib.parse import urlparse

# --- DATABASE CONFIGURATION ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'medi-guide_major'
}

# --- WEB SCRAPING LOGIC ---

def create_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"❌ Error connecting to MySQL: {e}")
        return None

def scrape_from_url(url, session):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = session.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.RequestException:
        return None

def get_medicine_data(medicine_name, session):
    sites_to_try = [
        f"https://www.1mg.com/search/all?name={medicine_name.replace(' ', '%20')}",
        f"https://pharmeasy.in/search/all?name={medicine_name.replace(' ', '%20')}"
    ]

    for url in sites_to_try:
        time.sleep(random.uniform(1.5, 3.5)) # Randomized delay
        soup = scrape_from_url(url, session)
        if not soup:
            continue

        price, count = "0.00", 1
        
        # Logic for 1mg
        if '1mg.com' in url:
            price_el = soup.find('div', class_='style__price-tag___cOxYc')
            count_el = soup.find('div', class_='style__pack-size___3jScl')
            if price_el and (price_match := re.search(r'[\d\.]+', price_el.text)):
                price = price_match.group()
            if count_el and ('tablet' in count_el.text.lower() or 'capsule' in count_el.text.lower()):
                if count_match := re.search(r'(\d+)', count_el.text):
                    count = int(count_match.group(1))

        # Logic for Pharmeasy
        elif 'pharmeasy.in' in url:
            price_el = soup.find('div', class_='PriceInfo_ourPrice__j2_iT')
            count_el = soup.find('div', class_='MedicineOverview_container__zS42d')
            if price_el and (price_match := re.search(r'[\d\.]+', price_el.text)):
                price = price_match.group()
            if count_el and ('tablet' in count_el.text.lower() or 'capsule' in count_el.text.lower()):
                if count_match := re.search(r'(\d+)', count_el.text):
                    count = int(count_match.group(1))

        if float(price) > 0:
            return price, count # Return as soon as we find a valid price

    return "0.00", 1

def run_data_processing_pipeline():
    conn = create_db_connection()
    if not conn: return
    
    cursor = conn.cursor()
    session = requests.Session()

    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `updated_prices` (
          `id` INT NOT NULL, `medicine_name` VARCHAR(255) NOT NULL, `original_price` DECIMAL(10, 2) NOT NULL,
          `count_per_strip` INT, `corrected_price` DECIMAL(10, 2), PRIMARY KEY (`id`)
        ) ENGINE=InnoDB;
        """)
        
        cursor.execute("SELECT p.id, p.medicinename, p.price FROM prices p LEFT JOIN updated_prices up ON p.id = up.id WHERE up.id IS NULL")
        medicines_to_process = cursor.fetchall()
        
        if not medicines_to_process:
            print("🎉 All medicines are already processed!")
            return
            
        print(f"✅ Found {len(medicines_to_process)} medicines to process.")
        
        batch_size = 100
        processed_batch = []

        for med in tqdm(medicines_to_process, desc="Processing Medicines"):
            med_id, med_name, original_price = med
            
            try:
                original_price_float = float(original_price)
            except (ValueError, TypeError):
                continue

            corrected_price, count_per_strip = get_medicine_data(med_name, session)
            
            processed_batch.append((med_id, med_name, original_price_float, count_per_strip, float(corrected_price)))

            if len(processed_batch) >= batch_size:
                insert_query = "INSERT INTO updated_prices (id, medicine_name, original_price, count_per_strip, corrected_price) VALUES (%s, %s, %s, %s, %s)"
                cursor.executemany(insert_query, processed_batch)
                conn.commit()
                processed_batch = []

        if processed_batch:
            insert_query = "INSERT INTO updated_prices (id, medicine_name, original_price, count_per_strip, corrected_price) VALUES (%s, %s, %s, %s, %s)"
            cursor.executemany(insert_query, processed_batch)
            conn.commit()

        print("🎉 Pipeline completed successfully!")

    except Error as e:
        print(f"❌ A database error occurred: {e}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
            print("MySQL connection closed.")

if __name__ == '__main__':
    run_data_processing_pipeline()