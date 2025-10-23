import os
import requests
import time
import logging
from tqdm import tqdm
from pathlib import Path
import sys
import concurrent.futures
import json
import re
from urllib.parse import urlparse

# ==============================================================================
# بخش تنظیمات
# ==============================================================================

# نام فایل JSON ورودی
JSON_FILE_NAME = "products_master.json"

# نام پوشه‌ای که عکس‌ها در آن ذخیره خواهند شد
OUTPUT_FOLDER_NAME = "product_images_downloaded" # نام جدید که با پوشه قبلی قاطی نشود

# --- تنظیمات پردازش موازی و تلاش مجدد ---
MAX_WORKERS = 10         # تعداد دانلودهای همزمان
MAX_RETRIES = 3          # تعداد تلاش مجدد برای هر عکس
RETRY_DELAY = 1          # ثانیه تاخیر اولیه

# --- تنظیمات لاگ (گزارش خطا) ---
LOG_FILE_NAME = "download_errors.log"

# ==============================================================================
# بخش راه‌اندازی لاگ (گزارش خطا)
# ==============================================================================

logger = logging.getLogger('ImageDownloader')
logger.setLevel(logging.ERROR)
handler = logging.FileHandler(LOG_FILE_NAME, mode='w', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# ==============================================================================
# هسته اصلی اسکریپت
# ==============================================================================

def find_project_root(start_path, marker_file):
    """پوشه اصلی پروژه را که حاوی فایل JSON است، پیدا می‌کند."""
    current_path = Path(start_path).resolve()
    if (current_path / marker_file).is_file():
        return current_path
    while not (current_path / marker_file).is_file():
        if current_path.parent == current_path: return None
        current_path = current_path.parent
    return current_path

def create_safe_filename(name, brand, image_url, index):
    """
    یک نام فایل امن و خوانا بر اساس نام، برند و لینک عکس ایجاد می‌کند.
    """
    try:
        path = urlparse(image_url).path
        ext = os.path.splitext(path)[1]
        if not ext or len(ext) > 5: # اگر پسوند عجیب یا بلند بود
            ext = '.jpg'
    except Exception:
        ext = '.jpg'

    base_name = f"{brand} - {name}"
    safe_name = re.sub(r'[\\/*?:"<>|]', "", base_name)
    
    max_len = 150
    if len(safe_name) > max_len:
        safe_name = safe_name[:max_len]

    final_filename = f"{safe_name.strip()} - {index}{ext}"
    return final_filename

def download_single_image(session, image_url, output_path):
    """
    یک عکس را با تلاش مجدد دانلود می‌کند.
    خروجی: 'SUCCESS', 'SKIPPED', 'FAILED'
    """
    if output_path.exists():
        return "SKIPPED"

    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(image_url, stream=True, timeout=15)
            response.raise_for_status() # بررسی خطاهای HTTP (مثل 404, 500)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return "SUCCESS"
        
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt)) # Exponential backoff
            else:
                logger.error(f"DOWNLOAD_FAILED for {image_url}: {e}")
                return "FAILED"
    return "FAILED" # اگر حلقه تمام شد

def batch_download_images():
    """تابع اصلی برای پردازش دسته‌ای عکس‌ها با ThreadPool."""
    
    script_location = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(script_location, JSON_FILE_NAME)
    
    if project_root is None:
        print(f"خطا: فایل '{JSON_FILE_NAME}' در مسیر '{script_location}' یا بالاتر از آن پیدا نشد.")
        return

    json_file_path = project_root / JSON_FILE_NAME
    output_dir = project_root / OUTPUT_FOLDER_NAME
    output_dir.mkdir(exist_ok=True)
    
    print(f"پوشه اصلی پروژه: '{project_root}'")
    print(f"فایل JSON ورودی: '{json_file_path}'")
    print(f"پوشه خروجی: '{output_dir}'")
    print(f"فایل لاگ خطاها: '{LOG_FILE_NAME}'")

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            products = json.load(f)
    except Exception as e:
        print(f"خطا در خواندن فایل JSON: {e}")
        return

    # ایجاد یک لیست کلی از تمام وظایف دانلود
    tasks = []
    for product in products:
        name = product.get('name')
        brand = product.get('brand')
        images = product.get('images', [])
        
        if not name or not brand: continue

        for i, image_url in enumerate(images):
            if not image_url or not image_url.startswith('http'):
                continue
                
            filename = create_safe_filename(name, brand, image_url, i + 1)
            output_path = output_dir / filename
            tasks.append((image_url, output_path))

    if not tasks:
        print("هیچ لینکی برای دانلود در فایل JSON پیدا نشد.")
        return

    print(f"تعداد {len(tasks)} عکس برای دانلود پیدا شد...")
    
    start_time_total = time.time()
    
    # --- استفاده از ThreadPoolExecutor برای پردازش موازی ---
    # از یک Session برای مدیریت بهینه کانکشن‌ها استفاده می‌کنیم
    with requests.Session() as session:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # ایجاد دیکشنری برای نگاشت Future ها به مسیر فایل
            futures = {
                executor.submit(download_single_image, session, url, path): path.name
                for url, path in tasks
            }
            
            # جمع‌آوری نتایج با نوار پیشرفت
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Downloading Images", unit="image"):
                pass # نوار پیشرفت به صورت خودکار جلو می‌رود

    end_time_total = time.time()
    print(f"\nعملیات دانلود با موفقیت پایان یافت.")
    print(f"کل زمان: {end_time_total - start_time_total:.2f} ثانیه.")
    print(f"خروجی‌ها در پوشه '{output_dir}' ذخیره شدند.")
    print(f"خطاهای احتمالی دانلود در فایل '{LOG_FILE_NAME}' ثبت شدند.")

if __name__ == "__main__":
    batch_download_images()
