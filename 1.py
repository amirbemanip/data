import os

# ----------------- تنظیمات -----------------
# مسیر فولدری که می‌خواهید بررسی کنید
folder_path = 'cleaned'

# پسوندهای رایج فایل‌های عکس که می‌خواهید شمرده شوند
# می‌توانید پسوندهای دیگری هم به این لیست اضافه کنید
image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
# -------------------------------------------

# شمارنده برای تعداد عکس‌ها
image_count = 0

try:
    # گرفتن لیست تمام فایل‌ها و فولدرها در مسیر مشخص شده
    items_in_folder = os.listdir(folder_path)

    # حلقه برای بررسی تک تک آیتم‌ها
    for item in items_in_folder:
        # چک می‌کنیم که آخر اسم فایل یکی از پسوندهای عکس باشد
        # .lower() برای این است که پسوندهای با حروف بزرگ مثل .JPG هم شمرده شوند
        if item.lower().endswith(image_extensions):
            # اگر فایل یک عکس بود، یکی به شمارنده اضافه کن
            image_count += 1

    print(f"✅ تعداد کل فایل‌های عکس در فولدر '{folder_path}': {image_count}")

except FileNotFoundError:
    print(f"❌ خطا: فولدر '{folder_path}' پیدا نشد. لطفاً مسیر را چک کنید.")
except Exception as e:
    print(f"یک خطای غیرمنتظره رخ داد: {e}")