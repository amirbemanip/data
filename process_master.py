import json
import re
from collections import Counter
from urllib.parse import unquote

# ==============================================================================
# بخش تنظیمات: این بخش بسیار گسترش یافته است
# ==============================================================================

# لیستی جامع از برندها برای تشخیص خودکار
BRAND_KEYWORDS = [
    "ساکی", "خانوم خانوما", "خانوم", "گلها", "آونگ", "یک و یک", "یک ویک", "سبزان", "هاتی کارا", 
    "برتر", "زرین", "پگاه", "شمس", "تکدانه", "شادلی", "الوند", "آبعلی", "راگا", 
    "تکرم", "مزمز", "سی تو", "لینا", "چی توز", "مینو", "ستایش", "هانیم", "پارنا", 
    "ترشین", "همپا", "شاهان", "دوغزال", "احمد", "لاهیجان", "صحت", "پارس خزر",
    "پرژک", "گلنار", "شمیم", "اکبر مشتی", "دلپذیر", "کامچین", "هانی", "نیک", 
    "دلگاتو", "کاله", "صباح", "عقاب", "شیررضا", "محمود", "محسن", "طبیعت", "اصالت",
    "آیدا", "عالیس", "زمزم", "کاسل", "سن ایچ", "بیژن", "بدر"
]

# کلمات کلیدی برای تشخیص دسته‌بندی محصولات (با اولویت‌بندی)
CATEGORY_KEYWORDS = {
    "لبنیات": ["دوغ", "ماست", "کشک", "پنیر"],
    "نوشیدنی": ["شربت", "نوشیدنی", "دلستر", "آبمیوه", "نوشابه", "آب", "لیموناد"],
    "چای و دمنوش": ["چای", "دمنوش"],
    "کنسرو و غذای آماده": ["کنسرو", "خوراک", "خورشت", "حلیم", "آش", "رب"],
    "خشکبار و غلات": ["برنج", "آرد", "لوبیا", "عدس", "نخود", "ماش", "لپه", "گندم", "جو", "بلغور", "سویا"],
    "تنقلات و شیرینی‌جات": ["بیسکویت", "نبات", "قند", "نقل", "شکرپنیر", "آبنبات", "چیپس", 
                           "کرانچی", "اسنک", "پفک", "آدامس", "سوهان", "گز", "باسلوق", "باقلوا", 
                           "کیک", "کلوچه", "پشمک", "لواشک", "ترشک"],
    "آجیل و میوه خشک": ["پسته", "بادام", "کشمش", "تخمه", "آجیل", "زردآلو", "انجیر", 
                        "گردو", "توت", "خرما", "برگه", "آلو"],
    "سس، مربا و ترشیجات": ["سس", "مربا", "ترشی", "خیارشور", "زیتون", "شیره"],
    "ادویه‌جات و چاشنی‌ها": ["ادویه", "زعفران", "نمک", "فلفل", "زردچوبه", "دارچین", "گلپر", 
                           "سماق", "زیره", "هل", "وانیل", "آویشن", "پودر"],
    "عرقیجات و طعم‌دهنده‌ها": ["گلاب", "عرق", "سرکه", "آبغوره", "آبلیمو"],
    "محصولات منجمد": ["منجمد", "یخی"],
    "لوازم غیر خوراکی": ["تنور", "سینی", "شامپو", "اجاق", "کباب پز", "صابون", "سیخ", 
                        "منقل", "اسپندسوز", "لیف", "کیسه", "پلوپز", "کتاب", "اسکاچ"]
}

# کلمات کلیدی برای تشخیص فاسدشدنی بودن
PERISHABLE_KEYWORDS = ["دوغ", "ماست", "کشک", "پنیر", "منجمد", "یخی", "بستنی", "فالوده"]

# ==============================================================================
# هسته اصلی اسکریپت
# ==============================================================================

def clean_text(text):
    """حذف کاراکترهای اضافی و فاصله‌های زائد"""
    text = text.replace('✏️', '').replace('🔗', '').strip()
    # حذف RTL/LTR marks
    text = text.replace('\u200e', '').replace('\u200f', '')
    return ' '.join(text.split())

def find_best_name(lines):
    """پیدا کردن بهترین نام از بین گزینه‌های موجود با شمارش فراوانی"""
    name_candidates = []
    for line in lines:
        if '✏️' in line:
            name = clean_text(line)
            if name.lower() != 'empty' and len(name) > 5:
                # حذف کلمه "عدد" از انتهای نام
                if name.endswith(" عدد"):
                    name = name[:-4].strip()
                name_candidates.append(name)
    
    if not name_candidates:
        return None
        
    # پیدا کردن رایج‌ترین نام
    counter = Counter(name_candidates)
    return counter.most_common(1)[0][0]

def extract_images(lines):
    """استخراج تمام لینک‌های تصویر از بلوک"""
    images = set()
    for line in lines:
        matches = re.findall(r'https?://\S+\.(?:jpg|jpeg|png|webp|gif)', line, re.IGNORECASE)
        for match in matches:
            images.add(match)
    return list(images)

def extract_attributes(lines):
    """استخراج قیمت، وزن و حجم از متن"""
    attributes = {'price': None, 'currency': None, 'weight': None, 'weight_unit': None}
    
    for line in lines:
        # استخراج قیمت
        price_match = re.search(r'€(\d+[\.,]?\d*)', line)
        if price_match and attributes['price'] is None:
            attributes['price'] = float(price_match.group(1).replace(',', '.'))
            attributes['currency'] = 'EUR'

        # استخراج وزن/حجم
        # Regex to capture value and unit, handling variations
        weight_match = re.search(r'(\d+[\.,]?\d*)\s*(kg|g|کیلوگرم|گرم|کیلو|لیتر|میلی|ml|l)', line, re.IGNORECASE)
        if weight_match and attributes['weight'] is None:
            value = float(weight_match.group(1).replace(',', '.'))
            unit = weight_match.group(2).lower()
            
            # نرمال‌سازی واحد
            if unit in ["کیلوگرم", "kg", "کیلو"]:
                attributes['weight_unit'] = 'kg'
            elif unit in ["g", "گرم"]:
                attributes['weight_unit'] = 'g'
            elif unit in ["l", "لیتر"]:
                attributes['weight_unit'] = 'l'
            elif unit in ["ml", "میلی"]:
                 attributes['weight_unit'] = 'ml'
            attributes['weight'] = value

    return attributes
    
def find_brand(name):
    """بر اساس نام محصول، برند را پیدا می‌کند"""
    for brand in BRAND_KEYWORDS:
        if brand in name:
            return brand
    return "متفرقه"

def find_category(name):
    """بر اساس نام محصول، دسته‌بندی را پیدا می‌کند"""
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in name:
                return category
    return "سایر موارد"

def is_perishable(name, category):
    """تشخیص می‌دهد که محصول فاسدشدنی است یا خیر"""
    if category in ["لبنیات", "محصولات منجمد"]:
        return True
    for keyword in PERISHABLE_KEYWORDS:
        if keyword in name:
            return True
    return False

def process_product_block(block):
    """پردازش کامل یک بلوک محصول"""
    lines = block.strip().split('\n')
    if len(lines) < 2:
        return None

    name = find_best_name(lines)
    if not name:
        return None
        
    images = extract_images(lines)
    attributes = extract_attributes(lines)
    brand = find_brand(name)
    category = find_category(name)
    perishable = is_perishable(name, category)
    
    description = f"محصول {name} از برند {brand}."
    if attributes['weight'] and attributes['weight_unit']:
        description += f" با وزن {attributes['weight']} {attributes['weight_unit']}."

    product_dict = {
        "name": name,
        "description": description,
        "brand": brand,
        "category": category,
        "images": images,
        "is_perishable": perishable,
        "price": attributes['price'],
        "currency": attributes['currency'],
        "weight": attributes['weight'],
        "weight_unit": attributes['weight_unit'],
    }
    
    return product_dict

def main():
    """تابع اصلی برنامه"""
    file_names = [
        "1.txt", "1 copy 2.txt", "1 copy 3.txt", "2.txt", "2 copy.txt", "3.txt",
        "persische-feinkost all.txt", "persische-lebensmittel24 copy.txt", "persische-lebensmittel24.txt"
    ]
    
    all_content = ""
    for file_name in file_names:
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                print(f"Reading file: {file_name}...")
                all_content += f.read() + "\n,,,\n" # Add separator for safety
        except FileNotFoundError:
            print(f"Warning: File '{file_name}' not found. Skipping.")
            continue
    
    # جداسازی بلوک‌ها با یک Regex قدرتمند که هم بر اساس شماره ردیف و هم ",,," کار می‌کند
    product_blocks = re.split(r'\n(?=\d+\t)|,,,', all_content)
    
    all_products = []
    print(f"\nFound {len(product_blocks)} potential product blocks. Starting processing...")

    for i, block in enumerate(product_blocks):
        if not block.strip():
            continue
        
        processed_product = process_product_block(block)
        if processed_product:
            all_products.append(processed_product)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(product_blocks)} blocks...")

    # ذخیره نتیجه در فایل JSON
    output_filename = "products_master.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_products, f, indent=2, ensure_ascii=False)

    print(f"\nProcessing complete. {len(all_products)} products successfully extracted and saved to '{output_filename}'.")


if __name__ == "__main__":
    main()
    