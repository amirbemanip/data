import json

def analyze_products():
    """
    Reads and analyzes the products_master.json file.
    """
    try:
        with open('products_master.json', 'r', encoding='utf-8') as f:
            products = json.load(f)

        print(f"Successfully loaded {len(products)} products from products_master.json")
        return products
    except FileNotFoundError:
        print("Error: products_master.json not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Could not decode JSON from products_master.json.")
        return None

import os
from thefuzz import process
from googletrans import Translator

def get_image_files():
    """Returns a list of image files from the 'cleaned' directory."""
    try:
        files = os.listdir('cleaned')
        return [f for f in files if f.endswith('_clean.png')]
    except FileNotFoundError:
        print("Error: 'cleaned' directory not found.")
        return []

def translate_text(text, dest_lang):
    """Translates text to the destination language."""
    try:
        translator = Translator()
        translation = translator.translate(text, dest=dest_lang)
        return translation.text
    except Exception as e:
        return f"Translation error: {e}"

def get_corrected_products(products, image_files):
    # (This function remains the same as the previous step)
    corrected_products = []
    for product in products:
        original_name = product.get('name', 'No name')
        best_match, score = process.extractOne(original_name, image_files)
        if score > 80:
            parts = best_match.replace('_clean.png', '').split(' - ')
            new_brand = parts[0].strip()
            new_name_fa = " - ".join(parts[1:]).strip()
        else:
            new_brand = "Uncertain"
            new_name_fa = original_name
        corrected_products.append({
            'original_product': product, 'new_name_fa': new_name_fa,
            'new_brand': new_brand, 'image_file': best_match if score > 80 else None
        })
    return corrected_products

def group_products(corrected_products):
    # (This function remains the same as the previous step)
    product_groups = {}
    for p in corrected_products:
        name = p['new_name_fa']
        keys = list(product_groups.keys())
        if keys:
            best_match, score = process.extractOne(name, keys)
            if score > 95:
                product_groups[best_match].append(p)
            else:
                product_groups[name] = [p]
        else:
            product_groups[name] = [p]
    return product_groups

if __name__ == "__main__":
    products = analyze_products()
    image_files = get_image_files()
    final_products = []

    if products and image_files:
        corrected_products = get_corrected_products(products, image_files)
        product_groups = group_products(corrected_products)

        for group_name, items in product_groups.items():
            main_item = items[0]

            # Translations
            name_fa = main_item['new_name_fa']
            brand = main_item['new_brand']
            desc_fa = f"محصول {name_fa} از برند {brand}."
            name_en = translate_text(name_fa, 'en')
            name_de = translate_text(name_fa, 'de')
            desc_en = translate_text(desc_fa, 'en')
            desc_de = translate_text(desc_fa, 'de')

            # Image URL
            image_url = f"/images/cleaned/{main_item['image_file']}" if main_item['image_file'] else ""

            # Duplicates
            duplicates = [item['original_product']['name'] for item in items[1:]]

            final_product = {
                "name": {"fa": name_fa, "en": name_en, "de": name_de},
                "description": {"fa": desc_fa, "en": desc_en, "de": desc_de},
                "brand": brand,
                "category": main_item['original_product'].get('category', ""),
                "images": [image_url],
                "is_perishable": main_item['original_product'].get('is_perishable', False),
                "duplicates": duplicates
            }
            final_products.append(final_product)

    # Write the final JSON data to a file
    with open('products_cleaned.json', 'w', encoding='utf-8') as f:
        json.dump(final_products, f, indent=4, ensure_ascii=False)

    print("Cleaned product data has been saved to products_cleaned.json")
