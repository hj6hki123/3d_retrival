import ikea_api
import json
import asyncio
import time
import random
import urllib.parse
import re
from tqdm import tqdm
from pymongo import MongoClient
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# IKEA API 設定
constants = ikea_api.Constants(country="us", language="en")
search = ikea_api.Search(constants)

# 連線至 MongoDB
client = MongoClient("mongodb://admin:zmxcnbv@localhost:27017/?authSource=admin")
db = client["furniture_db"]
collection = db.ikea_product

# 定義家具類別名稱
names = [
    #"Sofas", "Chairs", "Tables", 
    "Beds", "Wardrobes", "Bookcases", "Desks", 
    "Benches", "Stools", "Ottomans", "Sideboards", "Consoles", "Recliners", 
    "Sectionals", "Nightstands", "Cabinets"
]

# 欄位對應
field_mapping = {
    "image": "mainImageUrl", 
    "name": "name", 
    "url": "pipUrl", 
    "description": "description",
    "price": "salesPrice", 
    "color_options": "colors", 
    "view_url": "pipUrl",
    "Image_matting": "Image_matting", 
    "images": "images", 
    "price_string": "price_string",
    "size_options": "itemMeasureReferenceText", 
    "asin": "id", 
    "star": "ratingValue",
    "brand": "brand", 
    "starNum": "ratingCount", 
    "type": "typeName", 
    "category": "category",
    "variant_urls": "variant_urls"  
}

# 取得缺漏欄位資訊（保留原本的函式）
async def fetch_missing_fields(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(user_agent="Mozilla/5.0")
                page = await context.new_page()
                await page.goto(url, timeout=60000)
                await page.wait_for_timeout(2000)
                html = await page.content()
                await browser.close()

                soup = BeautifulSoup(html, 'html.parser')
                result = {}

                # 抓取產品描述
                container = soup.find("div", class_="pip-product-details__container")
                result["description"] = "\n".join(p.get_text(strip=True) for p in container.find_all("p")) if container else ""

                # 抓取評分
                rating_section = soup.find("span", class_="pip-highlight-reviews__header")
                result["star"] = rating_section.get_text(strip=True) if rating_section else ""

                # 抓取評論數
                reviews_section = soup.find("div", class_="pip-highlight-reviews__card-total-reviews")
                result["starNum"] = reviews_section.get_text(strip=True) if reviews_section else ""

                # 抓取產品類別
                breadcrumbs = soup.find("ol", class_="bc-breadcrumb__list")
                result["category"] = breadcrumbs.find_all("li")[1].find("span").get_text(strip=True) if breadcrumbs else ""

                # 抓取圖片
                gallery = soup.find('div', class_='pip-product-gallery__thumbnails')
                result["images"] = [img.get('src') for img in gallery.find_all('img')] if gallery else []
                
                ## 獲取尺寸
                size_container = soup.find("div", class_="pip-product-dimensions__dimensions-container")
                dimensions = {}

                if size_container:
                    measurements = size_container.find_all("p", class_="pip-product-dimensions__measurement-wrapper")
                    for m in measurements:
                        # 找出測量名稱所在的 span
                        name_span = m.find("span", class_="pip-product-dimensions__measurement-name")
                        if name_span:
                            # 取得測量名稱,移除冒號及空白字元
                            key = name_span.get_text(strip=True).replace(":", "")
                            # 取得該 span 之後的文字節點,作為測量值
                            value = name_span.next_sibling
                            if value:
                                value = value.strip()
                            dimensions[key] = value
                    result["size_options"] = dimensions
                else:
                    result["size_options"] = ""
                

                # 其他基本資料
                result["view_url"] = url
                result.update({
                    "source": "IKEA",
                    "price_string": "USD",
                    "brand": "IKEA",
                    "color_options": "",
                    "Image_matting": ""
                })

                return result

        except Exception as e:
            print(f"Error fetching missing fields, retry {attempt + 1}/{max_retries}: {e}")
            await asyncio.sleep(random.uniform(2, 5))  # `await` 讓 asyncio 正確運行

    return {}

# 使用 asyncio 執行 IKEA API
async def fetch_ikea_data(keyword):
    print(f"🔍 正在查詢 {keyword} 的產品總數...")

    # 先查詢 1 個,取得 `max` 產品數量
    endpoint = search.search(keyword, limit=1)
    result = await ikea_api.run_async(endpoint)

    # **檢查 API 回傳內容**
    if not isinstance(result, dict):
        print(f"❌ API 回傳異常（{keyword}）:{result}")
        return []

    if "searchResultPage" not in result:
        print(f"⚠️ API 沒有返回 `searchResultPage`,回傳內容:{json.dumps(result, indent=2)}")
        return []

    max_products = result["searchResultPage"]["products"]["main"].get("max", 0)
    print(f"✅ {keyword} 共有 {max_products} 項產品")

    # 設定最大獲取數量（避免 API 限制）
    max_limit = min(max_products, 2000)  # 最多 2000 項產品

    print(f"🔍 重新查詢 {keyword},最多獲取 {max_limit} 項產品...")
    endpoint = search.search(keyword, limit=max_limit)
    result = await ikea_api.run_async(endpoint)

    # **再次檢查 API 回傳內容**
    if not isinstance(result, dict):
        print(f"❌ API 回傳異常（{keyword}）:{result}")
        return []

    if "searchResultPage" not in result:
        print(f"⚠️ API 沒有返回 `searchResultPage`,回傳內容:{json.dumps(result, indent=2)}")
        return []

    # 取得產品清單
    products = result["searchResultPage"]["products"]["main"].get("items", [])

    # **確保回傳的是 list**
    if not isinstance(products, list):
        print(f"⚠️ API 回傳的產品格式異常（{keyword}）:{json.dumps(result, indent=2)}")
        return []

    return products


# 主程式
async def main():
    for category_name in tqdm(names, desc="Processing categories"):
        products = await fetch_ikea_data(category_name)
        if not products:
            print(f"{category_name} 分類資料獲取失敗。")
            continue

        for item in tqdm(products, desc=f"Processing {category_name} documents", leave=False):
            if not isinstance(item, dict):
                print(f"異常資料 (非dict): {item}")
                continue  # 跳過異常資料

            product_data = item["product"]  
            extracted = {k: product_data.get(v, None) for k, v in field_mapping.items()}

            # 解析價格
            price_info = extracted.get("price")
            if price_info and isinstance(price_info, dict):
                extracted["price"] = price_info.get("numeral")  # 提取價格數值
                extracted["price_string"] = price_info.get("currencyCode", "USD")  # 提取貨幣類型

            # 解析變體網址
            variants = item.get("product", {}).get("gprDescription", {}).get("variants", [])
            extracted["variant_urls"] = [v.get("pipUrl", "") for v in variants if v.get("pipUrl")]

            # 解析缺失資訊
            missing_fields = [k for k, v in extracted.items() if not v]
            if missing_fields and extracted.get("url"):
                additional_data = await fetch_missing_fields(extracted["url"])
                for field in missing_fields:
                    extracted[field] = additional_data.get(field, extracted[field])

            # 存入 MongoDB
            collection.insert_one(extracted)
            time.sleep(random.uniform(1, 3))  # 每個商品等待

        time.sleep(random.uniform(5, 10))  # 每個分類等待

    print("✅ 所有資料已存入 MongoDB!")

# 執行主程式
asyncio.run(main())
