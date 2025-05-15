import ikea_api
import json
import asyncio  # 用於執行 async 函式

# 設定國家與語言
constants = ikea_api.Constants(country="us", language="en")

# 搜尋 API
search = ikea_api.Search(constants)

async def fetch_ikea_data():
    """ 非同步執行 IKEA API 請求,確保獲取最大可能數量的產品 """
    
    # 先查詢 1 個,拿到 max 值
    print(" 正在獲取最大產品數量...")
    endpoint = search.search("bed", limit=1)
    result = await ikea_api.run_async(endpoint)

    # 取得 `max` 表示所有符合條件的產品數量
    max_products = result["searchResultPage"]["products"]["main"]["max"]
    print(f"產品總數：{max_products}")

    # 設定最大限制,避免超過 API 限制
    max_limit = min(max_products, 2)

    # 重新查詢所有可用產品
    print(f"🔍 重新查詢,最多獲取 {max_limit} 個產品...")
    endpoint = search.search("bed", limit=max_limit)
    result = await ikea_api.run_async(endpoint)

    # 取得產品清單
    products = result["searchResultPage"]["products"]["main"]["items"]
    print(f"✅ 共獲取 {len(products)} 項產品。")

    # 儲存 JSON 結果到檔案
    with open("ikea_products.json", "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)

    print("📁 資料已儲存到 ikea_products.json")

    # 解析 JSON 並提取產品資訊
    for product in products:
        name = product.get("product", {}).get("name", "N/A")
        type_name = product.get("product", {}).get("typeName", "N/A")
        price = product.get("product", {}).get("salesPrice", {}).get("numeral", "N/A")
        currency = product.get("product", {}).get("salesPrice", {}).get("currencyCode", "N/A")
        rating = product.get("product", {}).get("ratingValue", "N/A")
        rating_count = product.get("product", {}).get("ratingCount", "N/A")
        product_link = product.get("product", {}).get("pipUrl", "N/A")
        image_url = product.get("product", {}).get("mainImageUrl", "N/A")

        print(f"📌 產品名稱: {name} ({type_name})")
        print(f"💲 價格: {currency} {price}")
        print(f"⭐ 評價: {rating} 分 ({rating_count} 則評論)")
        print(f"🔗 購買連結: {product_link}")
        print(f"🖼 圖片連結: {image_url}")

        # 解析變體資訊
        variants = product.get("product", {}).get("gprDescription", {}).get("variants", [])
        if variants:
            print("🔄 產品變體:")
            for variant in variants:
                variant_url = variant.get("pipUrl", "N/A")
                variant_name = variant.get("name", "N/A")
                variant_size = variant.get("itemMeasureReferenceText", "N/A")
                print(f"  🔹 {variant_name} ({variant_size}) - 變體連結: {variant_url}")

        print("-" * 50)



# 使用 asyncio 執行非同步函式
asyncio.run(fetch_ikea_data())

