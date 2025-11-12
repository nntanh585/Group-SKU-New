from groupassigner import SemanticGroupAssigner
import psutil, os, gc, time

def measure_memory(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        gc.collect()
        mem_before = process.memory_info().rss / (1024 ** 2)
        start = time.time()

        result = func(*args, **kwargs)

        gc.collect()
        mem_after = process.memory_info().rss / (1024 ** 2)
        end = time.time()

        print(f"\n[Memory Profiling] {func.__name__}")
        print(f"RAM before: {mem_before:.2f} MB")
        print(f"RAM after:  {mem_after:.2f} MB")
        print(f"Δ RAM used: {mem_after - mem_before:.2f} MB")
        print(f"Execution time: {end - start:.2f}s")
        print("-" * 50)
        return result
    return wrapper

SemanticGroupAssigner._load_model = measure_memory(SemanticGroupAssigner._load_model)
SemanticGroupAssigner._read_json = measure_memory(SemanticGroupAssigner._read_json)
SemanticGroupAssigner._transform_data_from_json = measure_memory(SemanticGroupAssigner._transform_data_from_json)
SemanticGroupAssigner.build_from_json = measure_memory(SemanticGroupAssigner.build_from_json)
SemanticGroupAssigner.assign_new_product = measure_memory(SemanticGroupAssigner.assign_new_product)

# ---------------- Example usage ----------------
# Example minimal dataset (you can replace with your /content/example.json)
model = "paraphrase-multilingual-MiniLM-L12-v2"
mgr = SemanticGroupAssigner(embed_model_name=model,
                    similarity_threshold=0.9,
                    buffer_flush=2)
mgr.build_from_json("./example.json")

# Add a new product real-time
new_product = [
    {
    'product_id': 541187,
    'product_name': 'Google Pixel 4XL',
    'brand_name': 'No Brand',
    'platform_name': 'Preny Test',
    'shop_country': 'vn',
    'short_description': None,
    'price': 1000000,
    'quantity': 111106,
    'attribute_products': {'data': {'brand': 'No Brand',
    'warranty_type': 'Warranty Paper and Invoice',
    'is_hazardous': 'None'}}
    },
    {
    'product_id': 541187,
    'product_name': 'Google Pixel 2XL',
    'brand_name': 'No Brand',
    'platform_name': 'Preny Test',
    'shop_country': 'vn',
    'short_description': None,
    'price': 1000000,
    'quantity': 111106,
    'attribute_products': {'data': {'brand': 'No Brand',
    'warranty_type': 'Warranty Paper and Invoice',
    'is_hazardous': 'None'}}
    },
    {
    'product_id': 541187,
    'product_name': 'Samsung Galaxy S10',
    'brand_name': 'No Brand',
    'platform_name': 'Preny Test',
    'shop_country': 'vn',
    'short_description': None,
    'price': 1000000,
    'quantity': 111106,
    'attribute_products': {'data': {'brand': 'No Brand',
    'warranty_type': 'Warranty Paper and Invoice',
    'is_hazardous': 'None'}}
    },
    {
    'product_id': 541187,
    'product_name': 'Samsung Galaxy S20',
    'brand_name': 'No Brand',
    'platform_name': 'Preny Test',
    'shop_country': 'vn',
    'short_description': None,
    'price': 1000000,
    'quantity': 111106,
    'attribute_products': {'data': {'brand': 'No Brand',
    'warranty_type': 'Warranty Paper and Invoice',
    'is_hazardous': 'None'}}
    }
]
try:
    for new_prod in new_product:
        assigned_id = mgr.assign_new_product(new_prod)
        print(f"Result:\nNew product has been assigned Group SKU ID: {assigned_id}")
except Exception as e:
    print(f"Error: {e}")

# ------------------------------------------------
print(mgr.df[['product_name', 'group_sku_id']])
