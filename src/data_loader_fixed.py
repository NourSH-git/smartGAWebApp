import pandas as pd

REQUIRED = {
    "users": {"id_user", "age", "location"},
    "products": {"id_product", "category", "price"},
    "ratings": {"user_id", "product_id", "rating"},
    "behavior": {"user_id", "product_id", "viewed", "clicked", "purchased"},
}

def _clean_columns(df):
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in df.columns
    ]
    return df


def _check_columns(df, required, name):
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")

def validate_and_prepare(users_file, products_file, ratings_file, behavior_file):

    users = _clean_columns(pd.read_excel(users_file))    
    # ✅ الحل هون قبل التحقق
    users = users.rename(columns={
        "user_id": "id_user",
        "country": "location"
    })
    products = _clean_columns(pd.read_excel(products_file))
    products = products.rename(columns={
    "product_id": "id_product"
})

    ratings = _clean_columns(pd.read_excel(ratings_file))
    behavior = _clean_columns(pd.read_excel(behavior_file))

    _check_columns(users, REQUIRED["users"], "users.xlsx")
    _check_columns(products, REQUIRED["products"], "products.xlsx")
    _check_columns(ratings, REQUIRED["ratings"], "ratings.xlsx")
    _check_columns(behavior, REQUIRED["behavior"], "behavior.xlsx")



    products = products.drop_duplicates(subset=["id_product"])

    products["price"] = pd.to_numeric(products["price"], errors="coerce").fillna(0)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce").clip(1, 5).fillna(3)

    for col in ["viewed", "clicked", "purchased"]:
        behavior[col] = pd.to_numeric(behavior[col], errors="coerce").fillna(0).clip(0, 1)

    return users, products, ratings, behavior
