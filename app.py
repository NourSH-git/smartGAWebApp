from flask import Flask, render_template, request ,jsonify
import pandas as pd

from src.data_loader_fixed import validate_and_prepare
from src.recommender import GARecommender

app = Flask(__name__)

# 🔥 تحميل البيانات مباشرة
users, products, ratings, behavior = validate_and_prepare(
    "data/sample/users.xlsx",
    "data/sample/products.xlsx",
    "data/sample/ratings.xlsx",
    "data/sample/behavior.xlsx"
)

# توحيد الاسم
users = users.rename(columns={"id_user": "user_id"})

engine = GARecommender(users, products, ratings, behavior)

@app.route("/")
def home():
    user_ids = users["user_id"].unique()
    return render_template("index.html", users=user_ids, results=None)


@app.route("/api/recommend", methods=["POST"])
def recommend_api():
    data = request.get_json()
    user_id = data.get("user_id")

    if user_id is None:
        return jsonify({"error": "user_id is required"}), 400

    # الخوارزمية ترجع قيمتين: result_df و history
    result_df, history = engine.recommend(user_id)

    if result_df is None or result_df.empty:
        return jsonify({"recommendations": []})

    recommendations = []
    for _, row in result_df.iterrows():
        recommendations.append({
            "product_id": int(row["id_product"]),
            "category": row["category"],
            "price": float(row["price"]),
            "avg_rating": float(row["avg_rating"]),
            "reason": row["reason"],
            "score": float(row["final_fitness"])
        })

    return jsonify({"recommendations": recommendations})



if __name__ == "__main__":
    app.run(debug=True)
