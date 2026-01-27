from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd, json, re, uuid
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from models.review_pipeline import predict_label
import os

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-key")

# ---------------- Paths ----------------
DATA_PATH = Path("data/assignment3_II.csv")
INSTANCE_DIR = Path(app.instance_path)
INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
REVIEWS_PATH = INSTANCE_DIR / "reviews.json"
image_dir = Path("static/images")
image_files = os.listdir(image_dir) if image_dir.exists() else []


# ---------------- Load data ----------------
items = pd.read_csv(DATA_PATH)
items = items.drop_duplicates(subset=["Clothes Title", "Clothes Description"]).reset_index(drop=True)
items["item_id"] = range(len(items))
items["search_text"] = (
    items["Clothes Title"].fillna("") + " " + items["Clothes Description"].fillna("")
).str.lower()
items["Category"] = (
    items["Division Name"].fillna("") + " / " +
    items["Department Name"].fillna("") + " / " +
    items["Class Name"].fillna("")
)
image_files = os.listdir(image_dir)

dataset_reviews = (
    pd.read_csv(DATA_PATH)
    .groupby(["Clothes Title", "Clothes Description"])
    .apply(lambda df: df[["Title", "Review Text", "Rating", "Recommended IND"]].to_dict("records"))
    .to_dict()
)

def find_image(title):
    # Normalize: lowercase, remove spaces for matching
    clean_title = title.lower().replace(" ", "")
    for f in image_files:
        if os.path.splitext(f)[0].lower().replace(" ", "") == clean_title:
            return f  # return filename (e.g., "Elegant A-Line Dress.jpeg")
    return None  # no match found

items["Image"] = items["Clothes Title"].apply(find_image)

# ---------------- Review helpers ----------------
def read_reviews():
    if not REVIEWS_PATH.exists():
        return []
    with open(REVIEWS_PATH, "r") as f:
        return json.load(f)

def write_reviews(r):
    with open(REVIEWS_PATH, "w") as f:
        json.dump(r, f, indent=2)


# ---------------- Average rating helper ----------------
def compute_average_ratings():
    """Compute average rating (dataset + user reviews) for each item."""
    avg_map = {}

    # Dataset reviews grouped by item_id equivalents
    df = pd.read_csv(DATA_PATH)
    grouped = df.groupby(["Clothes Title", "Clothes Description"])
    for (title, desc), group in grouped:
        valid = group["Rating"].dropna().astype(float)
        if len(valid):
            avg_map[(title, desc)] = {"ratings": valid.tolist()}

    # Add user reviews
    for r in read_reviews():
        item = items.iloc[r["item_id"]]
        key = (item["Clothes Title"], item["Clothes Description"])
        try:
            val = float(r.get("rating", 0))
            if 1 <= val <= 5:
                avg_map.setdefault(key, {"ratings": []})["ratings"].append(val)
        except ValueError:
            continue

    # Compute average per item
    result = {}
    for k, v in avg_map.items():
        ratings = v["ratings"]
        result[k] = round(sum(ratings) / len(ratings), 2) if ratings else None
    return result


# Global cache (will update dynamically)
average_ratings = compute_average_ratings()

items["avg_rating"] = items.apply(
    lambda row: average_ratings.get((row["Clothes Title"], row["Clothes Description"]), None),
    axis=1
)


# Add this: store original review text & rating from CSV
if "Review Text" in items.columns:
    items["review_text"] = items["Review Text"].fillna("No reviews available.")
else:
    items["review_text"] = "No reviews available."

# ---------------- Build TF-IDF search ----------------
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
tfidf = vectorizer.fit_transform(items["search_text"])

# ---------------- Routes ----------------
@app.route("/")
def home():
    view = request.args.get("view", "featured")  # default view
    q = request.args.get("q", "").strip().lower()

    if q:
        vec = vectorizer.transform([q])
        sims = linear_kernel(vec, tfidf).flatten()
        idx = sims.argsort()[::-1]
        results = [items.iloc[i].to_dict() for i in idx if sims[i] > 0]
    else:
        if view == "featured":
            results = items[items["avg_rating"].fillna(0) >= 4].head(50).to_dict("records")
        elif view == "category":
            # show unique categories
            results = sorted(items["Category"].unique().tolist())
        else:
            results = items.head(50).to_dict("records")

    return render_template("index.html", items=results, query=q, view=view)


@app.route("/category/<path:cat>")
def category_page(cat):
    cat = cat.lower()
    subset = items[items["Category"].str.lower() == cat]
    if subset.empty:
        flash(f"No items found for category: {cat}")
        return redirect(url_for("home"))
    return render_template("category.html", category=cat, items=subset.to_dict("records"))

@app.route("/item/<int:item_id>")
def item_page(item_id):
    item = items.iloc[item_id].to_dict()

    # --- Dataset reviews (could be multiple per item) ---
    ds_key = (item["Clothes Title"], item["Clothes Description"])
    dataset_reviews_list = dataset_reviews.get(ds_key, [])
    dataset_reviews_formatted = [
        {
            "title": r.get("Title", "(no title)"),
            "description": r.get("Review Text", ""),
            "rating": r.get("Rating", "N/A"),
            "label": "Dataset"
        }
        for r in dataset_reviews_list
    ]

    # --- User-added reviews ---
    user_reviews = [r for r in read_reviews() if r["item_id"] == item_id]

    # --- Combine: user reviews first ---
    reviews = user_reviews + dataset_reviews_formatted

    # --- Compute average rating ---
    valid_ratings = []
    for r in reviews:
        try:
            rating = float(r.get("rating", 0))
            if 1 <= rating <= 5:
                valid_ratings.append(rating)
        except ValueError:
            pass

    avg_rating = round(sum(valid_ratings) / len(valid_ratings), 2) if valid_ratings else None

    return render_template("item.html", item=item, reviews=reviews, avg_rating=avg_rating)


@app.route("/item/<int:item_id>/review/new", methods=["GET", "POST"])
def new_review(item_id):
    item = items.iloc[item_id].to_dict()

    # form structure for template
    form = {
        "title": "",
        "description": "",
        "rating": "",
        "predicted_label": None,
        "label": None
    }

    if request.method == "POST":
        action = request.form.get("action")
        desc = request.form.get("description", "")
        form["title"] = request.form.get("title", "")
        form["description"] = desc
        form["rating"] = request.form.get("rating", "")

        if action == "predict":
            # call ML model to predict sentiment
            form["predicted_label"] = predict_label(desc)
            form["label"] = form["predicted_label"]
            flash(f"Model prediction: {'Recommended ✅' if form['predicted_label'] else 'Not Recommended ❌'}")
            return render_template("review_form.html", item=item, form=form)

        elif action == "save":
             # get label chosen in dropdown
            label = int(request.form.get("label", "1"))
            rid = str(uuid.uuid4())[:8]
            data = read_reviews()
            data.append({
                "review_id": rid,
                "item_id": item_id,
                "title": form["title"],
                "description": form["description"],
                "rating": form["rating"],
                "label": label
            })
            write_reviews(data)

        # Recompute average ratings dynamically
        global average_ratings
        average_ratings = compute_average_ratings()
        items["avg_rating"] = items.apply(
            lambda row: average_ratings.get((row["Clothes Title"], row["Clothes Description"])), axis=1
        )

        flash("✅ Review added successfully")
        return redirect(url_for("item_page", item_id=item_id))


    return render_template("review_form.html", item=item, form=form)

average_ratings = compute_average_ratings()
items["avg_rating"] = items.apply(
    lambda row: average_ratings.get((row["Clothes Title"], row["Clothes Description"]), None),
    axis=1
)



@app.route("/review/<review_id>")
def show_review(review_id):
    r = [x for x in read_reviews() if x["review_id"]==review_id]
    if not r:
        return redirect("/")
    review = r[0]
    item = items.iloc[review["item_id"]].to_dict()
    return render_template("review_show.html", review=review, item=item)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

