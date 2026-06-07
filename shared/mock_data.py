"""
Rich mock data generators simulating enterprise data sources.
Used when DEMO_MODE=true or when MuleSoft/Salesforce stubs are unavailable.
Generates realistic, deterministic data using Faker with fixed seeds.
"""
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Demo 1 – Sentiment: customer reviews & support tickets
# ---------------------------------------------------------------------------

REVIEW_TEMPLATES = [
    {"sentiment": "POSITIVE", "body": "Absolutely love this dress! The floral pattern is gorgeous and the fit is incredibly flattering. The material feels high quality and breathable for summer.", "rating": 5},
    {"sentiment": "NEGATIVE", "body": "Very disappointed. The design is cute but the zipper broke after wearing it just once. The fabric is also much thinner than it appeared in the photos. Not worth the price.", "rating": 2},
    {"sentiment": "MIXED", "body": "The color is exactly as pictured and the style is very trendy. However, the sizing runs extremely small, especially around the shoulders. I had to return it and size up two sizes.", "rating": 3},
    {"sentiment": "POSITIVE", "body": "Best purchase I've made this season. It drapes beautifully, doesn't wrinkle easily, and the stitching is perfect. I received so many compliments wearing this to a wedding.", "rating": 5},
    {"sentiment": "NEGATIVE", "body": "Terrible quality control. The hem started unraveling the moment I took it out of the package. The color also faded significantly after the very first cold wash.", "rating": 1},
    {"sentiment": "POSITIVE", "body": "Great value for money. The fabric has a nice stretch to it making it very comfortable for all-day office wear. The pockets are a fantastic and practical addition.", "rating": 4},
    {"sentiment": "NEUTRAL", "body": "It's an okay top. The material is a bit scratchy at first but softens up. The cut is somewhat boxy. It's fine for running errands but I wouldn't wear it to a nice dinner.", "rating": 3},
    {"sentiment": "NEGATIVE", "body": "The dress arrived with a strange chemical smell. The proportions are completely off—the torso is way too short. The return process was also quite frustrating.", "rating": 2},
    {"sentiment": "POSITIVE", "body": "Exceeded my expectations! The knit is so soft and cozy without being bulky. The neckline is very elegant. I've already ordered it in two more colors.", "rating": 5},
    {"sentiment": "MIXED", "body": "Beautiful embroidery and great design concept, but the lining is 100% polyester which makes it very sweaty to wear in the heat. A cotton lining would have made this a 5-star piece.", "rating": 3},
]

PRODUCT_NAMES = [
    "Floral Chiffon Midi Dress", "Belted Wide-Leg Trousers", "Ribbed Knit Turtleneck",
    "Tailored Linen Blazer", "High-Waist Mom Jeans", "Bohemian Ruffle Blouse",
    "Satin Slip Midi Skirt", "Cashmere Blend Cardigan", "Cotton Poplin Wrap Dress",
    "Faux Leather Moto Jacket", "Pleated Velvet Maxi Dress", "Embroidered Peasant Top",
]


def generate_reviews(count: int = 20) -> list[dict]:
    """Generate realistic product review data."""
    reviews = []
    for i in range(count):
        template = REVIEW_TEMPLATES[i % len(REVIEW_TEMPLATES)]
        reviews.append({
            "review_id": f"REV-{fake.unique.random_int(min=10000, max=99999)}",
            "customer_name": fake.name(),
            "product": random.choice(PRODUCT_NAMES),
            "rating": template["rating"],
            "title": fake.sentence(nb_words=6).rstrip("."),
            "body": template["body"],
            "sentiment_expected": template["sentiment"],
            "date": (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat(),
            "channel": random.choice(["website", "mobile_app", "email", "social_media"]),
        })
    return reviews


# ---------------------------------------------------------------------------
# Demo 2 – Recommender: product catalog
# ---------------------------------------------------------------------------

CATALOG_ITEMS = [
    {"name": "Floral Chiffon Midi Dress", "category": "Dresses", "price": 59.99, "description": "Flowy midi dress in sheer chiffon with a vibrant floral print. Features a V-neck, flutter sleeves, and a tiered skirt. Fully lined. Perfect for summer weddings and garden parties.", "tags": ["summer", "floral", "wedding guest", "midi", "chiffon"]},
    {"name": "Belted Wide-Leg Trousers", "category": "Trousers", "price": 49.99, "description": "High-waisted tailored trousers with a dramatic wide-leg silhouette. Includes a matching D-ring belt. Made from a breathable linen blend. Ideal for smart-casual office wear.", "tags": ["office", "tailored", "wide-leg", "linen", "smart casual"]},
    {"name": "Ribbed Knit Turtleneck", "category": "Knitwear", "price": 34.99, "description": "Fitted turtleneck sweater in a soft ribbed knit. Viscose blend for stretch and shape retention. A versatile layering piece for autumn and winter.", "tags": ["winter", "layering", "knit", "turtleneck", "basics"]},
    {"name": "Tailored Linen Blazer", "category": "Outerwear", "price": 89.99, "description": "Single-breasted blazer in pure lightweight linen. Features notch lapels, flap pockets, and a back vent. Unlined for maximum breathability during warm weather.", "tags": ["blazer", "linen", "summer", "office", "tailored"]},
    {"name": "High-Waist Mom Jeans", "category": "Denim", "price": 39.99, "description": "Classic 90s-inspired mom jeans in rigid cotton denim. High waist, relaxed fit through the thigh, and tapered leg. Vintage mid-blue wash with subtle distressing.", "tags": ["denim", "vintage", "jeans", "casual", "cotton"]},
    {"name": "Bohemian Ruffle Blouse", "category": "Tops", "price": 29.99, "description": "Lightweight cotton voile blouse with delicate ruffles at the neckline and cuffs. Relaxed fit with a tie-string detail. Perfect for a breezy, romantic weekend look.", "tags": ["boho", "ruffles", "blouse", "casual", "romantic"]},
    {"name": "Satin Slip Midi Skirt", "category": "Skirts", "price": 44.99, "description": "Elegant bias-cut slip skirt in luxurious liquid satin. Hidden elastic waistband. Drapes beautifully for a sleek, minimalist evening or date-night outfit.", "tags": ["satin", "skirt", "evening", "minimalist", "elegant"]},
    {"name": "Cashmere Blend Cardigan", "category": "Knitwear", "price": 129.00, "description": "Luxuriously soft V-neck cardigan knitted from a premium cashmere and merino wool blend. Features tortoiseshell buttons and ribbed trims. A cozy, timeless wardrobe staple.", "tags": ["cashmere", "cardigan", "cozy", "winter", "premium"]},
    {"name": "Cotton Poplin Wrap Dress", "category": "Dresses", "price": 69.99, "description": "Crisp cotton poplin dress in a flattering wrap silhouette. Features puff sleeves, a self-tie waist belt, and a flared skirt. Great for brunches and daytime events.", "tags": ["wrap dress", "cotton", "daytime", "puff sleeve", "brunch"]},
    {"name": "Faux Leather Moto Jacket", "category": "Outerwear", "price": 79.99, "description": "Edgy biker jacket in buttery-soft vegan leather. Features asymmetric zip fastening, notched lapels with snap fasteners, and zipped cuffs. Adds attitude to any outfit.", "tags": ["jacket", "leather", "biker", "edgy", "vegan"]},
    {"name": "Pleated Velvet Maxi Dress", "category": "Dresses", "price": 99.99, "description": "Stunning floor-length evening gown in rich, crushed velvet. Features a plunging neckline, empire waist, and accordion-pleated skirt. The ultimate holiday party dress.", "tags": ["evening", "velvet", "maxi", "party", "elegant", "holiday"]},
    {"name": "Embroidered Peasant Top", "category": "Tops", "price": 34.99, "description": "Relaxed-fit cotton tunic top featuring intricate floral embroidery on the yoke and billowy balloon sleeves. Perfect for festivals and warm-weather vacations.", "tags": ["embroidery", "tunic", "festival", "vacation", "cotton"]},
]


def generate_product_catalog() -> list[dict]:
    """Return the full product catalog."""
    return [
        {**item, "product_id": f"PROD-{i+1:04d}"}
        for i, item in enumerate(CATALOG_ITEMS)
    ]


# ---------------------------------------------------------------------------
# Demo 3 – Customer 360: interaction logs
# ---------------------------------------------------------------------------

INTERACTION_TEMPLATES = [
    {"type": "twitter_public", "summary": "Customer tweeted: 'Absolutely unacceptable. Ordered the {product} for an event and it still hasn\\'t shipped after 5 days. @HM_Support do better.' - Agent replied with DM link."},
    {"type": "email_ticket", "summary": "Customer emailed regarding damaged item in order #{order}. Sent photos of a ripped seam on the {product}. Refund processed."},
    {"type": "phone_support", "summary": "Customer called highly frustrated about billing/payment issue. Double-charged on credit card for order #{order}. Initiated charge reversal and issued a $20 coupon as apology. Call duration: 22 mins."},
    {"type": "purchase", "summary": "Customer purchased {product}. Total amount: ${amount:.2f}. Applied H&M Member loyalty discount code."},
    {"type": "survey_response", "summary": "Customer completed post-purchase NPS survey. Score: {nps}/10. Feedback: 'The selection of clothes is great, but the delivery took much longer than expected.'"},
    {"type": "app_feedback", "summary": "Customer submitted feedback via mobile app: 'The checkout screen keeps freezing when I try to apply my member points.' Rating: 2 stars."},
    {"type": "twitter_dm", "summary": "Follow-up DM regarding package refund. Customer stated: 'Tracking shows delivered but I never received my order. Please refund me.' Initiated replacement shipment."},
    {"type": "phone_support", "summary": "Customer called to ask about size exchange policy for {product}. Agent sent return label and processed exchange for next size up."},
]

CUSTOMER_PROFILES = [
    {"id": "CUST-1001", "name": "Sarah Chen", "segment": "Platinum Elite", "ltv": 45840.00, "tenure_months": 72},
    {"id": "CUST-1002", "name": "Marcus Johnson", "segment": "Silver", "ltv": 2890.50, "tenure_months": 14},
    {"id": "CUST-1003", "name": "Priya Patel", "segment": "Gold", "ltv": 14120.00, "tenure_months": 36},
    {"id": "CUST-1004", "name": "James O'Brien", "segment": "Basic", "ltv": 429.99, "tenure_months": 2},
    {"id": "CUST-1005", "name": "Aisha Williams", "segment": "Platinum Elite", "ltv": 31560.00, "tenure_months": 48},
]


def generate_interactions(customer_id: str = "CUST-1001", count: int = 12) -> list[dict]:
    """Generate realistic interaction history for a customer."""
    interactions = []
    for i in range(count):
        template = INTERACTION_TEMPLATES[i % len(INTERACTION_TEMPLATES)]
        order_num = random.randint(100000, 999999)
        product = random.choice(PRODUCT_NAMES)
        amount = round(random.uniform(25, 350), 2)
        nps = random.choice([6, 7, 8, 8, 9, 9, 10])
        summary = template["summary"].format(
            order=order_num, product=product, amount=amount, nps=nps
        )
        interactions.append({
            "interaction_id": f"INT-{fake.unique.random_int(min=100000, max=999999)}",
            "customer_id": customer_id,
            "type": template["type"],
            "summary": summary,
            "date": (datetime.now() - timedelta(days=count - i) * 7).isoformat(),
            "agent": fake.name() if template["type"] in ("phone_support", "twitter_dm") else None,
            "satisfaction_score": random.choice([None, 3, 4, 4, 5, 5, 5]),
        })
    return interactions



def get_customer_profiles() -> list[dict]:
    return CUSTOMER_PROFILES


# ---------------------------------------------------------------------------
# Demo 4 – Graph Explorer: product graph nodes/edges
# ---------------------------------------------------------------------------

GRAPH_NODES = {
    "products": [
        {"id": "p1", "name": "Organic Bananas", "category": "Fresh Fruits", "price": 2.99, "aisle": "Produce"},
        {"id": "p2", "name": "Almond Milk", "category": "Plant-Based", "price": 4.49, "aisle": "Dairy Alternatives"},
        {"id": "p3", "name": "Gluten-Free Bread", "category": "Bakery", "price": 6.99, "aisle": "Bread & Bakery"},
        {"id": "p4", "name": "Organic Avocados", "category": "Fresh Fruits", "price": 5.99, "aisle": "Produce"},
        {"id": "p5", "name": "Free-Range Eggs", "category": "Dairy", "price": 5.49, "aisle": "Dairy & Eggs"},
        {"id": "p6", "name": "Organic Baby Spinach", "category": "Fresh Vegetables", "price": 3.99, "aisle": "Produce"},
        {"id": "p7", "name": "Greek Yogurt", "category": "Dairy", "price": 1.99, "aisle": "Dairy & Eggs"},
        {"id": "p8", "name": "Cold Brew Coffee", "category": "Beverages", "price": 4.99, "aisle": "Coffee & Tea"},
        {"id": "p9", "name": "Kombucha", "category": "Beverages", "price": 3.49, "aisle": "Refrigerated Drinks"},
        {"id": "p10", "name": "Hummus", "category": "Deli", "price": 4.29, "aisle": "Dips & Spreads"},
    ],
    "categories": ["Fresh Fruits", "Plant-Based", "Bakery", "Dairy", "Fresh Vegetables", "Beverages", "Deli"],
    "aisles": ["Produce", "Dairy Alternatives", "Bread & Bakery", "Dairy & Eggs", "Coffee & Tea", "Refrigerated Drinks", "Dips & Spreads"],
}

GRAPH_EDGES = [
    {"source": "p1", "target": "p2", "relation": "FREQUENTLY_BOUGHT_WITH", "confidence": 0.82},
    {"source": "p1", "target": "p4", "relation": "FREQUENTLY_BOUGHT_WITH", "confidence": 0.75},
    {"source": "p6", "target": "p4", "relation": "OFTEN_IN_SAME_RECIPE", "confidence": 0.88},
    {"source": "p5", "target": "p3", "relation": "FREQUENTLY_BOUGHT_WITH", "confidence": 0.71},
    {"source": "p8", "target": "p2", "relation": "PAIRS_WELL_WITH", "confidence": 0.91},
    {"source": "p7", "target": "p1", "relation": "FREQUENTLY_BOUGHT_WITH", "confidence": 0.68},
    {"source": "p9", "target": "p10", "relation": "WEEKEND_SNACK_BASKET", "confidence": 0.85},
    {"source": "p3", "target": "p4", "relation": "OFTEN_IN_SAME_RECIPE", "confidence": 0.79},
    {"source": "p2", "target": "p3", "relation": "HEALTH_CONSCIOUS_BASKET", "confidence": 0.93},
    {"source": "p6", "target": "p10", "relation": "FREQUENTLY_BOUGHT_WITH", "confidence": 0.74},
]


def get_graph_data() -> dict:
    return {"nodes": GRAPH_NODES, "edges": GRAPH_EDGES}
