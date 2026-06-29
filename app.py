from pathlib import Path
import json
import streamlit as st

from detection.detect import detect_products
from recommendation.hybrid import recommend_basket

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="VisionCart",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 VisionCart")
st.subheader("AI Grocery Recommendation System")

uploaded_file = st.file_uploader(
    "Upload a grocery image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image_path = ROOT / "temp_image.jpg"

    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(image_path, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Detecting products..."):

        basket = detect_products(str(image_path))

    st.success("Detection Complete!")

    st.markdown("## Detected Products")

    for item in basket:
        st.write(f"• {item}")

    with st.spinner("Generating Recommendations..."):

        recommendations = recommend_basket(basket)

    st.markdown("---")
    st.header("Top Recommendations")

    for category, recs in recommendations.items():

        st.subheader(category.capitalize())

        if not recs:
         st.warning(f"No recommendations for {category}")
         continue

        for r in recs:
         st.write(
        f"• {r['product_name']} ({r['score']:.3f})"
    )

        

    output = {

        "basket": basket,

        "recommendations": recommendations

    }

    with open(
        ROOT / "outputs" / "hybrid_recommendations.json",
        "w"
    ) as f:

        json.dump(output, f, indent=4)

    st.success("Recommendations saved successfully!")