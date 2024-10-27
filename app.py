from flask import Flask, request, render_template, url_for, redirect
import numpy as np
import tensorflow as tf
import cv2
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import joblib
import json
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from twilio.rest import Client

app = Flask(__name__)

# new comment

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def encode_input_data(input_data):
    # Convert input_data to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical features
    for column in input_df.columns:
        if column in encoding_mappings:
            le = LabelEncoder()
            le.classes_ = np.array(list(encoding_mappings[column].keys()))
            input_df[column] = le.transform(input_df[column])
    
    return input_df

def predict_sustainability_index(input_data):
    # Encode the input data
    encoded_input = encode_input_data(input_data)
    
    # Make predictions using the model
    prediction = rf_model.predict(encoded_input)
    return prediction[0]

# Load the encoding mappings
with open('static/assets/encoding_mappings.json', 'r') as f:
    encoding_mappings = json.load(f)

# Load the trained Random Forest model
rf_model = joblib.load('models/sustainnability_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/challenge')
def challenge():
    return render_template('challenge.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the POST request has a file part
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    
    # If user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return "No selected file", 400

    # Save the file locally
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
    

    # Load and preprocess the image
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)  # Preprocess

    keras_model = tf.keras.models.load_model("models/flower_model_two.keras")
    predictions = keras_model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    os.remove(file_path)

    # Map the predicted class index to the class name
    cl_nm=['Astilbe', 'Bellflower', 'Black Eyed Susan', 'Calendula', 'California Poppy', 'Carnation', 'Common Daisy', 'Coreopsis', 'Dandelion', 'Iris', 'Rose', 'Sunflower', 'Tulip', 'Water Lily']
    flowers_info = {
    "Astilbe": [
        "Astilbe, known for its feathery plumes and vibrant colors, is a calming garden flower that encourages mindfulness and tranquility. Often used in shaded, meditative garden spaces, Astilbe’s softness and delicate nature promote a sense of groundedness, ideal for moments of introspection.",
        "However, climate change poses a challenge for Astilbe, as it thrives in cooler, moist environments. Warmer temperatures and unpredictable rain patterns can disrupt its blooming cycle and affect its lush growth, as prolonged dry spells and heatwaves become more frequent."
    ],
    "Bellflower": [
        "Bellflowers, with their delicate, bell-shaped blossoms, symbolize humility and constancy. Their gentle blue and purple tones are often associated with peace and healing, making them a grounding addition to spaces focused on mental well-being and spiritual growth.",
        "Bellflowers are vulnerable to climate change due to their preference for cool, moist soils. Increased temperatures and irregular rainfall stress the plants, leading to reduced flowering and, in severe cases, loss of entire populations in warmer regions."
    ],
    "Black Eyed Susan": [
        "Black-Eyed Susan, with its radiant yellow petals and dark center, embodies resilience and positive energy, often used as a symbol of encouragement and growth. This resilient flower serves as a reminder of joy and optimism, grounding individuals in moments of reflection and strength.",
        "Yet, climate change affects Black-Eyed Susan’s natural resilience, as rising temperatures and droughts can limit its blooming and growth. Although it adapts well, prolonged heat stress and water scarcity can reduce its vigor and presence in certain regions."
    ],
    "Calendula": [
        "Calendula, known as the 'flower of the sun,' is associated with warmth and healing, often used for its soothing properties in skincare and herbal remedies. Its vibrant yellow-orange blooms offer uplifting energy, promoting mental clarity and calm.",
        "Global warming, however, affects Calendula by altering its growth cycle. Higher temperatures may cause early blooming, shortening its lifecycle, while increased droughts may reduce its ability to flourish in traditional climates, impacting both wild and cultivated populations."
    ],
    "California Poppy": [
        "The California Poppy, a symbol of peace and rejuvenation, offers a radiant, orange-gold color that has a calming, grounding effect. It encourages introspection and mindfulness, with its presence in gardens and natural landscapes uplifting mental well-being.",
        "California Poppies are highly sensitive to climate shifts. As droughts and heatwaves intensify, the poppy’s native habitats in California face increased stress, often leading to shorter bloom seasons and reduced seed production."
    ],
    "Carnation": [
        "Carnations are cherished for their sweet fragrance and symbolic meanings of love and remembrance, often given as tokens of appreciation and gratitude. Their frilled petals and bright colors inspire positivity and emotional well-being.",
        "However, carnations are heavily impacted by warming temperatures and changes in rainfall patterns. Global warming can disrupt their flowering season, and they may struggle to thrive in regions experiencing extreme temperatures or irregular water availability."
    ],
    "Common Daisy": [
        "The Common Daisy, a symbol of innocence and purity, inspires joy and simplicity. Its bright white petals and sunny yellow center create a sense of ease and grounding, perfect for connecting with nature and finding balance in daily life.",
        "Climate change puts the Common Daisy at risk, as increasing temperatures and reduced soil moisture hinder its growth. Daisies thrive in cool, moist environments, and warmer, drier conditions could make it difficult for them to establish in some areas."
    ],
    "Coreopsis": [
        "Coreopsis, with its sunny yellow blooms, represents cheerfulness and resilience, making it a grounding and positive presence in gardens. Known as 'tickseed,' it can create a calm, meditative ambiance, ideal for spaces dedicated to mindfulness.",
        "As climate conditions shift, Coreopsis may experience stunted growth or reduced flowering due to increased droughts and warmer temperatures. Extended dry spells can weaken its resilience, affecting both wild and cultivated populations."
    ],
    "Dandelion": [
        "Dandelions are resilient flowers symbolizing hope and transformation, often seen as a reminder of change and growth. Their bright yellow blooms are grounding and can inspire mental clarity and resilience through times of challenge.",
        "Rising temperatures and shifts in seasonal patterns impact dandelions, particularly in regions with extreme drought. While dandelions are highly adaptive, intensified climate shifts may affect their seeding and growth cycle."
    ],
    "Iris": [
        "The Iris, with its royal colors and intricate petal structure, symbolizes wisdom and hope. Its beauty encourages reflection and inner peace, often linked to spirituality and meditation practices, making it a meaningful flower for mindfulness.",
        "Global warming, however, brings challenges for Irises, as they require stable, moderate climates. Rising temperatures and erratic rainfall patterns can lead to root rot or reduced blooming, potentially impacting their survival in some habitats."
    ],
    "Rose": [
        "Roses, with their timeless elegance and fragrance, are symbols of love and beauty. Their captivating blooms and rich scents invite individuals to pause, reflect, and connect with deeper emotions, promoting a sense of mental clarity and grounding.",
        "Climate change threatens roses with increased droughts and pests that thrive in warmer temperatures. Irregular rainfall and unpredictable frosts can weaken the rose’s growth, affecting its health, flowering, and fragrance."
    ],
    "Sunflower": [
        "Sunflowers, known for their bold yellow petals and tall stature, symbolize positivity and loyalty. Their growth pattern, following the sun, is a reminder of hope and resilience, fostering mental well-being and optimism.",
        "Sunflowers, though resilient, are affected by global warming. Rising temperatures and drought can lead to shorter growth cycles and reduced seed production, threatening their cultivation in some areas."
    ],
    "Tulip": [
        "Tulips, with their graceful, cup-shaped blooms, represent renewal and serenity. Their vibrant colors are grounding, encouraging mindfulness and presence, especially during early spring’s period of rejuvenation.",
        "Climate change disrupts the tulip’s growth cycle, as warming temperatures cause premature blooms. Earlier flowering can lead to shorter blooming periods, especially in regions where winters are less severe, affecting both wild and cultivated populations."
    ],
    "Water Lily": [
        "Water lilies, floating serenely on ponds, symbolize peace, purity, and spiritual awakening. Their presence in water creates a calming effect, encouraging reflection and grounding, ideal for meditative spaces and connecting with nature.",
        "Water lilies are sensitive to climate change due to their dependence on stable water temperatures and levels. Rising temperatures and increased evaporation rates can threaten their habitat, impacting their growth and survival in some regions."
    ]
}

    class_name = cl_nm[predicted_class[0]]

    return render_template('uploadedImage.html', flower_name=class_name, flower_mindfulness=flowers_info[class_name][0], flower_climatechange=flowers_info[class_name][1])

@app.route('/sustainabilityscore')
def sustainabilityscore():
    return render_template('sustainabilityscoreinput.html')

@app.route('/yoursustainabilityscore', methods=['POST'])
def yoursustainabilityscore():
    # Access the form data
    age = request.form.get('age')
    location = request.form.get('location')
    diet_type = request.form.get('Diet-type')
    local_food = request.form.get('LocalFood')
    transport = request.form.get('Transport')
    energy_source = request.form.get('EnergySource')
    housing = request.form.get('Housing')
    clothing = request.form.get('Clothing')
    sustainability = request.form.get('Sustainability')
    comm_involv = request.form.get('CommInvolv')
    gender = request.form.get('Gender')
    plastic = request.form.get('Plastic')
    disposal = request.form.get('Disposal')
    phys_act = request.form.get('PhysAct')
    env_aware = request.form.get('EnvAware')


    # Create the input data dictionary
    input_data = {
        'Age': age,                 # Assuming 'Age' is the column name in your dataset
        'Location': location,       # Assuming 'Location' is the column name
        'DietType': diet_type,      # Adjust according to the naming in your dataset
        'LocalFoodFrequency': local_food,
        'TransportationMode': transport,
        'EnergySource': energy_source,
        'HomeType': housing,
        'ClothingFrequency': clothing,
        'SustainableBrands': sustainability,
        'EnvironmentalAwareness': env_aware,
        'CommunityInvolvement': comm_involv,
        'Gender': gender,
        'UsingPlasticProducts': plastic,
        'DisposalMethods': disposal,
        'PhysicalActivities': phys_act
    }

    predicted_index = predict_sustainability_index(input_data)
    # Render a result template or redirect
    return render_template('calculatedsustainabilityscore.html', predicted_sustainability_index = predicted_index)

if __name__ == '__main__':
    app.run(debug=True)
