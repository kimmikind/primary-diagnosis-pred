from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pydantic import BaseModel
from xgboost import XGBClassifier
import xgboost as xgb

app = Flask(__name__)

# Модель для анализа симптомов
try:
    with open('model/xgb.pkl', 'rb') as f:
        symptoms_model = pickle.load(f)
except Exception as e:
    print(f"Error loading symptoms model: {e}")
    raise

try:
    # Модель для анализа крови
    with open('model/xgb_blood.pkl', 'rb') as f:
        blood_model = pickle.load(f)
    
    # Scaler для модели крови
    with open('model/blood_scaler.pkl', 'rb') as f:
        blood_scaler = pickle.load(f)
    
    # Загрузка колонок (можно сохранить и загрузить отдельно)
    blood_features = [
        'Glucose', 'Cholesterol', 'Hemoglobin', 'Platelets', 
        'White Blood Cells', 'Red Blood Cells', 'Hematocrit',
        'Mean Corpuscular Volume', 'Mean Corpuscular Hemoglobin',
        'Mean Corpuscular Hemoglobin Concentration', 'Insulin',
        'BMI', 'Systolic Blood Pressure', 'Diastolic Blood Pressure',
        'Triglycerides', 'HbA1c', 'LDL Cholesterol', 'HDL Cholesterol',
        'ALT', 'AST', 'Heart Rate', 'Creatinine', 'Troponin',
        'C-reactive Protein'
    ]
except Exception as e:
    print(f"Error loading blood models: {e}")
    raise
    
# Загрузка датасетов
def load_datasets():
    try:
        description = pd.read_csv('datasets/description.csv')
        precautions = pd.read_csv('datasets/precautions_df.csv')
        workout = pd.read_csv('datasets/workout_df.csv')
        medications = pd.read_csv('datasets/medications.csv')
        diets = pd.read_csv('datasets/diets.csv')
        symp_severity = pd.read_csv('datasets/Symptom-severity.csv')
        
        return description, precautions, workout, medications, diets, symp_severity
    except Exception as e:
        raise Exception(f"Error loading datasets: {str(e)}")

# Загружаем датасеты при старте сервера
try:
    description, precautions, workout, medications, diets, symp_severity = load_datasets()
except Exception as e:
    print(f"Failed to load datasets: {e}")
    raise

# Словарь симптомов
symptoms_dict = {
    'itching': 0,
    'skin rash': 1,
    'nodal skin eruptions': 2,
    'continuous sneezing': 3,
    'shivering': 4,
    'chills': 5,
    'joint pain': 6,
    'stomach pain': 7,
    'acidity': 8,
    'ulcers on tongue': 9,
    'muscle wasting': 10,
    'vomiting': 11,
    'burning micturition': 12,
    'spotting urination': 13,
    'fatigue': 14,
    'weight gain': 15,
    'anxiety': 16,
    'cold hands and feets': 17,
    'mood swings': 18,
    'weight loss': 19,
    'restlessness': 20,
    'lethargy': 21,
    'patches in throat': 22,
    'irregular sugar level': 23,
    'cough': 24,
    'high fever': 25,
    'sunken eyes': 26,
    'breathlessness': 27,
    'sweating': 28,
    'dehydration': 29,
    'indigestion': 30,
    'headache': 31,
    'yellowish skin': 32,
    'dark urine': 33,
    'nausea': 34,
    'loss of appetite': 35,
    'pain behind the eyes': 36,
    'back pain': 37,
    'constipation': 38,
    'abdominal pain': 39,
    'diarrhoea': 40,
    'mild fever': 41,
    'yellow urine': 42,
    'yellowing of eyes': 43,
    'acute liver failure': 44,
    'fluid overload': 45,
    'swelling of stomach': 46,
    'swelled lymph nodes': 47,
    'malaise': 48,
    'blurred and distorted vision': 49,
    'phlegm': 50,
    'throat irritation': 51,
    'redness of eyes': 52,
    'sinus pressure': 53,
    'runny nose': 54,
    'congestion': 55,
    'chest pain': 56,
    'weakness in limbs': 57,
    'fast heart rate': 58,
    'pain during bowel movements': 59,
    'pain in anal region': 60,
    'bloody stool': 61,
    'irritation in anus': 62,
    'neck pain': 63,
    'dizziness': 64,
    'cramps': 65,
    'bruising': 66,
    'obesity': 67,
    'swollen legs': 68,
    'swollen blood vessels': 69,
    'puffy face and eyes': 70,
    'enlarged thyroid': 71,
    'brittle nails': 72,
    'swollen extremeties': 73,
    'excessive hunger': 74,
    'extra marital contacts': 75,
    'drying and tingling lips': 76,
    'slurred speech': 77,
    'knee pain': 78,
    'hip joint pain': 79,
    'muscle weakness': 80,
    'stiff neck': 81,
    'swelling joints': 82,
    'movement stiffness': 83,
    'spinning movements': 84,
    'loss of balance': 85,
    'unsteadiness': 86,
    'weakness of one body side': 87,
    'loss of smell': 88,
    'bladder discomfort': 89,
    'foul smell of urine': 90,
    'continuous feel of urine': 91,
    'passage of gases': 92,
    'internal itching': 93,
    'toxic look (typhos)': 94,
    'depression': 95,
    'irritability': 96,
    'muscle pain': 97,
    'altered sensorium': 98,
    'red spots over body': 99,
    'belly pain': 100,
    'abnormal menstruation': 101,
    'dischromic patches': 102,
    'watering from eyes': 103,
    'increased appetite': 104,
    'polyuria': 105,
    'family history': 106,
    'mucoid sputum': 107,
    'rusty sputum': 108,
    'lack of concentration': 109,
    'visual disturbances': 110,
    'receiving blood transfusion': 111,
    'receiving unsterile injections': 112,
    'coma': 113,
    'stomach bleeding': 114,
    'distention of abdomen': 115,
    'history of alcohol consumption': 116,
    'fluid overload.1': 117,
    'blood in sputum': 118,
    'prominent veins on calf': 119,
    'palpitations': 120,
    'painful walking': 121,
    'pus filled pimples': 122,
    'blackheads': 123,
    'scurring': 124,
    'skin peeling': 125,
    'silver like dusting': 126,
    'small dents in nails': 127,
    'inflammatory nails': 128,
    'blister': 129,
    'red sore around nose': 130,
    'yellow crust ooze': 131
}

#Словарь болезней
diseases_list = {
    0: '(vertigo) Paroymsal Positional Vertigo',
    1: 'AIDS',
    2: 'Acne',
    3: 'Alcoholic hepatitis',
    4: 'Allergy',
    5: 'Arthritis',
    6: 'Bronchial Asthma',
    7: 'Cervical spondylosis',
    8: 'Chicken pox',
    9: 'Chronic cholestasis',
    10: 'Common Cold',
    11: 'Dengue',
    12: 'Diabetes',
    13: 'Dimorphic hemmorhoids(piles)',
    14: 'Drug Reaction',
    15: 'Fungal infection',
    16: 'GERD',
    17: 'Gastroenteritis',
    18: 'Heart attack',
    19: 'Hepatitis B',
    20: 'Hepatitis C',
    21: 'Hepatitis D',
    22: 'Hepatitis E',
    23: 'Hypertension',
    24: 'Hyperthyroidism',
    25: 'Hypoglycemia',
    26: 'Hypothyroidism',
    27: 'Impetigo',
    28: 'Jaundice',
    29: 'Malaria',
    30: 'Migraine',
    31: 'Osteoarthristis',
    32: 'Paralysis (brain hemorrhage)',
    33: 'Peptic ulcer diseae',
    34: 'Pneumonia',
    35: 'Psoriasis',
    36: 'Tuberculosis',
    37: 'Typhoid',
    38: 'Urinary tract infection',
    39: 'Varicose veins',
    40: 'hepatitis A'
}


# Полная таблица нормальных диапазонов (значения от 0 до 1)
NORMAL_RANGES = {
    'Glucose': (0.4, 0.55),
    'Cholesterol': (0.35, 0.5),
    'Hemoglobin': (0.6, 0.75),
    'Platelets': (0.6, 0.8),
    'White Blood Cells': (0.5, 0.7),
    'Red Blood Cells': (0.55, 0.65),
    'Hematocrit': (0.45, 0.6),
    'Mean Corpuscular Volume': (0.5, 0.7),
    'Mean Corpuscular Hemoglobin': (0.5, 0.7),
    'Mean Corpuscular Hemoglobin Concentration': (0.5, 0.7),
    'Insulin': (0.3, 0.5),
    'BMI': (0.45, 0.55),
    'Systolic Blood Pressure': (0.5, 0.7),
    'Diastolic Blood Pressure': (0.5, 0.7),
    'Triglycerides': (0.3, 0.5),
    'HbA1c': (0.3, 0.5),
    'LDL Cholesterol': (0.3, 0.5),
    'HDL Cholesterol': (0.4, 0.6),
    'ALT': (0.3, 0.5),
    'AST': (0.3, 0.5),
    'Heart Rate': (0.5, 0.7),
    'Creatinine': (0.4, 0.6),
    'Troponin': (0.1, 0.3),
    'C-reactive Protein': (0.1, 0.3)
}

class SymptomsRequest(BaseModel):
    symptoms: str

def helper(dis: str) -> Tuple[str, List[str], List[str], List[str], List[str]]:
    """Функция для получения описания болезни, рекомендаций, лекарств, диеты и упражнений"""
    try:
        # Нормализация названия болезни
        dis = dis.strip().lower()
        
        # Описание болезни
        desc = description[description['Disease'].str.lower() == dis]['Description']
        desc = " ".join([str(w) for w in desc]) if not desc.empty else "No description available"
        
        # Меры предосторожности
        pre_df = precautions[precautions['Disease'].str.lower() == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
        pre = []
        if not pre_df.empty:
            pre = [str(p) for p in pre_df.values[0] if pd.notna(p)]
        
        # Безопасное извлечение списка лекарств (без eval)
        med = []
        med_df = medications[medications['Disease'].str.lower() == dis]['Medication']
        if not med_df.empty and pd.notna(med_df.values[0]):
            med_str = med_df.values[0].strip("[]").replace("'", "")
            med = [m.strip() for m in med_str.split(",")]
        
        # Безопасное извлечение списка диет
        die = []
        die_df = diets[diets['Disease'].str.lower() == dis]['Diet']
        if not die_df.empty and pd.notna(die_df.values[0]):
            die_str = die_df.values[0].strip("[]").replace("'", "")
            die = [d.strip() for d in die_str.split(",")]
        
        # Безопасное извлечение списка упражнений
        wrkout = []
        wrkout_df = workout[workout['disease'].str.lower() == dis]['workout']
        if not wrkout_df.empty and pd.notna(wrkout_df.values[0]):
            wrkout_str = wrkout_df.values[0].strip("[]").replace("'", "")
            wrkout = [w.strip() for w in wrkout_str.split(",")]
        
        return desc, pre, med, die, wrkout
    
    except Exception as e:
        print(f"Error in helper function for disease '{dis}': {e}")
        return f"Error loading data for {dis}", [], [], [], []

def given_predicted_value(patient_symptoms: List[str]) -> Tuple[str, str]:
    """Функция для предсказания болезни на основе симптомов"""
    try:
        input_vector = np.zeros(len(symptoms_dict))
        for item in patient_symptoms:
            item_clean = item.strip().lower()
            if item_clean in symptoms_dict:
                input_vector[symptoms_dict[item_clean]] = 1
        
        # Получаем предсказание и вероятности
        prediction =  symptoms_model.predict([input_vector])[0]
        probabilities =  symptoms_model.predict_proba([input_vector])[0]
        
        # Получаем вероятность для предсказанного класса
        prediction_prob = probabilities[prediction] * 100  # в процентах
        
        return diseases_list[prediction], f"{prediction_prob:.2f}%"
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Unknown", "0%"

def safe_predict(input_values: dict, min_confidence: float = 0.8) -> Tuple[str, float, List[str]]:
    """Предсказание на основе анализа крови"""
    try:
        # 1. Проверка минимального количества показателей
        if len(input_values) < 2:
            return "Недостаточно данных", 0.0, ["Введите минимум 2 показателя"]
        
        # 2. Проверка нормальных диапазонов
        normal_count = 0
        abnormal_features = set()
        for feat, val in input_values.items():
            if feat in NORMAL_RANGES:
                if NORMAL_RANGES[feat][0] <= val <= NORMAL_RANGES[feat][1]:
                    normal_count += 1
                else:
                    abnormal_features.add(feat)
        
        # 3. Специальные правила для явных случаев
        # Thalasse (низкий MCV + низкий Hb)
        if ('Mean Corpuscular Volume' in abnormal_features and input_values['Mean Corpuscular Volume'] <= 0.35) and \
           ('Hemoglobin' in abnormal_features and input_values['Hemoglobin'] <= 0.5):
            return "Thalasse", 0.95, ["Hemoglobin Electrophoresis", "Iron Studies", "Genetic Testing"]
        
        # Thromboc (низкие тромбоциты + воспаление)
        if ('Platelets' in abnormal_features and input_values['Platelets'] < 0.3) and \
           ('C-reactive Protein' in abnormal_features and input_values['C-reactive Protein'] > 0.4):
            return "Thromboc", 0.9, ["Coagulation Tests", "D-Dimer", "Hematologist Consultation"]
        
        # Анемия (низкий Hb или RBC или Hct)
        anemia_markers = {'Hemoglobin', 'Red Blood Cells', 'Hematocrit'}
        if any(m in abnormal_features for m in anemia_markers):
            if (('Hemoglobin' in abnormal_features and input_values['Hemoglobin'] < 0.45) or \
               ('Red Blood Cells' in abnormal_features and input_values['Red Blood Cells'] < 0.45) or \
               ('Hematocrit' in abnormal_features and input_values['Hematocrit'] < 0.4)):
                return "Anemia", 0.92, ["Ferritin", "Vitamin B12", "Hematologist Consultation"]
        
        # Heart Di (высокий тропонин + аномальный пульс)
        if ('Troponin' in abnormal_features and input_values['Troponin'] > 0.65) and \
           ('Heart Rate' in abnormal_features and (input_values['Heart Rate'] < 0.4 or input_values['Heart Rate'] > 0.85)):
            return "Heart Di", 0.93, ["EKG", "Cardiac Enzymes", "Cardiologist Consultation"]
        
        # Diabetes (высокий глюкоз или HbA1c)
        diabetes_markers = {'Glucose', 'HbA1c'}
        if any(m in abnormal_features for m in diabetes_markers):
            if (('Glucose' in abnormal_features and input_values['Glucose'] > 0.65) or \
               ('HbA1c' in abnormal_features and input_values['HbA1c'] > 0.55)):
                return "Diabetes", 0.94, ["Urine Analysis", "Triglycerides", "Endocrinologist Consultation"]
        
        # 4. Пограничные состояния
        # Пограничная анемия (требуем 2+ маркера)
        anemia_borderline = 0
        if 'Hemoglobin' in input_values and 0.5 <= input_values['Hemoglobin'] < 0.6:
            anemia_borderline += 1
        if 'Red Blood Cells' in input_values and 0.5 <= input_values['Red Blood Cells'] < 0.55:
            anemia_borderline += 1
        if 'Hematocrit' in input_values and 0.4 <= input_values['Hematocrit'] < 0.45:
            anemia_borderline += 1
            
        if anemia_borderline >= 2:
            return "Пограничное состояние (Anemia)", 0.7, ["Iron Tests", "Follow-up in 1 Month"]
        
        # Пограничный диабет (требуем оба показателя)
        diabetes_borderline = 0
        if 'Glucose' in input_values and 0.55 <= input_values['Glucose'] <= 0.65:
            diabetes_borderline += 1
        if 'HbA1c' in input_values and 0.5 <= input_values['HbA1c'] <= 0.55:
            diabetes_borderline += 1
            
        if diabetes_borderline >= 2:
            return "Пограничное состояние (Diabetes)", 0.75, ["Repeat Glucose Test", "Endocrinologist Consultation"]
        
        # Пограничное кардио
        cardio_markers = {'Cholesterol', 'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Troponin'}
        if any(m in abnormal_features for m in cardio_markers):
            if (('Cholesterol' in abnormal_features and input_values['Cholesterol'] > 0.52) or \
               ('Systolic Blood Pressure' in abnormal_features and input_values['Systolic Blood Pressure'] > 0.65) or \
               ('Troponin' in abnormal_features and 0.5 < input_values['Troponin'] <= 0.65)):
                return "Пограничное состояние (Cardio)", 0.7, ["Lipid Profile", "Cardiologist Consultation"]
        
        # 5. Если все показатели в норме
        if normal_count == len(input_values):
            return "Healthy", 0.96, ["Annual Checkup"]
        
        # 6. Подготовка данных для модели
        input_vector = np.zeros((1, len(blood_features)))
        for i, col in enumerate(blood_features):
            if col in input_values and blood_scaler.scale_[i] > 0:
                input_vector[0, i] = (input_values[col] - blood_scaler.mean_[i]) / blood_scaler.scale_[i]
        
        # 7. Предсказание моделью
        proba = blood_model.predict_proba(input_vector)[0]
        pred_class = np.argmax(proba)
        confidence = np.max(proba)
        
        diagnosis = {
            0: 'Healthy',
            1: 'Diabetes',
            2: 'Anemia',
            3: 'Thalasse',
            4: 'Thromboc',
            5: 'Heart Di'
        }[pred_class]
        
        # Пост-обработка результатов модели
        if diagnosis == "Anemia" and not any(m in input_values for m in anemia_markers):
            return "Неопределенный результат", confidence, ["Complete Blood Count"]
        
        if diagnosis == "Diabetes" and not any(m in input_values for m in diabetes_markers):
            return "Неопределенный результат", confidence, ["Glucose Tolerance Test"]
        
        if diagnosis == "Heart Di" and 'Troponin' not in input_values:
            return "Неопределенный результат", confidence, ["Cardiac Screening"]
        
        if diagnosis == "Thalasse" and 'Mean Corpuscular Volume' not in input_values:
            return "Неопределенный результат", confidence, ["Hemoglobin Analysis"]
        
        if diagnosis == "Thromboc" and 'Platelets' not in input_values:
            return "Неопределенный результат", confidence, ["Coagulation Study"]
        
        if confidence < min_confidence:
            return "Неопределенный результат", confidence, ["Additional Tests Required"]
        
        return diagnosis, confidence, get_recommendations(diagnosis, input_values.keys())
    
    except Exception as e:
        return f"Ошибка: {str(e)}", 0.0, []    
    
def get_recommendations(diagnosis: str, input_keys: List[str]) -> List[str]:
    """Генерирует рекомендации на основе диагноза и введенных параметров"""
    recommendations = {
        'Healthy': ["Annual Checkup"],
        'Diabetes': ["Urine Analysis", "Triglycerides", "Endocrinologist Consultation"],
        'Anemia': ["Ferritin", "Vitamin B12", "Hematologist Consultation"],
        'Thalasse': ["Hemoglobin Electrophoresis", "Iron Studies", "Genetic Testing"],
        'Thromboc': ["Coagulation Tests", "D-Dimer", "Hematologist Consultation"],
        'Heart Di': ["EKG", "Cardiac Enzymes", "Cardiologist Consultation"],
        'Пограничное состояние (Anemia)': ["Iron Tests", "Follow-up in 1 Month"],
        'Пограничное состояние (Diabetes)': ["Repeat Glucose Test", "Endocrinologist Consultation"],
        'Пограничное состояние (Cardio)': ["Lipid Profile", "Cardiologist Consultation"]
    }
    
    return recommendations.get(diagnosis, ["Consult with a specialist"])
    
# ================== API Endpoints ==================
@app.route('/predict_symptoms', methods=['POST'])
def predict_symptoms():
    """Endpoint для предсказания по симптомам"""
    try:
        data = request.get_json()
        if 'symptoms' not in data:
            return jsonify({"error": "Missing 'symptoms' field"}), 400
            
        user_symptoms = [s.strip() for s in data['symptoms'].split(',')]
        user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
        
        predicted_disease, probability = given_predicted_value(user_symptoms)
        desc, pre, med, die, wrkout = helper(predicted_disease)
        
        response = {
            "predicted_disease": predicted_disease,
            "probability": probability,
            "description": desc,
            "precautions": pre,
            "medications": med,
            "diet": die,
            "workout": wrkout
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_blood', methods=['POST'])
def predict_blood():
    """Endpoint для предсказания по анализу крови"""
    try:
        data = request.get_json()
        if 'blood_values' not in data:
            return jsonify({"error": "Missing 'blood_values' field"}), 400
            
        input_values = data['blood_values']
        
        # Проверка что все значения числовые
        for key, value in input_values.items():
            try:
                input_values[key] = float(value)
            except ValueError:
                return jsonify({"error": f"Invalid value for {key}: must be numeric"}), 400
        
        diagnosis, confidence, recommendations = safe_predict(input_values)
        
        response = {
            "diagnosis": diagnosis,
            "confidence": float(confidence),  # преобразуем numpy.float в обычный float
            "recommendations": recommendations
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint для проверки работоспособности сервера"""
    return jsonify({"status": "healthy", "models_loaded": True})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
