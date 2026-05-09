
# کالری‌شمار هوشمند  

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

app = Flask(__name__) #ساخت برنامه وب 
CORS(app) #اجازه ارتباط با دامنه یا پورت دیگر


# بارگذاری دیتاست از فایل 

def load_foods_from_csv():
    foods = {}
    
    default_foods = {
        'سیب': {'calories': 95, 'unit': 'عدد', 'carbs': 25, 'protein': 0.5, 'fat': 0.3},
        'موز': {'calories': 105, 'unit': 'عدد', 'carbs': 27, 'protein': 1.3, 'fat': 0.4},
        'پرتقال': {'calories': 62, 'unit': 'عدد', 'carbs': 15.4, 'protein': 0.9, 'fat': 0.1},
        'کباب کوبیده': {'calories': 320, 'unit': 'سیخ', 'carbs': 0, 'protein': 31, 'fat': 21},
        'جوجه کباب': {'calories': 380, 'unit': 'سیخ', 'carbs': 0, 'protein': 55, 'fat': 18},
        'قرمه سبزی': {'calories': 450, 'unit': 'پرس', 'carbs': 25, 'protein': 30, 'fat': 25},
        'عدس پلو': {'calories': 290, 'unit': 'کفگیر', 'carbs': 55, 'protein': 10, 'fat': 5},
        'ماست': {'calories': 120, 'unit': 'لیوان', 'carbs': 5, 'protein': 4, 'fat': 3},
        'تخم مرغ آبپز': {'calories': 78, 'unit': 'عدد', 'carbs': 0.6, 'protein': 6.3, 'fat': 5.3},
        'نان بربری': {'calories': 440, 'unit': 'عدد', 'carbs': 90, 'protein': 16, 'fat': 4},
        'گوجه': {'calories': 22, 'unit': 'عدد', 'carbs': 5, 'protein': 1, 'fat': 0.2},
    }
    
    try:
        if os.path.exists('foods.csv'):
            print(" در حال خواندن فایل foods.csv...")
            df = pd.read_csv('foods.csv', encoding='utf-8')
            
            # پیدا کردن ستون نام غذا
            name_col = df.columns[0]
            
            # پیدا کردن ستون کالری
            cal_col = None
            for col in df.columns:
                if 'کالری' in col or 'calorie' in col.lower():
                    cal_col = col
                    break
            if cal_col is None:
                print('کالری وجود ندارد')
            
            
            # پیدا کردن ستون واحد
            
            unit_col = None
            for col in df.columns:
                if 'واحد' in col or 'unit' in col.lower():
                    unit_col = col
                    break
            
            
            # پیدا کردن ستون کربوهیدرات
            
            carbs_col = None
            for col in df.columns:
                if 'کربوهیدرات' in col or 'carb' in col.lower():
                    carbs_col = col
                    break
            
            # پیدا کردن ستون پروتئین
            
            protein_col = None
            for col in df.columns:
                if 'پروتئین' in col or 'protein' in col.lower():
                    protein_col = col
                    break
            
            
            # پیدا کردن ستون چربی
            
            fat_col = None
            for col in df.columns:
                if 'چربی' in col or 'fat' in col.lower():
                    fat_col = col
                    break
            
            # خواندن داده‌ها
            
            for _, row in df.iterrows():
                name = str(row[name_col]).strip()
                if pd.isna(row[cal_col]):
                    continue
                
                # خواندن واحد
                unit = 'عدد'
                if unit_col and unit_col in df.columns and pd.notna(row[unit_col]):
                    unit = str(row[unit_col])
                
                #  خواندن کربوهیدرات (اگر ستون وجود داشت)
                carbs = 0.0
                if carbs_col and carbs_col in df.columns and pd.notna(row[carbs_col]):
                    try:
                        carbs = float(row[carbs_col])
                    except:
                        carbs = 0.0
                
                #  خواندن پروتئین (اگر ستون وجود داشت)
                protein = 0.0
                if protein_col and protein_col in df.columns and pd.notna(row[protein_col]):
                    try:
                        protein = float(row[protein_col])
                    except:
                        protein = 0.0
                
                #  خواندن چربی (اگر ستون وجود داشت)
                fat = 0.0
                if fat_col and fat_col in df.columns and pd.notna(row[fat_col]):
                    try:
                        fat = float(row[fat_col])
                    except:
                        fat = 0.0
                
                foods[name] = {
                    'calories': float(row[cal_col]),
                    'unit': unit,
                    'carbs': carbs,     
                    'protein': protein,  
                    'fat': fat           
                }
            
            if len(foods) > 0:
                print(f" {len(foods)} غذا از فایل CSV بارگذاری شد")
            else:
                foods = default_foods.copy()
                print(f" {len(foods)} غذا از دیتاست پیش‌فرض بارگذاری شد")
        else:
            foods = default_foods.copy()
            print(f" {len(foods)} غذا از دیتاست پیش‌فرض بارگذاری شد")
    except Exception as e:
        print(f" خطا: {e}")
        foods = default_foods.copy()
        print(f" {len(foods)} غذا از دیتاست پیش‌فرض بارگذاری شد")
    
    return foods

foods = load_foods_from_csv()
food_list = list(foods.keys())
print(f" غذاهای بارگذاری شده: {food_list[:10]}...")
# مترادف‌ها
aliases = {
    'کوبیده': 'کباب کوبیده',
    'جوجه': 'جوجه کباب',
    'قرمه': 'قرمه سبزی',
    'عدس': 'عدس پلو',
}

# کلمات توقف
STOP_WORDS = {'من', 'تو', 'ما', 'شما', 'آنها', 'خوردم', 'خورد', 'خوردی', 
                            'خوردید', 'خوردند', 'میخورم', 'میخوری', 'میخورد', 'میخوریم',
                            'یک', 'دو', 'سه', 'چهار', 'پنج', 'این', 'آن', 'امروز', 
                            'دیروز', 'الان', 'بعد', 'قبل', 'صبح', 'ظهر', 'شب'           
                            }

history = []


# تابع جستجوی فازی

def get_suggestions(sentence, limit=5):
    if not sentence:
        return []
    
    sentence = sentence.strip()
    candidates = []
    
    if sentence in foods:
        return [{'name': sentence, 'score': 100, 'calories': foods[sentence]['calories'], 'unit': foods[sentence]['unit']}]
    
    if sentence in aliases:
        exact_match = aliases[sentence]
        return [{'name': exact_match, 'score': 100, 'calories': foods[exact_match]['calories'], 'unit': foods[exact_match]['unit']}]
    
    words = sentence.split()
    for word in words:
        if word in aliases:
            exact_match = aliases[word]
            return [{'name': exact_match, 'score': 100, 'calories': foods[exact_match]['calories'], 'unit': foods[exact_match]['unit']}]
        if word in foods:
            return [{'name': word, 'score': 100, 'calories': foods[word]['calories'], 'unit': foods[word]['unit']}]
    
    keywords = [w for w in words if w not in STOP_WORDS and len(w) >= 2]
    if not keywords:
        keywords = [w for w in words if len(w) >= 2]    
    #جستجوی تیکه تیکه جمله با غذا
    for word in keywords:
        for food in foods.keys():
            try:
                score1 = fuzz.ratio(word, food)
                score2 = fuzz.partial_ratio(word, food)
                score = (score1 * 0.5) + (score2 * 0.5)
                penalty = len(food) * 0.3
                final_score = max(0, score - penalty)
                
                if word in food:
                    final_score += 15
                
                candidates.append({
                    'name': food,
                    'score': round(final_score, 1),
                    'calories': foods[food]['calories'],
                    'unit': foods[food]['unit']
                })
            except:
                continue
    #جستجوی کامل جمله وارد شده با غذا
    for food in foods.keys():
        try:
            score1 = fuzz.ratio(sentence, food)
            score2 = fuzz.partial_ratio(sentence, food)
            score = (score1 * 0.5) + (score2 * 0.5)
            penalty = len(food) * 0.2
            final_score = max(0, score - penalty)
            
            candidates.append({
                'name': food,
                'score': round(final_score, 1),
                'calories': foods[food]['calories'],
                'unit': foods[food]['unit']
            })
        except:
            continue
    
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c['name'] not in seen:
                seen.add(c['name'])
                unique_candidates.append(c)
        
        unique_candidates.sort(key=lambda x: x['score'], reverse=True)
        filtered = [c for c in unique_candidates if c['score'] >= 40]
        
        if not filtered:
            filtered = unique_candidates[:limit]
        
        return filtered[:limit]


# مدل هوش مصنوعی (Random Forest)

class AIModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.load()
    
    def train(self):
        print(" در حال آموزش هوش مصنوعی...")
        
        data = []
        for name, info in foods.items():
            data.append({
                'carbs': info.get('carbs', 0),
                'protein': info.get('protein', 0),
                'fat': info.get('fat', 0),
                'calories': info['calories']
            })
        
        df = pd.DataFrame(data)
        X = df[['carbs', 'protein', 'fat']].fillna(0)
        y = df['calories'].fillna(0)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=1)
        self.model.fit(X_train_scaled, y_train)
        
        score = self.model.score(X_test_scaled, y_test)
        self.is_trained = True
        
        joblib.dump(self.model, 'ai_model.pkl')
        joblib.dump(self.scaler, 'ai_scaler.pkl')
        
        print(f" هوش مصنوعی آموزش دید! دقت: {score*100:.1f}%")
        return True
    
    def load(self):
        try:
            if os.path.exists('ai_model.pkl'):
                self.model = joblib.load('ai_model.pkl')
                self.scaler = joblib.load('ai_scaler.pkl')
                self.is_trained = True
                print(" مدل هوش مصنوعی بارگذاری شد")
                return True
        except:
            pass
        return False
    
    def predict(self, carbs, protein, fat):
        if not self.is_trained or self.model is None:
            return None
        features = np.array([[carbs, protein, fat]])
        features_scaled = self.scaler.transform(features)
        return round(self.model.predict(features_scaled)[0], 2)

ai_model = AIModel()


# کلاس مدیریت تاریخچه و هدف

class CalorieTracker:
    def __init__(self):
        self.load_history()
        self.daily_goal = 2000
        self.load_goal()
    
    def load_history(self):
        global history
        try:
            with open('history.json', 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            history = []
    
    def save_history(self):
        with open('history.json', 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def load_goal(self):
        try:
            with open('goal.json', 'r', encoding='utf-8') as f:
                self.daily_goal = json.load(f).get('goal', 2000)
        except:
            self.daily_goal = 2000
    
    def save_goal(self):
        with open('goal.json', 'w', encoding='utf-8') as f:
            json.dump({'goal': self.daily_goal}, f)
    
    def set_goal(self, new_goal):
        if new_goal > 0:
            self.daily_goal = new_goal
            self.save_goal()
            return {'success': True, 'goal': self.daily_goal}
        return {'success': False}
    
    def add_meal(self, food_name, quantity):
        if food_name in foods:
            calories = foods[food_name]['calories'] * quantity
            record = {
                'food': food_name,
                'quantity': quantity,
                'unit': foods[food_name]['unit'],
                'calories': calories
            }
            history.append(record)
            self.save_history()
            return {'success': True, 'calories': calories}
        return {'success': False}
    
    def add_predicted_food(self, food_name, calories, unit, carbs, protein, fat):
        global foods, food_list
        
        if food_name in foods:
            return {'success': False, 'error': 'این غذا قبلاً در دیتابیس وجود دارد'}
        
        foods[food_name] = {
            'calories': calories,
            'unit': unit,
            'carbs': carbs,
            'protein': protein,
            'fat': fat
        }
        food_list = list(foods.keys())
        
        try:
            df = pd.DataFrame([{
                'نام غذا': food_name,
                'واحد': unit,
                'کالری': calories,
                'کربوهیدرات': carbs,
                'پروتئین': protein,
                'چربی': fat
            }])
            if os.path.exists('foods.csv'):
                existing = pd.read_csv('foods.csv', encoding='utf-8')
                df = pd.concat([existing, df], ignore_index=True)
            df.to_csv('foods.csv', index=False, encoding='utf-8-sig')
        except:
            pass
        
        return {'success': True, 'message': f'غذای {food_name} با موفقیت اضافه شد', 'calories': calories}
    
    def get_today_summary(self):
        today = datetime.now().strftime('%Y-%m-%d')
        today_meals = [m for m in history if m['date'] == today]
        total = sum(m['calories'] for m in today_meals)
        return {
            'meals': today_meals,
            'total': total,
            'goal': self.daily_goal
        }
    
    def get_all_foods(self):
        return [{'name': n, 'calories': f['calories'], 'unit': f['unit']} for n, f in foods.items()]
    
    def get_weekly_summary(self):
        weekly = {}
        days = ['شنبه', 'یکشنبه', 'دوشنبه', 'سه‌شنبه', 'چهارشنبه', 'پنجشنبه', 'جمعه']
        
        today = datetime.now()
        days_since_saturday = (today.weekday() + 2) % 7
        start_of_week = today - timedelta(days=days_since_saturday)
        
        for i in range(7):
            date = (start_of_week + timedelta(days=i)).strftime('%Y-%m-%d')
            day_meals = [m for m in history if m['date'] == date]
            total = sum(m['calories'] for m in day_meals)
            weekly[days[i]] = total
        
        avg = sum(weekly.values()) / 7 if weekly else 0
        
        return {
            'weekly': weekly,
            'average': round(avg, 1),
            'goal': self.daily_goal
        }
    
    def clear_history(self):
        global history
        history = []
        self.save_history()
        return {'success': True}
    
    def delete_meal(self, index):
        global history
        if 0 <= index < len(history):
            deleted = history.pop(index)
            self.save_history()
            return {'success': True, 'deleted': deleted}
        return {'success': False, 'error': 'آیتم یافت نشد'}
    
    def train_ai(self):
        return ai_model.train()

tracker = CalorieTracker()


# مسیرهای API

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/foods', methods=['GET'])
def api_foods():
    """لیست همه غذاها"""
    foods_list = tracker.get_all_foods()
    print(f" درخواست لیست غذاها: {len(foods_list)} غذا")
    return jsonify({'foods': foods_list})

@app.route('/api/search-suggestions', methods=['POST'])
def api_search_suggestions():
    data = request.get_json()
    query = data.get('query', '')
    print(f" جستجو برای: '{query}'")
    
    suggestions = get_suggestions(query, limit=5)
    
    exact_match = None
    if query in foods:
        exact_match = {'name': query, 'calories': foods[query]['calories'], 'unit': foods[query]['unit']}
    elif query in aliases:
        name = aliases[query]
        if name in foods:
            exact_match = {'name': name, 'calories': foods[name]['calories'], 'unit': foods[name]['unit']}
    
    return jsonify({'suggestions': suggestions, 'exact_match': exact_match})

@app.route('/api/add', methods=['POST'])
def api_add():
    data = request.get_json()
    food_name = data.get('name')
    quantity = float(data.get('quantity', 1))
    result = tracker.add_meal(food_name, quantity)
    return jsonify(result)

@app.route('/api/today', methods=['GET'])
def api_today():
    return jsonify(tracker.get_today_summary())

@app.route('/api/clear-history', methods=['POST'])
def api_clear_history():
    return jsonify(tracker.clear_history())

@app.route('/api/delete-meal', methods=['POST'])
def api_delete_meal():
    data = request.get_json()
    index = data.get('index')
    if index is not None and isinstance(index, int):
        result = tracker.delete_meal(index)
        return jsonify(result)
    return jsonify({'success': False, 'error': 'شاخص نامعتبر است'})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    carbs = float(data.get('carbs', 0))
    protein = float(data.get('protein', 0))
    fat = float(data.get('fat', 0))
    predicted = ai_model.predict(carbs, protein, fat)
    if predicted:
        return jsonify({'success': True, 'calories': predicted, 'confidence': 85})
    else:
        estimated = (carbs * 4) + (protein * 4) + (fat * 9)
        return jsonify({'success': True, 'calories': round(estimated, 1), 'confidence': 70})

@app.route('/api/set-goal', methods=['POST'])
def api_set_goal():
    data = request.get_json()
    return jsonify(tracker.set_goal(data.get('goal', 2000)))

@app.route('/api/train-ai', methods=['POST'])
def api_train_ai():
    return jsonify({'success': tracker.train_ai(), 'trained': ai_model.is_trained})

@app.route('/api/add-predicted-food', methods=['POST'])
def api_add_predicted_food():
    data = request.get_json()
    result = tracker.add_predicted_food(
        data.get('name'),
        float(data.get('calories', 0)),
        data.get('unit', 'واحد'),
        float(data.get('carbs', 0)),
        float(data.get('protein', 0)),
        float(data.get('fat', 0))
    )
    return jsonify(result)

@app.route('/api/weekly', methods=['GET'])
def api_weekly():
    return jsonify(tracker.get_weekly_summary())


# اجرای سرور
if __name__ == '__main__':
    PORT = 5000
    print(f" آدرس: http://localhost:{PORT}")
    app.run(debug=True, host='0.0.0.0', port=PORT)
