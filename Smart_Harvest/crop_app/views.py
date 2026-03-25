from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
import joblib
import numpy as np
import os

from .models import Prediction


# Loads ML Model when server starts


MODEL_PATH   = os.path.join(settings.ML_MODEL_PATH, 'best_model.pkl')
SCALER_PATH  = os.path.join(settings.ML_MODEL_PATH, 'scaler.pkl')
ENCODER_PATH = os.path.join(settings.ML_MODEL_PATH, 'label_encoder.pkl')
RESULTS_PATH = os.path.join(settings.ML_MODEL_PATH, 'results.txt')

try:
    ml_model     = joblib.load(MODEL_PATH)
    ml_scaler    = joblib.load(SCALER_PATH)
    ml_encoder   = joblib.load(ENCODER_PATH)
    model_loaded = True
    print("✅ ML Model loaded successfully!")
except:
    model_loaded = False
    print("⚠️  ML Model not found! Run: python ml_model/train_model.py")


# info about each crop


crop_info = {
    "rice":        {"emoji": "🌾", "season": "Kharif",    "water": "High",   "tip": "Needs flooded fields and warm climate."},
    "maize":       {"emoji": "🌽", "season": "Kharif",    "water": "Medium", "tip": "Grows well in warm weather with good drainage."},
    "chickpea":    {"emoji": "🫘", "season": "Rabi",      "water": "Low",    "tip": "Cool and dry conditions are ideal."},
    "kidneybeans": {"emoji": "🫘", "season": "Kharif",    "water": "Medium", "tip": "Needs moderate temperature and well-drained soil."},
    "mungbean":    {"emoji": "🫘", "season": "Kharif",    "water": "Low",    "tip": "Quick growing crop, good for hot weather."},
    "blackgram":   {"emoji": "🫘", "season": "Kharif",    "water": "Low",    "tip": "Thrives in warm humid conditions."},
    "lentil":      {"emoji": "🫘", "season": "Rabi",      "water": "Low",    "tip": "Cool season crop, needs well-drained soil."},
    "banana":      {"emoji": "🍌", "season": "Year-round","water": "High",   "tip": "Tropical crop needing lots of water and warmth."},
    "mango":       {"emoji": "🥭", "season": "Summer",    "water": "Medium", "tip": "Loves hot and dry weather during flowering."},
    "grapes":      {"emoji": "🍇", "season": "Rabi",      "water": "Medium", "tip": "Needs well-drained soil and moderate climate."},
    "watermelon":  {"emoji": "🍉", "season": "Zaid",      "water": "Medium", "tip": "Hot weather and sandy soil work best."},
    "apple":       {"emoji": "🍎", "season": "Winter",    "water": "Medium", "tip": "Needs cold winters for good fruit set."},
    "orange":      {"emoji": "🍊", "season": "Winter",    "water": "Medium", "tip": "Subtropical climate with mild frost works well."},
    "papaya":      {"emoji": "🍑", "season": "Year-round","water": "Medium", "tip": "Fast growing tropical fruit. Avoid waterlogging."},
    "coconut":     {"emoji": "🥥", "season": "Year-round","water": "Medium", "tip": "Coastal areas with sandy loam soil are ideal."},
    "cotton":      {"emoji": "🌸", "season": "Kharif",    "water": "High",   "tip": "Warm climate with black soil is preferred."},
    "jute":        {"emoji": "🌿", "season": "Kharif",    "water": "High",   "tip": "Warm humid climate with alluvial soil is best."},
    "coffee":      {"emoji": "☕", "season": "Year-round","water": "Medium", "tip": "Shade-loving crop needing hilly terrain."},
    "pomegranate": {"emoji": "🍎", "season": "Year-round","water": "Low",    "tip": "Drought resistant, good for dry regions."},
    "mothbeans":   {"emoji": "🌱", "season": "Kharif",    "water": "Low",    "tip": "Very drought tolerant, grows in arid zones."},
    "pigeonpeas":  {"emoji": "🌿", "season": "Kharif",    "water": "Low",    "tip": "Deep roots make it drought tolerant."},
    "muskmelon":   {"emoji": "🍈", "season": "Zaid",      "water": "Medium", "tip": "Loves hot dry weather and sandy loam soil."},
}



# Auth Views (HTML pages)


def login_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Invalid username or password.')
    return render(request, 'crop_app/login.html')


def register_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    if request.method == 'POST':
        username  = request.POST.get('username')
        password  = request.POST.get('password')
        password2 = request.POST.get('password2')
        if password != password2:
            messages.error(request, 'Passwords do not match.')
        elif User.objects.filter(username=username).exists():
            messages.error(request, 'Username already taken.')
        else:
            User.objects.create_user(username=username, password=password)
            messages.success(request, 'Account created! Please log in.')
            return redirect('login')
    return render(request, 'crop_app/register.html')


def logout_view(request):
    logout(request)
    return redirect('login')



# Page Views


def home(request):
    recent_predictions = Prediction.objects.all().order_by('-created_at')[:5]
    return render(request, 'crop_app/home.html', {'recent': recent_predictions})


def predict(request):
    if request.method == 'POST':
        if not model_loaded:
            messages.error(request, 'ML Model not ready! Please run train_model.py first.')
            return redirect('home')

        N    = request.POST['nitrogen']
        P    = request.POST['phosphorus']
        K    = request.POST['potassium']
        temp = request.POST['temperature']
        hum  = request.POST['humidity']
        ph   = request.POST['ph']
        rain = request.POST['rainfall']

        try:
            N    = float(N)
            P    = float(P)
            K    = float(K)
            temp = float(temp)
            hum  = float(hum)
            ph   = float(ph)
            rain = float(rain)
        except ValueError:
            messages.error(request, 'Please enter valid numbers in all fields.')
            return redirect('home')

        input_data         = np.array([[N, P, K, temp, hum, ph, rain]])
        input_scaled       = ml_scaler.transform(input_data)
        prediction_encoded = ml_model.predict(input_scaled)
        crop_name          = ml_encoder.inverse_transform(prediction_encoded)[0]

       
        confidence = 0
        if hasattr(ml_model, 'predict_proba'):
            proba = ml_model.predict_proba(input_scaled)
            confidence = round(np.max(proba) * 100, 2)

        Prediction(
            nitrogen=N, phosphorus=P, potassium=K,
            temperature=temp, humidity=hum, ph=ph, rainfall=rain,
            crop_name=crop_name, confidence=confidence
        ).save()

        info = crop_info.get(crop_name, {"emoji": "🌱", "season": "N/A", "water": "N/A", "tip": ""})

        return render(request, 'crop_app/result.html', {
            'crop': crop_name,
            'confidence': confidence,
            'emoji': info['emoji'],
            'season': info['season'],
            'water': info['water'],
            'tip': info['tip'],
            'N': N, 'P': P, 'K': K,
            'temp': temp, 'hum': hum, 'ph': ph, 'rain': rain,
        })
    else:
        return redirect('home')


def history(request):
    all_predictions = Prediction.objects.all().order_by('-created_at')
    return render(request, 'crop_app/history.html', {'predictions': all_predictions})


def about(request):
    model_results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, 'r') as f:
            model_results = f.read()
    return render(request, 'crop_app/about.html', {'results': model_results})



# Prediction API — GET (list) + POST (create)


@api_view(['GET', 'POST'])
def api_history(request):

    # GET — return all predictions
    if request.method == 'GET':
        predictions = Prediction.objects.all().order_by('-created_at')
        data = []
        for p in predictions:
            data.append({
                'id':          p.id,
                'crop':        p.crop_name,
                'confidence':  p.confidence,
                'nitrogen':    p.nitrogen,
                'phosphorus':  p.phosphorus,
                'potassium':   p.potassium,
                'temperature': p.temperature,
                'humidity':    p.humidity,
                'ph':          p.ph,
                'rainfall':    p.rainfall,
                'date':        p.created_at.strftime('%d %b %Y %H:%M'),
            })
        return Response(data)

    # POST — manually add a prediction record
    if request.method == 'POST':
        try:
            N    = float(request.data['nitrogen'])
            P    = float(request.data['phosphorus'])
            K    = float(request.data['potassium'])
            temp = float(request.data['temperature'])
            hum  = float(request.data['humidity'])
            ph   = float(request.data['ph'])
            rain = float(request.data['rainfall'])
        except (KeyError, ValueError):
            return Response({'error': 'Please send all 7 fields with valid numbers.'}, status=400)

        crop = request.data.get('crop_name', 'unknown')
        conf = request.data.get('confidence', None)

        p = Prediction.objects.create(
            nitrogen=N, phosphorus=P, potassium=K,
            temperature=temp, humidity=hum, ph=ph, rainfall=rain,
            crop_name=crop, confidence=conf
        )
        return Response({
            'message':    'Prediction saved!',
            'id':         p.id,
            'crop':       p.crop_name,
            'confidence': p.confidence,
        }, status=201)



# Prediction API — GET (single) + PUT (update)


@api_view(['GET', 'PUT'])
def api_history_detail(request, pk):
    try:
        p = Prediction.objects.get(pk=pk)
    except Prediction.DoesNotExist:
        return Response({'error': f'Prediction with id {pk} not found.'}, status=404)

    # GET — return single prediction
    if request.method == 'GET':
        return Response({
            'id':          p.id,
            'crop':        p.crop_name,
            'confidence':  p.confidence,
            'nitrogen':    p.nitrogen,
            'phosphorus':  p.phosphorus,
            'potassium':   p.potassium,
            'temperature': p.temperature,
            'humidity':    p.humidity,
            'ph':          p.ph,
            'rainfall':    p.rainfall,
            'date':        p.created_at.strftime('%d %b %Y %H:%M'),
        })

    # PUT — update a prediction record
    if request.method == 'PUT':
        p.nitrogen    = float(request.data.get('nitrogen',    p.nitrogen))
        p.phosphorus  = float(request.data.get('phosphorus',  p.phosphorus))
        p.potassium   = float(request.data.get('potassium',   p.potassium))
        p.temperature = float(request.data.get('temperature', p.temperature))
        p.humidity    = float(request.data.get('humidity',    p.humidity))
        p.ph          = float(request.data.get('ph',          p.ph))
        p.rainfall    = float(request.data.get('rainfall',    p.rainfall))
        p.crop_name   = request.data.get('crop_name',         p.crop_name)
        p.confidence  = request.data.get('confidence',        p.confidence)
        p.save()
        return Response({
            'message':    'Prediction updated!',
            'id':         p.id,
            'crop':       p.crop_name,
            'confidence': p.confidence,
        })



# ML Predict API — POST only (runs the ML model)


@api_view(['POST'])
def api_predict(request):
    if not model_loaded:
        return Response({'error': 'ML Model not ready. Run train_model.py first.'}, status=503)

    try:
        N    = float(request.data['nitrogen'])
        P    = float(request.data['phosphorus'])
        K    = float(request.data['potassium'])
        temp = float(request.data['temperature'])
        hum  = float(request.data['humidity'])
        ph   = float(request.data['ph'])
        rain = float(request.data['rainfall'])
    except (KeyError, ValueError):
        return Response({'error': 'Please send all 7 fields with valid numbers.'}, status=400)

    input_data   = np.array([[N, P, K, temp, hum, ph, rain]])
    input_scaled = ml_scaler.transform(input_data)
    pred_encoded = ml_model.predict(input_scaled)
    crop_name    = ml_encoder.inverse_transform(pred_encoded)[0]

 
    confidence = 0
    if hasattr(ml_model, 'predict_proba'):
        proba = ml_model.predict_proba(input_scaled)
        confidence = round(np.max(proba) * 100, 2)

    Prediction.objects.create(
        nitrogen=N, phosphorus=P, potassium=K,
        temperature=temp, humidity=hum, ph=ph, rainfall=rain,
        crop_name=crop_name, confidence=confidence
    )

    info = crop_info.get(crop_name, {"emoji": "🌱", "season": "N/A", "water": "N/A", "tip": ""})

    return Response({
        'crop': crop_name,
        'confidence': confidence,
        'emoji': info['emoji'],
        'season': info['season'],
        'water': info['water'],
        'tip': info['tip'],
    })


@api_view(['GET'])
def api_model_info(request):
    if not os.path.exists(RESULTS_PATH):
        return Response({'error': 'No results yet. Run train_model.py first.'}, status=404)

    result_data = {'all_results': {}}
    with open(RESULTS_PATH, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith('Best Model:'):
                result_data['best_model'] = line.replace('Best Model:', '').strip()
            elif line.startswith('Best Accuracy:'):
                result_data['best_accuracy'] = line.replace('Best Accuracy:', '').strip()
            elif ':' in line and '%' in line and not line.startswith('Best'):
                parts = line.split(':')
                if len(parts) == 2:
                    name = parts[0].strip()
                    acc  = parts[1].strip().replace('%', '')
                    try:
                        result_data['all_results'][name] = float(acc)
                    except ValueError:
                        pass
    return Response(result_data)


# User API — GET (list) + POST (create)


@api_view(['GET', 'POST'])
def api_users(request):

    # GET — return all users
    if request.method == 'GET':
        users = User.objects.all().order_by('date_joined')
        data = []
        for u in users:
            data.append({
                'id':          u.id,
                'username':    u.username,
                'email':       u.email,
                'is_staff':    u.is_staff,
                'date_joined': u.date_joined.strftime('%d %b %Y %H:%M'),
                'last_login':  u.last_login.strftime('%d %b %Y %H:%M') if u.last_login else None,
            })
        return Response(data)

    # POST — create a new user
    if request.method == 'POST':
        username = request.data.get('username')
        password = request.data.get('password')
        email    = request.data.get('email', '')

        if not username or not password:
            return Response({'error': 'username and password are required.'}, status=400)
        if User.objects.filter(username=username).exists():
            return Response({'error': 'Username already taken.'}, status=400)

        user = User.objects.create_user(username=username, password=password, email=email)
        return Response({
            'message':  'User created successfully!',
            'id':       user.id,
            'username': user.username,
            'email':    user.email,
        }, status=201)



# User API — GET (single) + PUT (update)


@api_view(['GET', 'PUT'])
def api_user_detail(request, pk):
    try:
        u = User.objects.get(pk=pk)
    except User.DoesNotExist:
        return Response({'error': f'User with id {pk} not found.'}, status=404)

    # GET — return single user
    if request.method == 'GET':
        return Response({
            'id':          u.id,
            'username':    u.username,
            'email':       u.email,
            'is_staff':    u.is_staff,
            'date_joined': u.date_joined.strftime('%d %b %Y %H:%M'),
            'last_login':  u.last_login.strftime('%d %b %Y %H:%M') if u.last_login else None,
        })

    # PUT — update user email or username
    if request.method == 'PUT':
        u.username = request.data.get('username', u.username)
        u.email    = request.data.get('email',    u.email)
        if request.data.get('password'):
            u.set_password(request.data['password'])
        u.save()
        return Response({
            'message':  'User updated!',
            'id':       u.id,
            'username': u.username,
            'email':    u.email,
        })



# Register + Login API


@api_view(['POST'])
def api_register(request):
    username = request.data.get('username')
    password = request.data.get('password')
    email    = request.data.get('email', '')

    if not username or not password:
        return Response({'error': 'username and password are required.'}, status=400)
    if User.objects.filter(username=username).exists():
        return Response({'error': 'Username already taken.'}, status=400)

    user = User.objects.create_user(username=username, password=password, email=email)
    return Response({
        'message':  'User created successfully!',
        'id':       user.id,
        'username': user.username,
        'email':    user.email,
    }, status=201)


@api_view(['POST'])
def api_login(request):
    username = request.data.get('username')
    password = request.data.get('password')

    if not username or not password:
        return Response({'error': 'username and password are required.'}, status=400)

    user = authenticate(request, username=username, password=password)
    if user is not None:
        return Response({
            'message':  'Login successful!',
            'id':       user.id,
            'username': user.username,
            'email':    user.email,
            'is_staff': user.is_staff,
        })
    else:
        return Response({'error': 'Invalid username or password.'}, status=401)