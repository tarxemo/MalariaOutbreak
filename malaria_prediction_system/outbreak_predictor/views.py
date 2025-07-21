from django.shortcuts import render, redirect
from django.views import View
from .models import  MalariaPrediction
import pandas as pd
import joblib
from datetime import datetime
import numpy as np
import math

class PredictOutbreakView(View):
    template_name = 'predict.html'

    def get(self, request):
        districts = District.objects.all()
        return render(request, self.template_name, {'districts': districts})

    def post(self, request):
        # Load trained model & scaler
        model = joblib.load('malaria_outbreak_model.pkl')
        scaler = joblib.load('malaria_scaler.pkl')
        
        # Get form data
        district_id = request.POST.get('district')
        week = int(request.POST.get('week'))
        year = int(request.POST.get('year'))
        
        # Fetch district data
        district = District.objects.get(id=district_id)
        
        # Get historical data for this district (last 12 weeks ideally)
        historical_data = HistoricalData.objects.filter(
            district=district
        ).order_by('-year', '-week')[:12]  # Get up to 12 weeks of history
        
        # Prepare core input data
        input_data = {
            'District': district.name,
            'Year': year,
            'Week': week,
            'Humidity': float(request.POST.get('humidity')),
            'Rainfall': float(request.POST.get('rainfall')),
            'Min Temperature': float(request.POST.get('min_temp')),
            'Max Temperature': float(request.POST.get('max_temp')),
            'Effective_Net_Usage': float(request.POST.get('net_usage')),
            'IRS_Coverage': float(request.POST.get('irs_coverage')),
            'average_latitude': district.average_latitude,
            'average_longitude': district.average_longitude,
            'prop_female': district.prop_female,
            'prop_male': district.prop_male,
            'prop_<5_age': district.prop_under_5,
            'prop_5_14_age': district.prop_5_14,
            'prop_15_29_age': district.prop_15_29,
            'prop_30_44_age': district.prop_30_44,
            'prop_45+_age': district.prop_45_plus,
            'is_public': district.is_public,
            'is_private': district.is_private,
            'is_index': district.is_index,
            'is_other_member': district.is_other_member,
            'occ_student': district.occ_student,
            'occ_watchman': district.occ_watchman,
            'occ_housewife': district.occ_housewife,
            'occ_fisherman': district.occ_fisherman,
            'occ_farmer': district.occ_farmer,
        }
        
        # Initialize all expected features with default values (0 or appropriate)
        with open('selected_features.txt', 'r') as f:
            all_features = [line.strip() for line in f.readlines()]
        
        for feature in all_features:
            if feature not in input_data:
                # Set default values for different feature types
                if 'lag_' in feature or 'rolling_' in feature:
                    input_data[feature] = 0.0  # Default for historical features
                elif feature in ['threshold', 'Temperature_Range']:
                    continue  # Will be calculated later
                else:
                    input_data[feature] = 0.0
        
        # Add historical malaria cases if available
        malaria_history = []
        for i, hist in enumerate(historical_data, start=1):
            malaria_history.append(hist.malaria_cases)
            input_data[f'true_malaria_cases_lag_{i}'] = hist.malaria_cases
            input_data[f'Rainfall_lag_{i}'] = hist.rainfall
            input_data[f'Humidity_lag_{i}'] = hist.humidity
            input_data[f'Max Temperature_lag_{i}'] = hist.max_temp
            input_data[f'Min Temperature_lag_{i}'] = hist.min_temp
            input_data[f'Effective_Net_Usage_lag_{i}'] = hist.net_usage
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Feature engineering
        input_df['Temperature_Range'] = input_df['Max Temperature'] - input_df['Min Temperature']
        input_df['Rainfall_transformed'] = np.log1p(input_df['Rainfall'])
        input_df['Humidity_transformed'] = np.log1p(input_df['Humidity'])
        
        # Date features
        input_df['month_sin'] = np.sin(2 * np.pi * (week % 52) / 52)
        input_df['month_cos'] = np.cos(2 * np.pi * (week % 52) / 52)
        
        # Calculate rolling statistics based on available history
        for window in [4, 8, 12]:
            if len(malaria_history) >= window:
                # Calculate with actual history
                window_history = malaria_history[:window]
                input_df[f'true_malaria_cases_rolling_mean_{window}'] = np.mean(window_history)
                input_df[f'true_malaria_cases_rolling_std_{window}'] = np.std(window_history)
            else:
                # Use default values if not enough history
                input_df[f'true_malaria_cases_rolling_mean_{window}'] = 0.0
                input_df[f'true_malaria_cases_rolling_std_{window}'] = 0.0
        
        # Calculate threshold (use 0 if no history)
        if len(malaria_history) >= 12:
            input_df['threshold'] = input_df['true_malaria_cases_rolling_mean_12'] * 1.5
        else:
            input_df['threshold'] = 0.0
        
        # Add other derived features
        input_df['4wk_avg'] = input_df['true_malaria_cases_rolling_mean_4']
        input_df['12wk_moving_avg'] = input_df['true_malaria_cases_rolling_mean_12']
        input_df['4wk_std'] = input_df['true_malaria_cases_rolling_std_4']
        
        # Ensure all required features are present and fill any remaining NAs
        for feature in all_features:
            if feature not in input_df.columns:
                input_df[feature] = 0.0
        
        X = input_df[all_features].fillna(0)
        X_scaled = scaler.transform(X)
        
        # Predict
        proba = model.predict_proba(X_scaled)[0][1] * 4 * 100
        prediction = proba >= 30
        
        # Save result and historical data
        PredictionResult.objects.create(
            district=district,
            outbreak_probability=proba,
            prediction=prediction,
            week=week,
            year=year
        )
        
        HistoricalData.objects.create(
            district=district,
            year=year,
            week=week,
            malaria_cases=0,  # Actual cases would be updated later
            rainfall=input_data['Rainfall'],
            humidity=input_data['Humidity'],
            min_temp=input_data['Min Temperature'],
            max_temp=input_data['Max Temperature'],
            net_usage=input_data['Effective_Net_Usage']
        )
        
        return render(request, 'result.html', {
            'district': district.name,
            'probability': f"{proba:.2f}%",
            'outbreak': "HIGH RISK" if prediction else "LOW RISK",
            'date': datetime.now().strftime("%Y-%m-%d"),
            'week': week,
            'year': year
        })

class DashboardView(View):
    template_name = 'dashboard.html'

    def get(self, request):
        predictions = PredictionResult.objects.all().order_by('-year', '-week')
        return render(request, self.template_name, {'predictions': predictions})
    
    

import numpy as np
import pandas as pd
from django.shortcuts import render
import joblib

# Load model
model = joblib.load('model.pkl')  # e.g., saved from sklearn
le = joblib.load(f"encoder.pkl")

def predict_outbreak(request):
    if request.method == 'POST':
        data = request.POST
        import datetime

        # Get the submitted date string
        date_str = request.POST.get("date")  # e.g. '2025-07-20'
        if date_str:
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()

                year = date_obj.year
                month = date_obj.month
                week = date_obj.isocalendar()[1]  # ISO week number

                # Now you can use year, month, and week in your logic or save them
                print("Year:", year, "Month:", month, "Week:", week)

            except ValueError:
                # Handle invalid date format
                print("Invalid date format received:", date_str)
        # Numerical inputs
        inputs = {
            "District": data["District"],
            "average_latitude": float(data["average_latitude"]),
            "average_longitude": float(data["average_longitude"]),
            "Year": year,
            "Month": month,
            "Week": week,
            "Humidity": float(data["Humidity"]),
            "Rainfall": float(data["Rainfall"]),
            "Min Temperature": float(data["Min_Temperature"]),
            "Max Temperature": float(data["Max_Temperature"]),
            "temp_range": float(data["Max_Temperature"]) - float(data["Min_Temperature"]),
            "temp_humidity": (float(data["Max_Temperature"]) + float(data["Min_Temperature"])) / 2 * float(data["Humidity"]),
            "Effective_Net_Usage": float(data["Effective_Net_Usage"]),
            "IRS_Coverage": float(data["IRS_Coverage"]),
        }

        # One-hot normalized fields
        def one_hot(keys, selected):
            return {k: 1 if k == selected else 0 for k in keys}

        # Gender proportions
        gender = data["gender"]
        inputs.update(one_hot(["prop_male", "prop_female"], f"prop_{gender}"))

        # Age groups
        age_group = data["age_group"]
        inputs.update(one_hot([
            "prop_<5_age", "prop_5_14_age", "prop_15_29_age", "prop_30_44_age", "prop_45+_age"
        ], f"prop_{age_group}_age"))

        # Member type
        member_type = data["member_type"]
        inputs.update(one_hot([
            "is_index", "is_additional_index", "is_other_member"
        ], f"is_{member_type}"))

        # Facility type
        facility = data["facility_type"]
        inputs.update(one_hot(["is_private", "is_public"], f"is_{facility}"))

        # Occupation
        # occupation = data["occupation"]
        # inputs.update(one_hot([
        #     "occ_student", "occ_health_worker", "occ_police_officer", "occ_watchman",
        #     "occ_farmer", "occ_fisherman", "occ_unemployed", "occ_housewife", "occ_business_owner"
        # ], f"occ_{occupation}"))

        # Order features
        feature_order = [  # Same as your feature list
            "District", "average_latitude", "average_longitude", "Year", "Month", "Week",
            "Humidity", "Rainfall", "Min Temperature", "Max Temperature",
            "temp_range", "Effective_Net_Usage", "IRS_Coverage"
        ]

        # Encode District as categorical numeric if model expects numeric
        try:
            inputs["District"] = le.transform([inputs["District"]])[0]
        except ValueError:
            # Handle unseen district (not in training data)
            print(f"Unknown district: {inputs['District']}")
            # You can decide how to handle unknowns, e.g., assign a default, raise error, or use a special index
            inputs["District"] = -1
        print(inputs)
        input_df = pd.DataFrame([inputs])[feature_order]
        # Make prediction
        prediction = model.predict_proba(input_df)[0][1]
        print(prediction)
        malaria_prediction = MalariaPrediction(
                district=data["District"],
                year=year,
                month=month,
                week=week,
                humidity = float(data["Humidity"]),
                rainfall = float(data["Rainfall"]),
                min_temperature = float(data["Min_Temperature"]),
                max_temperature = float(data["Max_Temperature"]),
                prediction=(prediction  + 0.3)*100,
            )
        malaria_prediction.save()
        return render(request, "predict_form.html", {"prediction": malaria_prediction})

    return render(request, "predict_form.html")

def view_predictions(request):
    predictions = MalariaPrediction.objects.all().order_by('-date_predicted')
    
    # Filter by district if provided
    district = request.GET.get('district')
    if district:
        predictions = predictions.filter(district=district)
    
    # Filter by date range if provided
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    if start_date and end_date:
        predictions = predictions.filter(
            date_predicted__range=[start_date, end_date]
        )
    
    context = {
        'predictions': predictions,
        'district_choices': MalariaPrediction.DISTRICT_CHOICES,
    }
    return render(request, 'view_predictions.html', context)