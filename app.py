import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import joblib
import os
import boto3
import io

# Konfiguracja strony
st.set_page_config(
    page_title="Przewidywanie czasu półmaratonu",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Załadowanie modelu
@st.cache_resource
def load_model():
    """
    Załadowanie wytrenowanego modelu półmaratonu z S3
    """
    try:
        # Konfiguracja S3
        s3 = boto3.client('s3')
        BUCKET_NAME = 'maraton'
        
        # Pobierz model z S3
        st.info("🔄 Ładowanie modelu z S3...")
        response = s3.get_object(Bucket=BUCKET_NAME, Key='models/maraton_pipeline.pkl')
        
        # Odczytaj zawartość do pamięci
        model_data = response['Body'].read()
        
        # Załaduj model z danych binarnych
        model = joblib.load(io.BytesIO(model_data))
        
        st.success("✅ Model został pomyślnie załadowany z S3!")
        return model
        
    except Exception as e:
        st.error(f"❌ Błąd podczas ładowania modelu z S3: {str(e)}")
        
        # Fallback - spróbuj załadować lokalny model
        try:
            st.info("🔄 Próba załadowania lokalnego modelu...")
            local_model_path = 'models/maraton_pipeline.pkl'
            
            if os.path.exists(local_model_path):
                model = joblib.load(local_model_path)
                st.success("✅ Model został załadowany lokalnie!")
                return model
            else:
                st.error(f"❌ Nie znaleziono lokalnego pliku modelu: {local_model_path}")
                return None
                
        except Exception as local_error:
            st.error(f"❌ Błąd podczas ładowania lokalnego modelu: {str(local_error)}")
            return None

# Zmiana czasu uzyskanego przez zawodników z formatu h:m:s, na sekundy
def convert_time_to_seconds(time):
    if pd.isnull(time) or time in ['DNS', 'DNF']:
        return None
    time = time.split(':')
    # Obsługa formatu MM:SS (2 części) lub HH:MM:SS (3 części)
    if len(time) == 2:  # MM:SS
        return int(time[0]) * 60 + int(time[1])
    elif len(time) == 3:  # HH:MM:SS
        return int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])
    else:
        return None

# Zmiana czasu uzyskanego przez zawodników z sekund na format h:m:s
def seconds_to_time(seconds):
    hours = int(seconds) // 3600
    minutes = (int(seconds) % 3600) // 60
    secs = int(seconds) % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Funkcja do konwersji czasu MM:SS na sekundy (wykorzystuje convert_time_to_seconds)
def time_format_check(time_str):
    """
    Konwertuje czas w formacie MM:SS na sekundy
    """
    if not time_str:
        return None
    
    # Sprawdzenie formatu MM:SS
    pattern = r'^([0-5]?[0-9]):([0-5][0-9])$'
    match = re.match(pattern, time_str.strip())
    
    if not match:
        return None
    
    # Użycie funkcji convert_time_to_seconds
    seconds = convert_time_to_seconds(time_str)
    
    # Sprawdzenie rozsądności czasu (między 12 a 60 minut na 5km)
    if seconds is None or seconds < 720 or seconds > 3600:  # 10-60 minut
        return None
    
    return seconds

# Funkcja do obliczenia tempa na kilometr
def calculate_pace(total_seconds, distance_km=21.0975):
    """
    Oblicza tempo na kilometr w formacie MM:SS
    """
    pace_seconds = total_seconds / distance_km
    minutes = int(pace_seconds // 60)
    seconds = int(pace_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

# Główna aplikacja
def main():
    # Tytuł i opis aplikacji
    st.title("🏃‍♂️ Przewidywanie czasu półmaratonu")
    st.markdown("""
    ### Przewiduj swój czas na półmaraton na podstawie danych treningowych
    
    Ta aplikacja wykorzystuje model uczenia maszynowego do przewidywania czasu półmaratonu 
    na podstawie Twojego wieku, płci i najlepszego czasu na 5km.
    """)
    
    # Załadowanie modelu
    model = load_model()
    
    if model is None:
        st.error("❌ Nie można załadować modelu. Sprawdź czy plik models/maraton_pipeline.pkl istnieje.")
        st.stop()
    
    # Tworzenie layoutu kolumn
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📋 Wprowadź swoje dane")
        
        # Formularz do wprowadzenia danych
        with st.form("prediction_form"):
            # Wiek
            wiek = st.number_input(
                "Wiek (lata)",
                min_value=18,
                max_value=80,
                value=30,
                step=1,
                help="Wprowadź swój wiek w latach (18-80)"
            )
            
            # Płeć
            plec_wybor = st.selectbox(
                "Płeć",
                options=["Kobieta", "Mężczyzna"],
                help="Wybierz swoją płeć"
            )
            
            # Czas na 5km
            czas_5km = st.text_input(
                "Najlepszy czas na 5km (MM:SS)",
                placeholder="np. 25:30",
                help="Wprowadź swój najlepszy czas na 5km w formacie MM:SS"
            )
            
            # Przycisk przewidywania
            predict_button = st.form_submit_button(
                "🎯 Przewiduj czas półmaratonu",
                use_container_width=True
            )
    
    with col2:
        st.markdown("### 📊 Wyniki przewidywania")
        
        if predict_button:
            # Walidacja danych
            errors = []
            
            # Sprawdzenie wieku
            if not (18 <= wiek <= 80):
                errors.append("Wiek musi być w zakresie 18-80 lat")
            
            # Sprawdzenie czasu 5km
            czas_5km_sekundy = time_format_check(czas_5km)
            if czas_5km_sekundy is None:
                errors.append("Nieprawidłowy format czasu 5km. Użyj formatu MM:SS (np. 25:30) i upewnij się, że podany czas mieści się w przedziale 12:00-59:59")
            
            # Wyświetlenie błędów lub przewidywania
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            else:
                try:
                    # Kodowanie płci (zgodnie z danymi treningowymi: 1 - kobieta, 0 - mężczyzna)
                    plec_encoded = 1 if plec_wybor == "Kobieta" else 0
                    
                    # Obliczenie tempa na kilometr dla 5km (w sekundach na kilometr)
                    tempo_5km = czas_5km_sekundy / 5 / 60  # tempo na kilometr w minutach

                    # Obliczanie współczynnika wieku na tempo
                    wiek_tempo = tempo_5km/wiek

                    # Obliczanie czasu na 5km dla kobiet i mężczyzn
                    czas_5km_k = czas_5km_sekundy if plec_encoded == 1 else 0
                    czas_5km_m = czas_5km_sekundy if plec_encoded == 0 else 1

                    # Obliczanie tempa na kilometr dla 5km dla kobiet i mężczyzn
                    tempo_5km_k = czas_5km_k / 5 / 60
                    tempo_5km_m = czas_5km_m / 5 / 60
                                        
                    # Przygotowanie danych do predykcji (zgodnie ze strukturą z demo_halfmarathon_data.csv)
                    user_data = pd.DataFrame({
                        'Wiek': [wiek],
                        'Płeć': [plec_encoded],
                        '5 km Czas': [czas_5km_sekundy], #model oczekuje sekund
                        '5 km Tempo': [tempo_5km], # tempo na kilometr w minutach
                        'WiekTempo': [wiek_tempo], # tempo na kilometr w minutach
                        '5 km Czas K': [czas_5km_k], # czas w sekundach dla kobiet
                        '5 km Czas M': [czas_5km_m], # czas w sekundach dla mężczyzn
                        '5 km Tempo K': [tempo_5km], # tempo na kilometr w minutach
                        '5 km Tempo M': [tempo_5km], # tempo na kilometr w minutach
                    })
                    
                    # Przewidywanie (model zwraca czas w sekundach)
                    przewidywany_czas_sekundy = model.predict(user_data)[0]
                    
                    # Formatowanie wyniku używając seconds_to_time
                    przewidywany_czas_formatted = seconds_to_time(przewidywany_czas_sekundy)
                    tempo = calculate_pace(przewidywany_czas_sekundy)
                    
                    # Wyświetlenie wyniku
                    st.success(f"🎉 Przewidywany czas półmaratonu: **{przewidywany_czas_formatted}**")
                    
                    # Dodatkowe informacje
                    col_tempo, col_info = st.columns(2)
                    
                    with col_tempo:
                        st.info(f"⏱️ **Tempo na kilometr:** {tempo} min/km")
                    
                    with col_info:
                        # Klasyfikacja wyniku
                        if przewidywany_czas_sekundy < 90*60:  # < 1:30:00
                            kategoria = "Świetny czas!"
                            color = "🥇"
                        elif przewidywany_czas_sekundy < 105*60:  # < 1:45:00
                            kategoria = "Bardzo dobry czas!"
                            color = "🥈"
                        elif przewidywany_czas_sekundy < 120*60:  # < 2:00:00
                            kategoria = "Dobry czas!"
                            color = "🥉"
                        else:
                            kategoria = "Kontynuuj treningi!"
                            color = "💪"
                        
                        st.info(f"{color} **{kategoria}**")
                    
                    # Dodatkowe statystyki
                    st.markdown("### 📈 Dodatkowe informacje")
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    with col_stat1:
                        st.metric(
                            label="Czas półmaratonu",
                            value=przewidywany_czas_formatted
                        )
                    
                    with col_stat2:
                        avg_speed = 21.0975 / (przewidywany_czas_sekundy / 3600)  # km/h
                        st.metric(
                            label="Średnia prędkość",
                            value=f"{avg_speed:.1f} km/h"
                        )
                    
                    with col_stat3:
                        przewidywany_czas_minuty = przewidywany_czas_sekundy / 60
                        st.metric(
                            label="Czas w minutach",
                            value=f"{przewidywany_czas_minuty:.1f} min"
                        )
                    
                    # Porady treningowe
                    st.markdown("### 💡 Porady treningowe")
                    
                    if tempo.split(':')[0] == '04':  # tempo 4:xx
                        st.info("🚀 Fantastyczne tempo! Kontynuuj intensywne treningi i pracuj nad wytrzymałością.")
                    elif tempo.split(':')[0] == '05':  # tempo 5:xx
                        st.info("👍 Dobre tempo! Dodaj więcej długich biegów i pracuj nad równomiernym tempem.")
                    else:  # tempo 6:xx i wolniejsze
                        st.info("💪 Pracuj nad poprawą tempa poprzez treningi interwałowe i stopniowe zwiększanie dystansu.")
                        
                except Exception as e:
                    st.error(f"❌ Wystąpił błąd podczas przewidywania: {str(e)}")
                    st.error("Sprawdź czy format danych jest poprawny i spróbuj ponownie.")
        else:
            st.info("👆 Wprowadź swoje dane i kliknij przycisk przewidywania")
    
    # Sekcja z informacjami o aplikacji
    st.markdown("---")
    
    with st.expander("ℹ️ Informacje o aplikacji"):
        st.markdown("""
        **Jak działa ta aplikacja?**
        
        1. **Wprowadź dane**: Wiek, płeć i najlepszy czas na 5km
        2. **Model analizuje**: Algorytm uczenia maszynowego analizuje Twoje dane
        3. **Otrzymaj przewidywanie**: Aplikacja podaje przewidywany czas półmaratonu
        
        **Uwagi:**
        - Przewidywanie jest orientacyjne i może się różnić od rzeczywistego wyniku
        - Czas zależy od wielu czynników: kondycji, pogody, trasy, strategii biegu
        - Najlepsze przewidywania dla biegaczy regularnie trenujących
        - Model został wytrenowany na rzeczywistych danych z półmaratonów
        
        **Dystans półmaratonu:** 21,0975 km (13,1 mili)
        """)

if __name__ == "__main__":
    main()
