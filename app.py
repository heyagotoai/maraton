#%%
import streamlit as st
import pandas as pd
import re
import joblib
import json
import os
import boto3
import io
import openai
from supabase import create_client, Client  # Dodano import Supabase
from dotenv import load_dotenv
from langfuse import Langfuse, observe
from langfuse.openai import OpenAI as LangfuseOpenAI
from langfuse import Langfuse

load_dotenv(".env")

# Konfiguracja strony
st.set_page_config(
    page_title="Twój czasu półmaratonu",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Funkcja do wyciągania danych z tekstu za pomocą LLM, z obserwacją
def wyciagnij_dane_z_tekstu(opis_uzytkownika):
    """
    Wyciąga dane treningowe z swobodnego tekstu użytkownika za pomocą LLM
    """
    llm_client = LangfuseOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
    )

       
    # Prompt do analizy tekstu
    system_prompt = """Jesteś ekspertem w analizie tekstu dotyczącego biegania. 
    Twoim zadaniem jest wyciągnięcie z tekstu użytkownika następujących informacji:
    1. Imię (jeśli zostało podane, w przeciwnym razie "Brak")
    2. Wiek (w latach, liczba całkowita)
    3. Płeć (Kobieta lub Mężczyzna)
    4. Czas na 5km (w formacie MM:SS)
    
    Odpowiedz w formacie JSON z następującymi kluczami:
    {
        "Imię": "[imię lub Brak]",
        "Wiek": "[liczba]",
        "Płeć": "[Kobieta/Mężczyzna]",
        "Czas na 5km": "[MM:SS]"
    }
    """

    # NIE koduj/dekoduj tekstu, po prostu użyj go bezpośrednio
    user_prompt = f"Przeanalizuj ten tekst i wyciągnij dane treningowe: {opis_uzytkownika}"

    # Wywołanie OpenAI API
    response = llm_client.chat.completions.create(
        response_format={"type": "json_object"},
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=200
    )

    dane_z_tekstu = response.choices[0].message.content
    if dane_z_tekstu:
        dane_z_tekstu = dane_z_tekstu.strip()
        try:
            dane_z_tekstu = json.loads(dane_z_tekstu)
        except:
            dane_z_tekstu = {"error": dane_z_tekstu}
    else:
        dane_z_tekstu = {"error": "Brak odpowiedzi z AI"}
    return dane_z_tekstu

@observe(name="parsuj_dane_z_ai")
def parsuj_dane_z_ai(dane_json):
    """
    Parsuje odpowiedź JSON z AI i wyciąga konkretne wartości
    """
    try:
        # dane_json jest już słownikiem, nie trzeba go parsować
        if isinstance(dane_json, dict):
            dane = {}
            
            # Parsowanie imienia
            if 'Imię' in dane_json:
                imie_str = dane_json['Imię'].strip()
                if imie_str != 'Brak' and imie_str.lower() != 'brak':
                    dane['imie'] = imie_str
            
            # Parsowanie wieku
            if 'Wiek' in dane_json:
                wiek_str = str(dane_json['Wiek']).strip()
                if wiek_str != 'Brak' and wiek_str.lower() != 'brak':
                    # Bezpieczne wyciąganie liczby z tekstu
                    liczby = re.findall(r'\d+', wiek_str)
                    if liczby:
                        dane['wiek'] = int(liczby[0])
            
            # Parsowanie płci
            if 'Płeć' in dane_json:
                plec_str = dane_json['Płeć'].strip()
                if plec_str != 'Brak' and plec_str.lower() != 'brak':
                    if 'Kobieta' in plec_str or 'kobieta' in plec_str:
                        dane['plec'] = 'Kobieta'
                    elif 'Mężczyzna' in plec_str or 'mężczyzna' in plec_str:
                        dane['plec'] = 'Mężczyzna'
            
            # Parsowanie czasu na 5km
            if 'Czas na 5km' in dane_json:
                czas_str = dane_json['Czas na 5km'].strip()
                if czas_str != 'Brak' and czas_str.lower() != 'brak':
                    # Wyciągnij format MM:SS używając regex
                    match = re.search(r'(\d{1,2}):(\d{2})', czas_str)
                    if match:
                        dane['czas_5km'] = f"{match.group(1)}:{match.group(2)}"
            
            return dane
        else:
            st.error("Nieprawidłowy format odpowiedzi z AI")
            return None
            
    except Exception as e:
        st.error(f"Błąd podczas parsowania danych z AI: {str(e)}")
        return None

# Załadowanie modelu
@st.cache_resource
def load_model():
    """
    Załadowanie wytrenowanego modelu półmaratonu z S3
    """
    # Będziemy zbierać komunikaty o błędach z poszczególnych źródeł
    error_messages = []

    try:
        # Konfiguracja S3
        s3 = boto3.client('s3')
        BUCKET_NAME = 'maraton'
        
        # Pobierz model z S3
        #st.write("🔄 Ładowanie modelu z S3")
        response = s3.get_object(Bucket=BUCKET_NAME, Key='models/maraton_pipeline.pkl')
        
        # Odczytaj zawartość do pamięci
        model_data = response['Body'].read()
        
        # Załaduj model z danych binarnych
        model = joblib.load(io.BytesIO(model_data))
        
        st.write("✅ Model został pomyślnie załadowany z S3!")
        return model
        
    except Exception as e:
        error_messages.append(f"S3: {str(e)}")
        
        # Fallback 1 – spróbuj załadować model z Supabase Storage
        try:
            SUPABASE_URL = os.getenv("SUPABASE_URL")
            SUPABASE_KEY = os.getenv("SUPABASE_KEY")

            if SUPABASE_URL and SUPABASE_KEY:
                supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

                # Domyślne wartości można nadpisać zmiennymi środowiskowymi
                SUPABASE_BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME")
                SUPABASE_MODEL_PATH = os.getenv("SUPABASE_MODEL_PATH")

                if not SUPABASE_BUCKET_NAME or not SUPABASE_MODEL_PATH:
                    raise ValueError("SUPABASE_BUCKET_NAME i SUPABASE_MODEL_PATH muszą być ustawione")

                # Pobierz plik z Supabase Storage
                response = supabase.storage.from_(SUPABASE_BUCKET_NAME).download(SUPABASE_MODEL_PATH)

                # Supabase storage download zwraca bytes
                model_bytes = response

                model = joblib.load(io.BytesIO(model_bytes))
                st.write("✅ Model został pomyślnie załadowany z Supabase Storage!")
                return model
            else:
                st.warning("⚠️ Zmiennie środowiskowe SUPABASE_URL lub SUPABASE_KEY nie są ustawione – pomijam ładowanie z Supabase.")

        except Exception as supabase_error:
            error_messages.append(f"Supabase Storage: {str(supabase_error)}")

        # Fallback 2 – spróbuj załadować lokalny model
        try:
            #st.info("🔄 Próba załadowania lokalnego modelu...")
            local_model_path = 'models/maraton_pipeline.pkl'
            
            if os.path.exists(local_model_path):
                model = joblib.load(local_model_path)
                st.write("✅ Model został załadowany lokalnie!")
                return model
            else:
                error_messages.append(f"Lokalny plik: nie znaleziono '{local_model_path}'")
                
        except Exception as local_error:
            error_messages.append(f"Lokalny odczyt: {str(local_error)}")

    # Jeśli dotarliśmy tutaj, oznacza to, że żadna z metod się nie powiodła
    st.error("❌ Nie udało się załadować modelu z żadnego źródła:\n" + "\n".join(error_messages))
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
def sprawdz_format_czasu(time_str):
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

def obliczenia(czas_5km_sekundy, wiek, plec_wybor):
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
    dane_do_predykcji = pd.DataFrame({
        'Wiek': [wiek],
        'Płeć': [plec_encoded],
        '5 km Czas': [czas_5km_sekundy], #model oczekuje sekund
        '5 km Tempo': [tempo_5km], # tempo na kilometr w minutach
        'WiekTempo': [wiek_tempo], # tempo na kilometr w minutach
        '5 km Czas K': [czas_5km_k], # czas w sekundach dla kobiet
        '5 km Czas M': [czas_5km_m], # czas w sekundach dla mężczyzn
        '5 km Tempo K': [tempo_5km_k], # tempo na kilometr w minutach
        '5 km Tempo M': [tempo_5km_m], # tempo na kilometr w minutach
    })
    return dane_do_predykcji

@observe(name="predykcja_czasu_półmaratonu")
def predykcja(dane_do_predykcji, model, imie=None):
    # Przewidywanie (model zwraca czas w sekundach)
    przewidywany_czas_sekundy = model.predict(dane_do_predykcji)[0]
    
    # Formatowanie wyniku używając seconds_to_time
    przewidywany_czas_formatted = seconds_to_time(przewidywany_czas_sekundy)
    tempo = calculate_pace(przewidywany_czas_sekundy)
    
    # Spersonalizowane powitanie
    if imie:
        powitanie = f"🎉 **{imie}**, Twój czas półmaratonu: **{przewidywany_czas_formatted}**"
    else:
        powitanie = f"🎉 Przewidywany czas półmaratonu: **{przewidywany_czas_formatted}**"
    
    # Wyświetlenie wyniku
    st.markdown(f"## {powitanie} ##")
    
    # Dodatkowe statystyki
    st.markdown("### 📈 Twoje statystyki")
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        st.metric(
        label="Tempo na kilometr",
        value=tempo
    )
    
    with col_stat2:
        srednia_predkosc = 21.0975 / (przewidywany_czas_sekundy / 3600)  # km/h
        st.metric(
            label="Średnia prędkość",
            value=f"{srednia_predkosc:.1f} km/h"
        )
    
    with col_stat3:
        przewidywany_czas_minuty = przewidywany_czas_sekundy / 60
        st.metric(
            label="Czas w minutach",
            value=f"{przewidywany_czas_minuty:.1f} min"
        )
    
    st.markdown("")
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
    
    st.metric(
        label="Kategoria",
        value=f"{color} **{kategoria}**"
    )
    
    # AI-generowane podsumowanie motywujące
    st.markdown("### 🤖 Spersonalizowane podsumowanie AI")
    with st.spinner("Generuję spersonalizowane podsumowanie..."):
        # Pobierz dane z DataFrame
        wiek_user = dane_do_predykcji['Wiek'].iloc[0]
        plec_user = "Kobieta" if dane_do_predykcji['Płeć'].iloc[0] == 1 else "Mężczyzna"
        czas_5km_user = dane_do_predykcji['5 km Czas'].iloc[0]
        
        # Generuj podsumowanie AI
        podsumowanie = generuj_motywujace_podsumowanie_ai(
            wiek_user, plec_user, czas_5km_user, przewidywany_czas_sekundy, imie
        )
        
        # Wyświetl w ładnym kontenerze
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            {podsumowanie}
        </div>
        """, unsafe_allow_html=True)

@observe(name="generuj_motywujace_podsumowanie_ai")
def generuj_motywujace_podsumowanie_ai(wiek, plec, czas_5km, przewidywany_czas, imie=None):
    """
    Generuje spersonalizowane, motywujące podsumowanie z sugestiami treningowymi
    """
    try:
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        
        # Konwersja danych na czytelny format
        czas_5km_formatted = f"{czas_5km//60}:{czas_5km%60:02d}"
        przewidywany_czas_formatted = seconds_to_time(przewidywany_czas)
        tempo_na_km = calculate_pace(przewidywany_czas)
        
        # Przygotowanie danych o użytkowniku
        dane_uzytkownika = f"""
        Imię: {imie if imie else "Biegacz"}
        Wiek: {wiek} lat
        Płeć: {plec}
        Czas na 5km: {czas_5km_formatted}
        Przewidywany czas półmaratonu: {przewidywany_czas_formatted}
        Przewidywane tempo na kilometr: {tempo_na_km} min/km
        """
        
        # Prompt dla AI
        system_prompt = f"""
        Jesteś doświadczonym trenerem biegania i motywatorem. Na podstawie danych użytkownika napisz krótkie, 
        motywujące podsumowanie (maksymalnie 150 słów) które zawiera:
        
        1. Ciepłe, zachęcające powitanie (używaj imienia jeśli zostało podane)
        2. Pozytywną ocenę obecnego poziomu biegowego
        3. Realistyczną motywację dotyczącą celu półmaratonu
        4. 2-3 konkretne, praktyczne sugestie treningowe
        5. Zachęcające zakończenie
        
        Ton: pozytywny, motywujący, profesjonalny ale przyjazny.
        Unikaj zbyt technicznych terminów. Pisz po polsku.
        Jeśli imię to "Biegacz", nie używaj go - po prostu zwracaj się bezpośrednio.
        """

        user_prompt = f"Dane biegacza: {dane_uzytkownika}"

        # Wywołanie OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,  # Wyższa temperatura dla kreatywności
            max_tokens=250
        )

        content = response.choices[0].message.content
        return content.strip() if content else ""
        
    except Exception as e:
        # Fallback - podstawowy tekst motywujący
        imie_text = f"{imie}, " if imie else ""
        return f"""
        💪 **{imie_text}świetna robota!** 
        
        Twój przewidywany czas półmaratonu to **{przewidywany_czas_formatted}** - to fantastyczny cel! 
        
        **Sugestie:**
        • Stopniowo zwiększaj dystanse długich biegów
        • Dodaj 1-2 treningi interwałowe tygodniowo  
        • Nie zapominaj o regeneracji i rozciąganiu
        
        Pamiętaj - każdy krok przybliża Cię do mety! 🏃‍♂️✨
        """

# Funkcja do analizy tekstu
def analiza_tekstu(wiek, plec_encoded, czas_5km_sekundy):
    """
    Analizuje tekst użytkownika i wyciąga dane treningowe
    """
    # Przygotowanie danych do analizy
    dane_do_predykcji = pd.DataFrame({
        'Wiek': [wiek],
        'Płeć': [plec_encoded],
        '5 km Czas': [czas_5km_sekundy], #model oczekuje sekund
    })
    return dane_do_predykcji

def generuj_prosbe_o_brakujace_dane(brakujace_dane, dane_z_ai, imie=None):
    """
    Generuje przyjazną prośbę o uzupełnienie brakujących danych
    """
    try:
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        
        # Przygotuj informacje o tym co już mamy
        posiadane_dane = []
        if imie:
            posiadane_dane.append(f"imię: {imie}")
        if 'wiek' in dane_z_ai:
            posiadane_dane.append(f"wiek: {dane_z_ai['wiek']} lat")
        if 'plec' in dane_z_ai:
            posiadane_dane.append(f"płeć: {dane_z_ai['plec']}")
        if 'czas_5km' in dane_z_ai:
            posiadane_dane.append(f"czas na 5km: {dane_z_ai['czas_5km']}")
        
        posiadane_tekst = ", ".join(posiadane_dane) if posiadane_dane else "brak danych"
        brakujace_tekst = ", ".join(brakujace_dane)
        
        # Prompt dla AI
        system_prompt = f"""
        Jesteś przyjaznym asystentem aplikacji do przewidywania czasu półmaratonu. 
        Użytkownik podał niepełne dane i potrzebujesz go poprosić o uzupełnienie w sposób:
        
        1. Ciepły i zachęcający
        2. Konkretny - wskaż dokładnie czego brakuje
        3. Pomocny - podaj przykłady jak podać dane
        4. Krótki (maksymalnie 80 słów)
        
        Użyj imienia jeśli zostało podane. Pisz po polsku.
        """

        user_prompt = f"""
        Użytkownik podał: {posiadane_tekst}
        Brakuje: {brakujace_tekst}
        
        Wygeneruj przyjazną prośbę o uzupełnienie brakujących danych.
        """

        # Wywołanie OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6,
            max_tokens=150
        )

        content = response.choices[0].message.content
        return content.strip() if content else ""
        
    except Exception as e:
        # Fallback - podstawowa prośba
        imie_text = f"{imie}, " if imie else ""
        return f"""
        📝 **{imie_text}prawie gotowe!** 
        
        Aby przewidzieć Twój czas półmaratonu, potrzebuję jeszcze: **{brakujace_tekst}**.
        
        **Przykład:** "Mam 30 lat, jestem kobietą i mój czas na 5km to 25:30"
        
        Uzupełnij brakujące informacje i spróbuj ponownie! 😊

                """
# Funkcja do logowania danych użytkownika i wyników predykcji do Langfuse
@observe(name="log_predykcji_uzytkownika")
def log_to_langfuse(dane_uzytkownika, wyniki_predykcji, wskazowki, dane_dla_ai=None):
    """
    Loguje dane użytkownika i wyniki predykcji do Langfuse dataset
    """
    try:
        # Inicjalizacja klienta Langfuse
        langfuse = Langfuse(
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        
        # Przygotowanie danych wejściowych (input)
        input_data = {
            "imie": dane_uzytkownika.get("imie", "Brak"),
            "wiek": dane_uzytkownika.get("wiek", None),
            "plec": dane_uzytkownika.get("plec", None),
            "czas_5km": dane_uzytkownika.get("czas_5km", None),
            "opis_uzytkownika": dane_uzytkownika.get("opis_oryginalny", "")
        }
        
        # Generowanie podsumowania AI jeśli dane są dostępne
        podsumowanie_ai = None
        if dane_dla_ai:
            try:
                podsumowanie_ai = generuj_motywujace_podsumowanie_ai(
                    dane_dla_ai['wiek'],
                    dane_dla_ai['plec'],
                    dane_dla_ai['czas_5km_sekundy'],
                    dane_dla_ai['przewidywany_czas_sekundy'],
                    dane_dla_ai.get('imie', None)
                )
            except Exception as e:
                print(f"Błąd generowania podsumowania AI: {str(e)}")
                podsumowanie_ai = "Nie udało się wygenerować spersonalizowanego podsumowania."
        
        # Przygotowanie danych wyjściowych (expected output)
        output_data = {
            "przewidywany_czas_polmaraton": wyniki_predykcji.get("czas_formatted", None),
            "tempo_na_km": wyniki_predykcji.get("tempo", None),
            "srednia_predkosc": wyniki_predykcji.get("predkosc", None),
            "wskazowki_treningowe": wskazowki,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Dodaj podsumowanie AI jeśli zostało wygenerowane
        if podsumowanie_ai:
            output_data["podsumowanie_ai"] = podsumowanie_ai
        
        # Dodanie danych do dataset
        dataset_name = "halfmaraton"
        
        # Utworzenie lub pobranie dataset
        try:
            dataset = langfuse.get_dataset(dataset_name)
        except:
            # Dataset nie istnieje, utwórz nowy
            dataset = langfuse.create_dataset(
                name=dataset_name,
                description="Dataset z przewidywaniami czasu półmaratonu użytkowników"
            )
        
        # Dodanie item do dataset
        dataset_item = langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input=input_data,
            expected_output=output_data,
            metadata={
                "model_version": "v1.0",
                "app_version": "streamlit_app",
                "data_source": "user_input_ai_analysis"
            }
        )
        
        return dataset_item
        
    except Exception as e:
        # Nie przerywamy działania aplikacji jeśli logowanie się nie powiedzie
        print(f"❌ Błąd logowania do Langfuse: {str(e)}")
        import traceback
        print(f"🔍 Pełny stack trace: {traceback.format_exc()}")
        return None

# Główna aplikacja
def main():

    # Tytuł i opis aplikacji
    col1, col2 = st.columns([1, 10])
    with col1:
        st.image("maraton.png", width=100)
    with col2:
        st.title("Półmaraton - predykcja czasu i statystyki")

    st.markdown("""
    ### Sprawdź swój czas półmaratonu na podstawie danych historycznych zawodników Półmaratonu Wrocławskiego 2023-2024
    """)
    
    # Załadowanie modelu
    model = load_model()
    
    if model is None:
        st.error("❌ Nie można załadować modelu. Sprawdź czy plik models/maraton_pipeline.pkl istnieje.")
        st.stop()
    
    # Tworzenie layoutu kolumn
    col1, col2 = st.columns([2, 3])
    
    with col1:
            st.markdown("### 💬 Opowiedz o sobie")

            # Formularz do wprowadzenia tekstu
            with st.form("user_text_form"):
                opis_uzytkownika = st.text_area(
                    "Potrzebuję Twoje imię, wiek, płeć i przybliżony czas na 5km",
                    height=300,
                    placeholder="Np. Nazywam się Anna, mam 30 lat, jestem kobietą i mój czas na 5km to 25:30"
                )
                
                analizuj = st.form_submit_button(
                    "🤖 Analizuj tekst i przewiduj",
                    use_container_width=True
                )

            # Wyświetlanie rozpoznanych danych pod formularzem
            if analizuj:
                # Analiza tekstu przez AI
                dane_z_ai_json = wyciagnij_dane_z_tekstu(opis_uzytkownika)
                
                # Wyświetlanie danych w czytelnej formie
                st.markdown("### 📋 Rozpoznane dane:")
                
                # Tworzenie czytelnego podsumowania danych
                if isinstance(dane_z_ai_json, dict):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        if 'Imię' in dane_z_ai_json and dane_z_ai_json['Imię'] != 'Brak':
                            st.info(f"👤 **Imię:** {dane_z_ai_json['Imię']}")
                        else:
                            st.warning("👤 **Imię:** nie podano")
                        
                        if 'Wiek' in dane_z_ai_json:
                            st.info(f"🎂 **Wiek:** {dane_z_ai_json['Wiek']} lat")
                        else:
                            st.warning("🎂 **Wiek:** nie rozpoznano")
                    
                    with col_b:
                        if 'Płeć' in dane_z_ai_json:
                            icon = "👩" if dane_z_ai_json['Płeć'] == 'Kobieta' else "👨"
                            st.info(f"{icon} **Płeć:** {dane_z_ai_json['Płeć']}")
                        else:
                            st.warning("⚧️ **Płeć:** nie rozpoznano")
                        
                        if 'Czas na 5km' in dane_z_ai_json:
                            st.info(f"🏃‍♂️ **Czas na 5km:** {dane_z_ai_json['Czas na 5km']}")
                        else:
                            st.warning("⏱️ **Czas na 5km:** nie rozpoznano")
                    
                    if st.button("🔄 Wyczyść dane i wykonaj analizę ponownie"):
                        st.rerun()
                else:
                    st.error("❌ Nie udało się rozpoznać danych z tekstu")
                    st.write("Odpowiedź AI:", dane_z_ai_json)

    with col2:
                    
        # Zmienna do śledzenia czy wyświetlono jakiekolwiek wyniki
        wyswietlono_wyniki = False
        
        if analizuj:
            # Analiza tekstu przez AI (przeniesione do lewej kolumny)
            dane_z_ai_json = wyciagnij_dane_z_tekstu(opis_uzytkownika)
            
            # Parsowanie danych z AI
            dane_z_ai = parsuj_dane_z_ai(dane_z_ai_json)
            
            if dane_z_ai is None:
                st.error("❌ Nie udało się wyciągnąć danych z tekstu. Spróbuj ponownie.")
            else:
                # Sprawdzenie jakich danych brakuje
                brakujace_dane = []
                czas_5km_sekundy = None
                
                # Sprawdzenie wieku
                if 'wiek' not in dane_z_ai or not (18 <= dane_z_ai['wiek'] <= 80):
                    brakujace_dane.append("wiek (18-80 lat)")
                
                # Sprawdzenie płci
                if 'plec' not in dane_z_ai or dane_z_ai['plec'] not in ['Kobieta', 'Mężczyzna']:
                    brakujace_dane.append("płeć (Kobieta/Mężczyzna)")
                
                # Sprawdzenie czasu 5km
                if 'czas_5km' not in dane_z_ai:
                    brakujace_dane.append("czas na 5km (w formacie MM:SS)")
                else:
                    czas_5km_sekundy = sprawdz_format_czasu(dane_z_ai['czas_5km'])
                    if czas_5km_sekundy is None:
                        brakujace_dane.append("prawidłowy czas na 5km (12:00-59:59)")
                
                # Jeśli brakuje danych, wygeneruj przyjazną prośbę
                if brakujace_dane:
                    st.markdown("### 💬 Potrzebuję więcej informacji")
                    with st.spinner("Przygotowuję spersonalizowaną prośbę..."):
                        imie_uzytkownika = dane_z_ai.get('imie', None)
                        prosba = generuj_prosbe_o_brakujace_dane(brakujace_dane, dane_z_ai, imie_uzytkownika)
                        
                        # Wyświetl prośbę w ładnym kontenerze
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
                            padding: 20px;
                            border-radius: 10px;
                            color: #333;
                            margin: 10px 0;
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                            border-left: 5px solid #ff6b6b;
                        ">
                            {prosba}
                        </div>
                        """, unsafe_allow_html=True)
                    wyswietlono_wyniki = True
                else:
                    try:
                        # Wszystkie dane są dostępne - wykonaj predykcję
                        dane_do_predykcji = obliczenia(czas_5km_sekundy, dane_z_ai['wiek'], dane_z_ai['plec'])
                        
                        # Obliczenie wyników predykcji dla logowania
                        przewidywany_czas_sekundy = model.predict(dane_do_predykcji)[0]
                        przewidywany_czas_formatted = seconds_to_time(przewidywany_czas_sekundy)
                        tempo = calculate_pace(przewidywany_czas_sekundy)
                        srednia_predkosc = 21.0975 / (przewidywany_czas_sekundy / 3600)
                        
                        # Przygotowanie danych do logowania
                        wyniki_predykcji = {
                            "czas_formatted": przewidywany_czas_formatted,
                            "tempo": tempo,
                            "predkosc": f"{srednia_predkosc:.1f} km/h"
                        }
                        
                        # Generowanie wskazówek
                        if tempo.split(':')[0] == '04':  # tempo 4:xx
                            wskazowki = "Fantastyczne tempo! Kontynuuj intensywne treningi i pracuj nad wytrzymałością."
                        elif tempo.split(':')[0] == '05':  # tempo 5:xx
                            wskazowki = "Dobre tempo! Dodaj więcej długich biegów i pracuj nad równomiernym tempem."
                        else:  # tempo 6:xx i wolniejsze
                            wskazowki = "Pracuj nad poprawą tempa poprzez treningi interwałowe i stopniowe zwiększanie dystansu."
                        
                        # Dodanie oryginalnego opisu do danych użytkownika dla logowania
                        dane_z_ai['opis_oryginalny'] = opis_uzytkownika
                        
                        # Przygotowanie danych dla podsumowania AI
                        dane_dla_ai = {
                            'wiek': dane_z_ai['wiek'],
                            'plec': dane_z_ai['plec'],
                            'czas_5km_sekundy': czas_5km_sekundy,
                            'przewidywany_czas_sekundy': przewidywany_czas_sekundy,
                            'imie': dane_z_ai.get('imie', None)
                        }
                        
                        # Logowanie do Langfuse
                        try:
                            logged_data = log_to_langfuse(dane_z_ai, wyniki_predykcji, wskazowki, dane_dla_ai)
                            if logged_data:
                                st.write("✅ Dane zostały zapisane do Langfuse!")
                            else:
                                st.warning("⚠️ Wystąpił problem z zapisem do Langfuse - sprawdź logi.")
                        except Exception as e:
                            st.error(f"❌ Błąd podczas logowania do Langfuse: {str(e)}")
                            print(f"🔍 Szczegóły błędu Langfuse: {str(e)}")
                        
                        # Wykonanie predykcji z imieniem (jeśli zostało podane)
                        imie_uzytkownika = dane_z_ai.get('imie', None)
                        predykcja(dane_do_predykcji, model, imie_uzytkownika)
                        wyswietlono_wyniki = True

                    except Exception as e:
                        st.error(f"❌ Wystąpił błąd podczas przewidywania: {str(e)}")
                        st.error("Sprawdź czy format danych jest poprawny i spróbuj ponownie.")

        # Wyświetl komunikat pomocniczy tylko jeśli nie wyświetlono żadnych wyników
        if not wyswietlono_wyniki and not analizuj: #and not przewiduj:
            st.info("👈Wprowadź swoje dane i kliknij przycisk przewidywania")
    
    # Sekcja z informacjami o aplikacji
    st.markdown("---")
    
    with st.expander("ℹ️ Informacje o aplikacji"):
        st.markdown("""
        **Jak działa ta aplikacja?**
        
        1. **Wprowadź dane**: Imię, wiek, płeć i czas na 5km
        2. **Model analizuje**: Algorytm uczenia maszynowego analizuje Twoje dane
        3. **Otrzymaj przewidywanie**: Aplikacja podaje przewidywany czas półmaratonu
        
        **Uwagi:**
        - Przewidywanie jest orientacyjne i może się różnić od rzeczywistego wyniku
        - Czas zależy od wielu czynników: kondycji, pogody, trasy, strategii biegu
        - Najlepsze przewidywania dla biegaczy regularnie trenujących
        - Model został wytrenowany na rzeczywistych danych Półmaratonu Wrocławskiego z lat 2023 i 2024
        
        **Dystans półmaratonu:** 21,0975 km (13,1 mili)
        """)

if __name__ == "__main__":
    main()
# %%
