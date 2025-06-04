import streamlit as st
import pandas as pd
import re
import joblib
import os
import boto3
import io
import openai
from dotenv import load_dotenv

load_dotenv(".env")

# Konfiguracja strony
st.set_page_config(
    page_title="Przewidywanie czasu półmaratonu",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Funkcja do wyciągania danych z tekstu za pomocą LLM
def wyciagnij_dane_z_tekstu(opis_uzytkownika):
    """
    Wyciąga dane treningowe z swobodnego tekstu użytkownika za pomocą LLM
    """
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )


       
    # Prompt do analizy tekstu
    system_prompt = """Jesteś ekspertem w analizie tekstu dotyczącego biegania. 
    Twoim zadaniem jest wyciągnięcie z tekstu użytkownika następujących informacji:
    1. Imię (jeśli zostało podane, w przeciwnym razie "Brak")
    2. Wiek (w latach, liczba całkowita)
    3. Płeć (Kobieta lub Mężczyzna)
    4. Czas na 5km (w formacie MM:SS)
    
    Odpowiedz w dokładnie takim formacie:
    Imię: [imię lub Brak]
    Wiek: [liczba]
    Płeć: [Kobieta/Mężczyzna]  
    Czas na 5km: [MM:SS]
    """

    # NIE koduj/dekoduj tekstu, po prostu użyj go bezpośrednio
    user_prompt = f"Przeanalizuj ten tekst i wyciągnij dane treningowe: {opis_uzytkownika}"

    # Wywołanie OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=200
    )

    dane_z_tekstu = response.choices[0].message.content.strip()
    return dane_z_tekstu

def parsuj_dane_z_ai(tekst_ai):
    """
    Parsuje odpowiedź z AI i wyciąga konkretne wartości
    """
    try:
        lines = tekst_ai.strip().split('\n')
        dane = {}
        
        for line in lines:
            if 'Imię:' in line:
                imie_str = line.split('Imię:')[1].strip()
                if imie_str != 'Brak' and imie_str.lower() != 'brak':
                    dane['imie'] = imie_str
            elif 'Wiek:' in line:
                wiek_str = line.split('Wiek:')[1].strip()
                if wiek_str != 'Brak' and wiek_str.lower() != 'brak':
                    # Bezpieczne wyciąganie liczby z tekstu
                    liczby = re.findall(r'\d+', wiek_str)
                    if liczby:
                        dane['wiek'] = int(liczby[0])
            elif 'Płeć:' in line:
                plec_str = line.split('Płeć:')[1].strip()
                if plec_str != 'Brak' and plec_str.lower() != 'brak':
                    if 'Kobieta' in plec_str or 'kobieta' in plec_str:
                        dane['plec'] = 'Kobieta'
                    elif 'Mężczyzna' in plec_str or 'mężczyzna' in plec_str:
                        dane['plec'] = 'Mężczyzna'
            elif 'Czas na 5km:' in line:
                czas_str = line.split('Czas na 5km:')[1].strip()
                if czas_str != 'Brak' and czas_str.lower() != 'brak':
                    # Wyciągnij format MM:SS używając regex
                    match = re.search(r'(\d{1,2}):(\d{2})', czas_str)
                    if match:
                        dane['czas_5km'] = f"{match.group(1)}:{match.group(2)}"
        
        return dane
    except Exception as e:
        st.error(f"Błąd podczas parsowania danych z AI: {str(e)}")
        return None

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

def predykcja(dane_do_predykcji, model, imie=None):
    # Przewidywanie (model zwraca czas w sekundach)
    przewidywany_czas_sekundy = model.predict(dane_do_predykcji)[0]
    
    # Formatowanie wyniku używając seconds_to_time
    przewidywany_czas_formatted = seconds_to_time(przewidywany_czas_sekundy)
    tempo = calculate_pace(przewidywany_czas_sekundy)
    
    # Spersonalizowane powitanie
    if imie:
        powitanie = f"🎉 **{imie}**, Twój przewidywany czas półmaratonu: **{przewidywany_czas_formatted}**"
    else:
        powitanie = f"🎉 Przewidywany czas półmaratonu: **{przewidywany_czas_formatted}**"
    
    # Wyświetlenie wyniku
    st.success(powitanie)
    
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
    
    # Spersonalizowane porady treningowe
    st.markdown("### 💡 Porady treningowe")
    
    # Podstawa porad
    if tempo.split(':')[0] == '04':  # tempo 4:xx
        podstawowa_porada = "🚀 Fantastyczne tempo! Kontynuuj intensywne treningi i pracuj nad wytrzymałością."
    elif tempo.split(':')[0] == '05':  # tempo 5:xx
        podstawowa_porada = "👍 Dobre tempo! Dodaj więcej długich biegów i pracuj nad równomiernym tempem."
    else:  # tempo 6:xx i wolniejsze
        podstawowa_porada = "💪 Pracuj nad poprawą tempa poprzez treningi interwałowe i stopniowe zwiększanie dystansu."
    
    # Spersonalizowana porada
    if imie:
        spersonalizowana_porada = f"**{imie}**, {podstawowa_porada.lower()}"
        st.info(spersonalizowana_porada)
    else:
        st.info(podstawowa_porada)
    
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

        return response.choices[0].message.content.strip()
        
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

        return response.choices[0].message.content.strip()
        
    except Exception as e:
        # Fallback - podstawowa prośba
        imie_text = f"{imie}, " if imie else ""
        return f"""
        📝 **{imie_text}prawie gotowe!** 
        
        Aby przewidzieć Twój czas półmaratonu, potrzebuję jeszcze: **{brakujace_tekst}**.
        
        **Przykład:** "Mam 30 lat, jestem kobietą i mój czas na 5km to 25:30"
        
        Uzupełnij brakujące informacje i spróbuj ponownie! 😊
        """

# Główna aplikacja
def main():

    # Tytuł i opis aplikacji
    st.title("🏃‍♂️ Przewidywanie czasu półmaratonu")
    st.markdown("""
    ### Przewiduj swój czas na półmaraton na podstawie danych treningowych
    
    Ta aplikacja wykorzystuje model uczenia maszynowego do przewidywania czasu półmaratonu 
    na podstawie Twojego wieku, płci i czasu na 5km.
    """)
    
    # Załadowanie modelu
    model = load_model()
    
    if model is None:
        st.error("❌ Nie można załadować modelu. Sprawdź czy plik models/maraton_pipeline.pkl istnieje.")
        st.stop()
    
    # Tworzenie layoutu kolumn
    col1, col2 = st.columns([2, 3])
    
    # with col1:
    #     st.markdown("### 📋 Wprowadź swoje dane")
        
    #     # Formularz do wprowadzenia danych
    #     with st.form("prediction_form"):
    #         # Wiek
    #         wiek = st.number_input(
    #             "Wiek (lata)",
    #             min_value=18,
    #             max_value=80,
    #             value=30,
    #             step=1,
    #             help="Wprowadź swój wiek w latach (18-80)"
    #         )
            
    #         # Płeć
    #         plec_wybor = st.selectbox(
    #             "Płeć",
    #             options=["Kobieta", "Mężczyzna"],
    #             help="Wybierz swoją płeć"
    #         )
            
    #         # Czas na 5km
    #         czas_5km = st.text_input(
    #             "Czas na 5km (MM:SS)",
    #             placeholder="np. 25:30",
    #             help="Wprowadź swój czas na 5km w formacie MM:SS"
    #         )
            
    #         # Przycisk przewidywania
    #         przewiduj = st.form_submit_button(
    #             "🎯 Przewiduj czas półmaratonu",
    #             use_container_width=True
    #         )
    with col1:
        st.markdown("### 💬 Opowiedz o sobie")

        # Formularz do wprowadzenia tekstu
        with st.form("user_text_form"):
            opis_uzytkownika = st.text_area(
                "Podaj swoje imię, wiek, płeć i czas na 5km",
                height=300,
                placeholder="Np. Nazywam się Anna, mam 30 lat, jestem kobietą i mój czas na 5km to 25:30"
            )
            
            analizuj = st.form_submit_button(
                "🤖 Analizuj tekst i przewiduj",
                use_container_width=True
            )

    with col2:
        st.markdown("### 📊 Wyniki przewidywania")
        
        # Zmienna do śledzenia czy wyświetlono jakiekolwiek wyniki
        wyswietlono_wyniki = False
        
        if analizuj:
            # Analiza tekstu przez AI
            dane_z_ai_tekst = wyciagnij_dane_z_tekstu(opis_uzytkownika)
            st.write("**Twoje dane:**")
            st.write(dane_z_ai_tekst)
            
            # Parsowanie danych z AI
            dane_z_ai = parsuj_dane_z_ai(dane_z_ai_tekst)
            
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
                        
                        # Wykonanie predykcji z imieniem (jeśli zostało podane)
                        imie_uzytkownika = dane_z_ai.get('imie', None)
                        predykcja(dane_do_predykcji, model, imie_uzytkownika)
                        wyswietlono_wyniki = True

                    except Exception as e:
                        st.error(f"❌ Wystąpił błąd podczas przewidywania: {str(e)}")
                        st.error("Sprawdź czy format danych jest poprawny i spróbuj ponownie.")

        # if przewiduj:
        #     # Walidacja danych
        #     errors = []
            
        #     # Sprawdzenie wieku
        #     if not (18 <= wiek <= 80):
        #         errors.append("Wiek musi być w zakresie 18-80 lat")
            
        #     # Sprawdzenie czasu 5km
        #     czas_5km_sekundy = sprawdz_format_czasu(czas_5km)
        #     if czas_5km_sekundy is None:
        #         errors.append("Nieprawidłowy format czasu 5km. Użyj formatu MM:SS (np. 25:30) i upewnij się, że podany czas mieści się w przedziale 12:00-59:59")
            
        #     # Wyświetlenie błędów lub przewidywania
        #     if errors:
        #         for error in errors:
        #             st.error(f"❌ {error}")
        #     else:
        #         try:
        #             # Przygotowanie danych do predykcji
        #             dane_do_predykcji = obliczenia(czas_5km_sekundy, wiek, plec_wybor)
                    
        #             # Wykonanie predykcji
        #             predykcja(dane_do_predykcji, model)
        #             wyswietlono_wyniki = True

        #         except Exception as e:
        #             st.error(f"❌ Wystąpił błąd podczas przewidywania: {str(e)}")
        #             st.error("Sprawdź czy format danych jest poprawny i spróbuj ponownie.")
        
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