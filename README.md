# Aplikacja przewidywania czasu półmaratonu

## Opis
Aplikacja Streamlit do przewidywania czasu półmaratonu na podstawie wieku, płci i czasu na 5km.
Wykorzystuje wytrenowany model uczenia maszynowego.

## Instalacja

1. Zainstaluj wymagane biblioteki:
```bash
pip install -r requirements.txt
```

2. Upewnij się, że model `maraton_pipeline.pkl` znajduje się w katalogu `models/`

## Uruchomienie

```bash
streamlit run apptest.py
```

## Funkcjonalności

- ✅ Walidacja danych wejściowych
- ✅ Przewidywanie czasu półmaratonu
- ✅ Obliczanie tempa na kilometr
- ✅ Klasyfikacja poziomu biegacza
- ✅ Dodatkowe statystyki (prędkość, czas w minutach)
- ✅ Porady treningowe
- ✅ Polski interfejs użytkownika

## Format danych

**Dane wejściowe:**
- Wiek: 18-80 lat
- Płeć: Kobieta/Mężczyzna
- Czas 5km: Format MM:SS (np. 25:30)

**Dane wyjściowe:**
- Przewidywany czas półmaratonu (HH:MM:SS)
- Tempo na kilometr (MM:SS)
- Średnia prędkość (km/h)
- Klasyfikacja poziomu

## Uwagi

- Model został wytrenowany na rzeczywistych danych z półmaratonów
- Przewidywania są orientacyjne
- Najlepsze wyniki dla regularnie trenujących biegaczy 