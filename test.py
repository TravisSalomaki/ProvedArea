from ProvedArea import ProvedArea
import pytest

headers = 'ComboCurve Header Example.csv'
headers2 = '/Users/travissalomaki/Desktop/MissingLatLongs.csv'
forecast_parameters = 'ComboCurve Forecast Parameter Example.csv'

def test_invalid_realization_input_string():
    with pytest.raises(ValueError):
        obj = ProvedArea('10', headers, headers2)

def test_invalid_realization_input_negative_number():
    with pytest.raises(ValueError):
        obj = ProvedArea(-1, headers, headers2)