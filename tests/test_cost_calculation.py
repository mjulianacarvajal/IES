import pandas as pd

from src.preprocessing import create_total_cost


def test_cost_calculation():

    df = pd.DataFrame({
        "Tuition_USD": [10000],
        "Duration_Years": [2],
        "Rent_USD": [500],
        "Insurance_USD": [1000],
        "Visa_Fee_USD": [200],
    })

    result = create_total_cost(df)

    expected = (
        10000 * 2
        + 500 * 12 * 2
        + 1000 * 2
        + 200
    )

    assert (
        result["Costo_Total"].iloc[0]
        == expected
    )