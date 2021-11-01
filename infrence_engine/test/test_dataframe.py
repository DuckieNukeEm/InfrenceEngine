import pandas as pd
import numpy as np


def make_test_dataframe(
    Categorical: bool = False,
    Time_Diff: bool = True,
    Complex: bool = False,
    cols: list = None,
):
    """reads in and proper orients the test dataframe

    Argument:
        Categorical {bool} -- Make City a cateogrical variables instead of a string
        time_diff {bool} -- Make Hire_Age a timedelta64 instead of all 0
        complex {bool} -- adds a complex field Imaginar = Dob + i * Hieght

    Details:
        this returns a small test dataframe with the following properties:
            Records 1 ("Bob") has an outlier year compared to all of Dob with 2 STD
            Record 1 ("BoB") isnt' an outlier when compared to Year, when compared to Tokyo with 2 STD
            Record 2 ("Carl") has an outlier hieght compared to all of Hieght with 2  STD
            Record 2 ("Carl") is an outlier for Dob and Hieght when compared to Tokyo with 2 STD
            Record 7 ("Igor") is an outlier in Hieght when compared to Tokyo with 2 STD

    """
    age_list = [
        [
            "Bob",
            1901,
            180,
            "Tokyo",
            True,
            19.25,
            "2010-12-14 12:00:52",
            "1901-10-19",
            1,
        ],
        ["Carl", 1993, 92, "Edo", True, 7.25, "2017-01-01 13:44:12", "1993-06-04", 1],
        [
            "Derrel",
            1954,
            170,
            "Osaka",
            False,
            5.25,
            "2014-04-02 06:02:33",
            "1954-10-02",
            0,
        ],
        [
            "Ed",
            1953,
            185,
            "Tokyo",
            False,
            10.25,
            "2001-09-11 03:44:22",
            "1953-10-25",
            0,
        ],
        [
            "Frank",
            1954,
            178,
            "Tokyo",
            True,
            10.3333,
            "2009-07-02 18:42:00",
            "1954-01-01",
            1,
        ],
        [
            "Garth",
            1953,
            179,
            "Edo",
            False,
            11.99,
            "2020-02-29 22:10:11",
            "1953-02-28",
            0,
        ],
        [
            "Hank",
            1953,
            212,
            "Tokyo",
            False,
            16.4,
            "2019-02-28 11:57:02",
            "1953-11-28",
            0,
        ],
        [
            "Igor",
            1952,
            149,
            "Osaka",
            True,
            16.45,
            "2015-10-31 00:00:00",
            "1952-12-31",
            1,
        ],
    ]
    # creating a pandas dataframe
    Df = pd.DataFrame(
        age_list,
        columns=[
            "Name",
            "DoB",
            "Hieght",
            "City",
            "Married",
            "PayRate",
            "Hire_DateTime",
            "Birth_Date",
            "Flag",
        ],
    )

    Df["Hire_Age"] = 0
    Df["Imaginary"] = np.nan
    Df["Married"] = Df["Married"].astype(bool)
    Df["Hire_DateTime"] = pd.to_datetime(
        Df["Hire_DateTime"], format="%Y-%m-%d %H:%M:%S"
    )
    Df["Birth_Date"] = pd.to_datetime(Df["Birth_Date"], format="%Y-%m-%d")

    if Categorical is True:
        Df["City"] = pd.Categorical(Df["City"])

    if Complex is True:
        Df["Imaginary"] = Df.apply(lambda x: complex(x.DoB, x.Hieght), axis=1)

    if Time_Diff is True:
        Df["Hire_Age"] = Df["Birth_Date"] - Df["Hire_DateTime"]

    if cols is None:
        return Df
    else:
        return Df[cols]
