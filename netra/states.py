"""
netra.states
============

Lookup table for the first two letters of an Indian registration number.

Indian plates follow ``SS DD XX NNNN`` where ``SS`` is the state / union
territory code. The dashboard uses this table to show *where* a vehicle is
registered, and the post-processor uses it to decide whether a reading is
plausible (an unknown state code lowers confidence in the reading).

Legacy codes that still appear on older vehicles (OR, UA, DN) are included.
"""

STATE_CODES: dict[str, str] = {
    "AN": "Andaman and Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Dadra and Nagar Haveli and Daman and Diu",
    "DN": "Dadra and Nagar Haveli (legacy)",
    "DL": "Delhi",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha (legacy)",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "TS": "Telangana",
    "TG": "Telangana",
    "UK": "Uttarakhand",
    "UA": "Uttarakhand (legacy)",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}


def state_name(code: str) -> str | None:
    """Return the state / UT name for a two-letter code, or None if unknown."""
    return STATE_CODES.get(code.upper()[:2]) if code else None
