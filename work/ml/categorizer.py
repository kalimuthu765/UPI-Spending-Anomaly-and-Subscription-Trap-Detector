"""
Guess a spending category from the shop name when the statement has none
(Google Pay / PhonePe / bank PDFs usually only give the payee). Rules are
plain keyword matches, checked in order, so the result is predictable and
easy to extend. Anything not matched stays "Other" rather than a wrong guess.
"""
import re

_RULES = [
    ("Fuel", ["petrol", "diesel", "indian oil", "indianoil", "iocl", "bharat petroleum", "bpcl", "hpcl",
              "hp pay", "shell", "fuel", "filling station"]),
    ("Entertainment", ["netflix", "spotify", "hotstar", "prime video", "amazon prime", "youtube", "bookmyshow",
                       "pvr", "inox", "zee5", "sonyliv", "jiocinema", "gaming", "steam", "apple music",
                       "movie", "cinema", "theatre", "theater"]),
    ("Bills", ["jio", "airtel", "vodafone", "bsnl", "vi prepaid", "vi postpaid", "broadband", "recharge",
               "prepaid", "postpaid", "electricity", "tneb", "tangedco", "bescom", "water bill", "gas bill",
               "lpg", "indane", "bharat gas", "dth", "tata play", "tatasky", "act fibernet", "bill", "insurance",
               "lic", "rent", "emi", "google one", "icloud", "subscription", "chatgpt", "openai", "microsoft",
               "adobe", "linkedin", "notion"]),
    ("Travel", ["uber", "ola", "rapido", "metro", "irctc", "railway", "redbus", "makemytrip", "goibibo",
                "yatra", "indigo", "air india", "flight", "cab", "taxi", "bus", "fastag", "toll", "auto ride"]),
    ("Health", ["pharmacy", "pharmacies", "pharma", "medical", "medicals", "medicine", "medicines", "chemist",
                "druggist", "drug store", "doctor", "dr", "scan", "scans", "hospitals", "clinics", "medplus", "apollo", "hospital", "clinic", "1mg", "pharmeasy",
                "diagnostic", "lab", "dental", "gym", "fitness", "cult.fit", "cultfit"]),
    ("Education", ["college", "school", "tuition", "fees", "udemy", "coursera", "stationery", "xerox",
                   "photocopy", "library", "exam", "book store", "bookstore", "university"]),
    ("Food", ["swiggy", "zomato", "restaurant", "hotel", "cafe", "coffee", "tea", "idli", "dosa", "biryani",
              "bakery", "pizza", "burger", "kfc", "mcdonald", "domino", "starbucks", "anjappar", "saravana bhavan",
              "juice", "sweets", "mess", "canteen", "dhaba", "eatery", "food", "kitchen", "meals", "tiffin"]),
    ("Groceries", ["reliance smart", "bazaar", "bazar", "dmart", "d-mart", "supermarket", "super market", "grocer",
                   "bigbasket", "big basket", "blinkit", "zepto", "instamart", "milk", "aavin", "provision",
                   "vegetable", "fruits", "kirana", "more retail", "nature's basket", "mart"]),
    ("Shopping", ["amazon", "flipkart", "myntra", "croma", "reliance digital", "apple", "ajio", "nykaa",
                  "meesho", "decathlon", "electronics", "mobile", "pothys", "lifestyle", "max fashion", "zudio",
                  "westside", "clothing", "textile", "jewel", "fashion", "footwear", "store", "stores", "shop"]),
]
_COMPILED = [(cat, re.compile(r"\b(?:" + "|".join(re.escape(k) for k in kws) + r")\b", re.I))
             for cat, kws in _RULES]

# shop names that usually mean a recurring plan (used by subscription detection)
SUBSCRIPTION_HINTS = re.compile(
    r"\b(?:netflix|spotify|hotstar|prime|youtube|google one|icloud|apple music|zee5|sonyliv|jiocinema|jio|airtel|"
    r"vodafone|bsnl|broadband|recharge|prepaid|postpaid|dth|tata play|insurance|lic|sip|mutual fund|rent|emi|"
    r"linkedin|chatgpt|openai|notion|adobe|microsoft|gym|cult|subscription|membership|premium)\b", re.I)

# categories that are everyday habits, not plans, unless the name says otherwise
HABIT_CATEGORIES = {"Fuel", "Groceries", "Food", "Travel", "Shopping"}


# Common Indian first names: a lone word that is one of these is treated as a person.
_FIRST_NAMES = set("""rahul rohit rajesh ramesh suresh mahesh dinesh ganesh karthik karthi kumar kiran arun arjun
ajay vijay sanjay ravi raja raju murugan senthil saravanan selvam selvi priya divya kavya anitha anita sunita
sunil anil ashok ashwin vignesh vicky vishnu sathish satish prakash prabhu pradeep praveen manoj mohan mohamed
muhammad ahmed ali imran irfan farhan faisal sneha swathi swati pooja puja deepa deepak deepika meena meera
mani manikandan lakshmi laxmi latha lavanya nithya nandhini harini hari harish gopal gokul govind ilango
jagan jayanthi jeeva kalai kamal kannan krishna kishore madhan naveen naresh nikhil nitin pavan ponnu rajan
ranjith sakthi sangeetha santhosh shankar sharma shiva siva subash surya tamil thiru vasanth venkat yogesh
amma appa anna akka thambi""".split())
_BUSINESS_WORD = re.compile(
    r"\b(?:ltd|pvt|private|limited|llp|inc|corp|company|co|enterprises|traders|trading|agency|agencies|services|"
    r"service|solutions|technologies|tech|systems|bank|upi|neft|imps|atm|pos|payment|payments|wallet|bill|"
    r"store|stores|shop|mart|centre|center|hub|club|school|college|hospital|hotel|restaurant|works|industries|"
    r"foods|studio|labs|group|india|online|retail|supermarket|stationery|salon|parlour|parlor|garage|motors)\b", re.I)


def looks_like_person(merchant):
    """True for a name that reads like a private person (Rahul, Priya K,
    S Murugan, Mr Kumar). A guess from the name alone: unusual shop names made
    of two plain words can be mistaken for people."""
    name = re.sub(r"^(?:mr|mrs|ms|dr|shri|smt)\.?\s+", "", str(merchant or "").strip(), flags=re.I)
    if not name or _BUSINESS_WORD.search(name) or not re.fullmatch(r"[A-Za-z][A-Za-z.\s]{1,40}", name):
        return False
    words = name.replace(".", " ").split()
    if not 1 <= len(words) <= 3 or any(len(w) > 15 for w in words):
        return False
    if len(words) == 1:
        return words[0].lower() in _FIRST_NAMES
    return True


def guess_category(merchant):
    name = str(merchant or "")
    for cat, rx in _COMPILED:
        if rx.search(name):
            return cat
    if looks_like_person(name):
        return "Person"
    return "Other"


def fill_categories(merchants, categories):
    """Keep a real category when the statement gave one; guess only when it is
    missing / 'Other' / 'Unknown'."""
    out = []
    for m, c in zip(merchants, categories):
        c = None if c is None else str(c).strip()
        if not c or c.lower() in ("other", "unknown", "nan", "none", "misc", "miscellaneous", "uncategorized"):
            out.append(guess_category(m))
        else:
            out.append(c)
    return out
